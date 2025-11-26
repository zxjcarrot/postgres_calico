/*
 * calico.c
 *
 * Two-level indirection directory for PostgreSQL buffer mapping
 * with strict equivalence to hashtable-based semantics.
 *
 * Key invariants (aligned with hashtable path):
 *  - For a given BufferTag, we ALWAYS choose the SAME buffer-mapping partition
 *    lock as the original hashtable path:
 *        newHash = BufTableHashCode(&tag);
 *        plock   = BufMappingPartitionLock(newHash);
 *  - The hit path validates BufferTag AFTER PinBuffer() while still holding
 *    the partition lock (mirrors hashtable timing protection).
 *  - The miss path publishes the mapping under the SAME partition lock,
 *    using CAS on the directory entry, then sets BufferDesc->tag/state.
 *  - Invalidate paths compute the partition lock from the *tag* as above.
 *  - Prefetch NEVER creates directory structures (find-only semantics).
 *
 * This keeps concurrency semantics identical to the original hashtable-based
 * buffer mapping, preventing zero-page / wrong-page exposure.
 */

#include "postgres.h"

#include "access/hash.h"
#include "access/xact.h"
#include "catalog/pg_class_d.h"
#include "miscadmin.h"
#include "port/atomics.h"
#include "storage/buf_internals.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "storage/ipc.h"
#include "storage/lwlock.h"
#include "storage/pg_shmem.h"
#include "storage/shmem.h"
#include "utils/guc.h"
#include "utils/hsearch.h"

/* For IOContext / strategies / smgrprefetch */
#include "storage/smgr.h"
#include <sys/mman.h>
/* -------------------------------------------------------------------------
 *                    Calico: Two-Level Indirection Directory
 * ------------------------------------------------------------------------- */

/*
 * CalicoTranslationEntry (64-bit atomic word)
 * [ 31:0 ]  - frame id (BufferDesc->buf_id), 0 = invalid
 * [ 55:32 ] - version (24-bit)
 * [ 63:56 ] - state
 */
typedef pg_atomic_uint64 CalicoTranslationEntry;

#define CALICO_FRAMEID_MASK     ((uint64)0x00000000FFFFFFFFULL)
#define CALICO_VERSION_MASK     ((uint64)0x00FFFFFF00000000ULL)
#define CALICO_STATE_MASK       ((uint64)0xFF00000000000000ULL)
#define CALICO_VERSION_SHIFT    32
#define CALICO_STATE_SHIFT      56

/* 
 * State bit flags (8-bit state field):
 * Bit 0: VALID flag - indicates entry has a valid mapping
 * Bit 1: LOCKED flag - indicates buffer has exclusive content lock
 * 
 * Valid states:
 * - 0x00 (EVICTED): No valid mapping
 * - 0x01 (VALID): Valid mapping, not locked
 * - 0x03 (VALID|LOCKED): Valid mapping with exclusive lock
 * 
 * Invariants:
 * - EVICTED (0) is mutually exclusive with VALID and LOCKED
 * - LOCKED can only be set when VALID is set
 */
#define CALICO_STATE_VALID      0x01  /* Bit 0: Valid mapping exists */
#define CALICO_STATE_LOCKED     0x02  /* Bit 1: Buffer has exclusive lock */
#define CALICO_STATE_EVICTED    0x00  /* No bits set: evicted state */

static inline uint64
CalicoPackEntry(uint32 frame_id, uint32 version, uint8 state)
{
    return ((uint64) state << CALICO_STATE_SHIFT) |
           ((uint64) version << CALICO_VERSION_SHIFT) |
           ((uint64) frame_id);
}
static inline uint32 CalicoGetFrameId(uint64 entry) { return (uint32)(entry & CALICO_FRAMEID_MASK); }
static inline uint32 CalicoGetVersion(uint64 entry) { return (uint32)((entry & CALICO_VERSION_MASK) >> CALICO_VERSION_SHIFT); }
static inline uint8  CalicoGetState  (uint64 entry) { return (uint8 )((entry & CALICO_STATE_MASK)   >> CALICO_STATE_SHIFT); }
static inline bool   CalicoIsValid   (uint64 entry) { return (CalicoGetState(entry) & CALICO_STATE_VALID) != 0; }
static inline bool   CalicoIsLocked  (uint64 entry) { return (CalicoGetState(entry) & CALICO_STATE_LOCKED) != 0; }

/* split of BufferDesc->state bits (kept for clarity, mirrors buf_internals.h) */
#define BUF_FLAG_MASK           (~(PG_UINT32_MAX >> 14)) /* Bits 18-31 */
#define BUF_REFC_MASK           (PG_UINT32_MAX >> 14)    /* Bits 0-17 */

/* GUC: total shared memory reserved for second-level tables */
static int calico_second_level_total_size_mb;

/* request extra shmem early */
static void calico_shmem_request_hook(void);
static void
calico_shmem_request_hook(void)
{
    static bool requested = false;

    if (requested || calico_second_level_total_size_mb <= 0)
        return;

    /* Request space for second-level tables */
    RequestAddinShmemSpace((Size)calico_second_level_total_size_mb * 1024L * 1024L);
    
    /* Request space for dedicated partition locks */
    RequestAddinShmemSpace((Size)NUM_BUFFER_PARTITIONS * sizeof(LWLockPadded));
    
    requested = true;
}

/* Two-level directory: top-level maps relation_id -> pointer-to-L2 table */
typedef CalicoTranslationEntry *SecondLevelTable;

/*
 * We store L2-table pointers inside a pg_atomic_uint64 array (as uint64).
 * Pointers are published once and never freed.
 */
static pg_atomic_uint64 *CalicoTopLevelTable;

#define MAX_RELATION_IDS        (1U << 14) /* ~1M relation forks */
#define MAX_BLOCKS_PER_REL      (1U << 24) /* up to 16M blocks per rel-fork */

/* -------------------------------------------------------------------------
 *          Relation-fork → compact RelationID (partitioned hash)
 * ------------------------------------------------------------------------- */

#define NUM_CALICO_RELMAP_PARTITIONS 128

typedef struct
{
    Oid             spcOid;
    Oid             dbOid;
    RelFileNumber   relNumber;
    ForkNumber      forkNum;
} CalicoRelFork;

typedef struct CalicoRelMapEntry
{
    CalicoRelFork   key;        /* must be first */
    uint32          relation_id;
} CalicoRelMapEntry;

static HTAB              *CalicoRelMap;
static pg_atomic_uint32  *CalicoNextRelationID;

/*
 * Separate LWLock array for Calico buffer mapping partitions.
 * This is independent from MainLWLockArray and provides dedicated locks
 * for the Calico directory operations.
 */
static LWLockPadded *CalicoPartitionLocks = NULL;

#define NUM_CALICO_PARTITIONS NUM_BUFFER_PARTITIONS

/*
 * BufMappingPartitionLock2 returns a partition lock from the Calico-specific
 * lock array. The partitioning logic mirrors the original buffer mapping
 * to maintain equivalent concurrency semantics.
 */
static inline LWLock *
BufMappingPartitionLock2(uint32 hashcode)
{
    return &CalicoPartitionLocks[BufTableHashPartition(hashcode)].lock;
}

/* -------------------------------------------------------------------------
 *                         Shared Memory Initialization
 * ------------------------------------------------------------------------- */

static volatile bool CalicoReady = false;

bool Calico_IsReady(void)
{
    return CalicoReady;
}

void
CalicoShmemInit(void)
{
    HASHCTL hash_ctl;
    bool    found;

    /* 1) relation-fork → relation_id shared hash (partitioned) */
    MemSet(&hash_ctl, 0, sizeof(hash_ctl));
    hash_ctl.keysize        = sizeof(CalicoRelFork);
    hash_ctl.entrysize      = sizeof(CalicoRelMapEntry);
    hash_ctl.num_partitions = NUM_CALICO_RELMAP_PARTITIONS;

    CalicoRelMap = ShmemInitHash("Calico Relation Map",
                                 4096,                /* initial */
                                 MAX_RELATION_IDS,      /* max   */
                                 &hash_ctl,
                                 HASH_ELEM | HASH_BLOBS | HASH_PARTITION);

    /* 2) next relation-id counter */
    CalicoNextRelationID = (pg_atomic_uint32 *) ShmemAlloc(sizeof(pg_atomic_uint32));
    pg_atomic_init_u32(CalicoNextRelationID, 1); /* 0 reserved */

    /* 3) top-level table (array of atomic u64 that carry pointers as uint64) */
    CalicoTopLevelTable = (pg_atomic_uint64 *)
        ShmemInitStruct("Calico Top-Level Table",
                        MAX_RELATION_IDS * sizeof(pg_atomic_uint64),
                        &found);
    if (!found)
    {
        /* Initialize all slots atomically to 0 */
        // for (uint32 i = 0; i < MAX_RELATION_IDS; i++)
        //     pg_atomic_init_u64(&CalicoTopLevelTable[i], 0);
        // madvise huge page
        madvise(CalicoTopLevelTable,
                 MAX_RELATION_IDS * sizeof(pg_atomic_uint64),
                 MADV_NOHUGEPAGE);
    }

    /* 4) Dedicated partition locks for Calico directory operations */
    CalicoPartitionLocks = (LWLockPadded *)
        ShmemInitStruct("Calico Partition Locks",
                        NUM_CALICO_PARTITIONS * sizeof(LWLockPadded),
                        &found);
    if (!found)
    {
        /* Initialize all partition locks with LWTRANCHE_BUFFER_MAPPING tranche */
        for (uint32 i = 0; i < NUM_CALICO_PARTITIONS; i++)
            LWLockInitialize(&CalicoPartitionLocks[i].lock, LWTRANCHE_BUFFER_MAPPING);
    }

    /*
     * Keep the original shared buffer hash table allocated as well.
     * Some core code paths assume it exists even if we don't use it
     * for the fast path. This avoids null deref / unexpected asserts.
     */
    //InitBufTable(NBuffers + NUM_BUFFER_PARTITIONS);
    // Use a tiny hashtable to reduce memory usage on Calico
    InitBufTable(128 + NUM_BUFFER_PARTITIONS);
    CalicoReady = true;
}

/* GUC init */
void
CalicoGucInit(void)
{
    DefineCustomIntVariable("calico.second_level_total_size_mb",
                            "Total shared memory size to reserve for Calico L2 tables (MB).",
                            NULL,
                            &calico_second_level_total_size_mb,
                            1024,   /* default 1GB */
                            0,
                            INT_MAX,
                            PGC_SIGHUP,
                            0,
                            NULL, NULL, NULL);

    shmem_request_hook = calico_shmem_request_hook;
}

/* -------------------------------------------------------------------------
 *      RelationID helpers: create-or-find and find-only (no create)
 * ------------------------------------------------------------------------- */

/*
 * Thread-Local Storage (TLS) Cache for Translation Path
 * 
 * Each thread caches the most recently accessed translation path to eliminate
 * repeated top-level lookups. The cache stores:
 * 1. The last accessed BufferTag (spcOid, dbOid, relNumber, forkNum)
 * 2. The corresponding relation_id
 * 3. The corresponding last-level translation table pointer
 * 
 * On each page access, the thread first checks if the current BufferTag's
 * high bits (relation identity) match the cached value. On a cache hit,
 * the thread directly uses the cached last-level table pointer.
 */
typedef struct CalicoTLSCache
{
    /* Cached BufferTag components (relation identity) */
    Oid             cached_spcOid;
    Oid             cached_dbOid;
    RelFileNumber   cached_relNumber;
    ForkNumber      cached_forkNum;
    
    /* Cached translation path */
    uint32          cached_relation_id;
    SecondLevelTable cached_l2_table;
    
    /* Validity flag */
    bool            valid;
} CalicoTLSCache;

/* Thread-local storage for translation path cache */
static __thread CalicoTLSCache tls_cache = {0};

/*
 * CalicoTLSCacheLookup - Check if the translation path is cached
 * 
 * Returns true if the BufferTag matches the cached entry, and sets
 * out_relation_id and out_l2_table to the cached values.
 */
static inline bool
CalicoTLSCacheLookup(Oid spcOid, Oid dbOid, RelFileNumber relNumber, ForkNumber forkNum,
                     uint32 *out_relation_id, SecondLevelTable *out_l2_table)
{
    if (!tls_cache.valid)
        return false;
    
    /* Check if BufferTag matches cached value */
    if (tls_cache.cached_spcOid == spcOid &&
        tls_cache.cached_dbOid == dbOid &&
        tls_cache.cached_relNumber == relNumber &&
        tls_cache.cached_forkNum == forkNum)
    {
        *out_relation_id = tls_cache.cached_relation_id;
        *out_l2_table = tls_cache.cached_l2_table;
        return true;
    }
    
    return false;
}

/*
 * CalicoTLSCacheUpdate - Update the thread-local cache
 * 
 * Stores the translation path for the given BufferTag.
 */
static inline void
CalicoTLSCacheUpdate(Oid spcOid, Oid dbOid, RelFileNumber relNumber, ForkNumber forkNum,
                     uint32 relation_id, SecondLevelTable l2_table)
{
    tls_cache.cached_spcOid = spcOid;
    tls_cache.cached_dbOid = dbOid;
    tls_cache.cached_relNumber = relNumber;
    tls_cache.cached_forkNum = forkNum;
    tls_cache.cached_relation_id = relation_id;
    tls_cache.cached_l2_table = l2_table;
    tls_cache.valid = true;
}

/* create-or-find - internal function */
static uint32
CalicoGetRelationIDInternal(Oid spcOid, Oid dbOid, RelFileNumber relNumber, ForkNumber forkNum)
{
    CalicoRelFork key;
    bool found;
    CalicoRelMapEntry *entry;
    uint32 rid;
    SecondLevelTable dummy_l2;

    /* TLS cache first - check if this relation is cached */
    if (CalicoTLSCacheLookup(spcOid, dbOid, relNumber, forkNum, &rid, &dummy_l2))
        return rid;

    key.spcOid    = spcOid;
    key.dbOid     = dbOid;
    key.relNumber = relNumber;
    key.forkNum   = forkNum;

    /* shared hash (double-checked, partition-locked) */
    uint32 hashv = get_hash_value(CalicoRelMap, (const void *)&key);
    LWLock *plock = BufMappingPartitionLock2(hashv);

    LWLockAcquire(plock, LW_SHARED);
    entry = (CalicoRelMapEntry *) hash_search_with_hash_value(CalicoRelMap,
                                                              (const void *)&key,
                                                              hashv,
                                                              HASH_FIND,
                                                              &found);
    if (found)
    {
        rid = entry->relation_id;
        LWLockRelease(plock);
        
        /* Note: We don't update TLS cache here because we don't have L2 table yet.
         * The L2 table will be fetched and cached in CalicoLookupAndPin. */
        return rid;
    }

    LWLockRelease(plock);
    LWLockAcquire(plock, LW_EXCLUSIVE);

    entry = (CalicoRelMapEntry *) hash_search_with_hash_value(CalicoRelMap,
                                                              (const void *)&key,
                                                              hashv,
                                                              HASH_ENTER,
                                                              &found);
    if (!found)
    {
        uint32 new_id = pg_atomic_fetch_add_u32(CalicoNextRelationID, 1);
        if (new_id >= MAX_RELATION_IDS)
        {
            LWLockRelease(plock);
            ereport(ERROR, (errmsg("maximum number of Calico relation IDs exceeded")));
        }
        entry->relation_id = new_id;
    }
    rid = entry->relation_id;
    LWLockRelease(plock);

    return rid;
}

/*
 * CalicoGetRelationID - Public wrapper for getting relation ID
 * 
 * This exposes the internal CalicoGetRelationID functionality through the
 * BufferManagerAPI interface. Allows callers to query and cache relation IDs
 * to avoid repeated lookups.
 * 
 * Returns true on success, false if the relation ID cannot be determined.
 */
bool
CalicoGetRelationID(SMgrRelation smgr, ForkNumber forkNum, uint32 *relationIdOut)
{
    if (!smgr || !smgr->smgr_rlocator.locator.relNumber || !relationIdOut)
        return false;

    *relationIdOut = CalicoGetRelationIDInternal(smgr->smgr_rlocator.locator.spcOid,
                                                   smgr->smgr_rlocator.locator.dbOid,
                                                   smgr->smgr_rlocator.locator.relNumber,
                                                   forkNum);
    return true;
}

/*
 * CalicoSetBufferLocked - Set or clear the LOCKED bit in buffer state
 * 
 * This function manages the LOCKED bit in the translation entry state:
 * - When locked=true: Sets LOCKED bit (state becomes VALID|LOCKED)
 * - When locked=false: Clears LOCKED bit and bumps version (state becomes VALID)
 * 
 * Uses partition lock to ensure atomicity with buffer mapping operations.
 * 
 * State transitions:
 * - Lock:   VALID (0x01) -> VALID|LOCKED (0x03)
 * - Unlock: VALID|LOCKED (0x03) -> VALID (0x01) with version++
 * 
 * Invariants enforced:
 * - Entry must be VALID to set LOCKED
 * - Entry must have LOCKED bit to clear it
 * - EVICTED state cannot have LOCKED bit
 */
void
CalicoSetBufferLocked(BufferDesc *buf, bool locked)
{
    BufferTag *tag;
    uint32 rid;
    SecondLevelTable l2;
    CalicoTranslationEntry *entry;
    uint64 cur, desired;
    uint32 frame_id, version, newHash;
    uint8 state, new_state;
    LWLock *plock;

    if (!buf)
        return;

    tag = &buf->tag;
    
    /* Get the relation ID for this buffer's tag */
    rid = CalicoGetRelationIDInternal(tag->spcOid, tag->dbOid, tag->relNumber, tag->forkNum);
    if (rid == 0 || rid >= MAX_RELATION_IDS){
        *(int*)(NULL)=1;
        return; /* Invalid relation ID */
    }
        

    /* Get the second-level table */
    l2 = (SecondLevelTable)(uintptr_t)pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
    if (!l2) {
        *(int*)(NULL)=1;
        return; /* No L2 table exists */
    }
        

    if (tag->blockNum >= MAX_BLOCKS_PER_REL){
        *(int*)(NULL)=1;
        return; /* Block number out of range */
    }
        

    entry = &l2[tag->blockNum];

    /* Compute partition lock for this mapping (same as buffer operations) */
    newHash = BufTableHashCode(tag);
    plock = BufMappingPartitionLock(newHash);

    /* Take partition lock to ensure consistency with mapping operations */
    LWLockAcquire(plock, LW_EXCLUSIVE);

    /* Use CAS loop to update the LOCKED bit */
    for (;;)
    {
        cur = pg_atomic_read_u64(entry);
        frame_id = CalicoGetFrameId(cur);
        version = CalicoGetVersion(cur);
        state = CalicoGetState(cur);

        /* Verify this entry still points to our buffer */
        if (frame_id != buf->buf_id)
        {
            /* Entry changed - points to different buffer or evicted */
            if (locked)
            {
                /* Trying to set LOCKED on evicted/changed entry - error */
                elog(ERROR, "CalicoSetBufferLocked: buffer %d entry changed when locking (now points to buf %u)", 
                     buf->buf_id, frame_id);
                LWLockRelease(plock);
                *(int*)(NULL)=1;
                return;
            }
            else
            {
                /* Trying to clear LOCKED on evicted/changed entry - silently succeed
                 * The eviction already cleared the LOCKED bit */
                elog(ERROR, "CalicoSetBufferLocked: buffer %d entry changed when unlocking (now points to buf %u)", 
                     buf->buf_id, frame_id);
                LWLockRelease(plock);
                *(int*)(NULL)=1;
                return;
            }
        }

        /* Check state validity */
        if ((state & CALICO_STATE_VALID) == 0)
        {
            // invalid

            elog(ERROR, "CalicoSetBufferLocked: buffer %d entry not valid (evicted)", buf->buf_id);

            /* Entry is not valid (evicted) */
            if (locked)
            {
                /* Trying to lock evicted entry - error */
                elog(WARNING, "CalicoSetBufferLocked: buffer %d not valid when locking", buf->buf_id);
                LWLockRelease(plock);
                return;
            }
            else
            {
                /* Trying to unlock evicted entry - silently succeed
                 * The eviction already cleared the LOCKED bit */
                LWLockRelease(plock);
                *(int*)(NULL)=1;
                return;
            }
        }

        if (locked)
        {
            /* Setting lock: add LOCKED bit, keep version */
            /* Invariant: VALID must be set */
            if (state & CALICO_STATE_LOCKED)
            {
                elog(ERROR, "CalicoSetBufferLocked: buffer %d already locked", buf->buf_id);
                LWLockRelease(plock);
                return;
            }
            Assert(state & CALICO_STATE_VALID);
            new_state = state | CALICO_STATE_LOCKED;
            desired = CalicoPackEntry(frame_id, version + 1, new_state);
        }
        else
        {
            if (!(state & CALICO_STATE_VALID))
            {
                elog(ERROR, "CalicoSetBufferLocked: buffer %d not valid while unlocking", buf->buf_id);
                LWLockRelease(plock);
                return;
            }
            // if (!(state & CALICO_STATE_LOCKED))
            // {
            //     LWLockRelease(plock);
            //     return;
            // }
            
            Assert(state & CALICO_STATE_VALID);
            Assert(state & CALICO_STATE_LOCKED);
            new_state = state & ~CALICO_STATE_LOCKED;
            desired = CalicoPackEntry(frame_id, version + 1, new_state);
        }

        if (pg_atomic_compare_exchange_u64(entry, &cur, desired))
            break; /* Success */

        /* CAS failed, retry */
    }

    LWLockRelease(plock);
}

/* find-only (no create). return true if found and *out is set */
static bool
CalicoFindRelationIDIfExists(const BufferTag *tag, uint32 *out_relation_id)
{
    CalicoRelFork key;
    key.spcOid    = tag->spcOid;
    key.dbOid     = tag->dbOid;
    key.relNumber = tag->relNumber;
    key.forkNum   = tag->forkNum;

    uint32 hashv = get_hash_value(CalicoRelMap, (const void *)&key);
    LWLock *plock = BufMappingPartitionLock2(hashv);

    bool found;
    CalicoRelMapEntry *entry;

    LWLockAcquire(plock, LW_SHARED);
    entry = (CalicoRelMapEntry *) hash_search_with_hash_value(CalicoRelMap,
                                                              (const void *)&key,
                                                              hashv,
                                                              HASH_FIND,
                                                              &found);
    if (found)
        *out_relation_id = entry->relation_id;
    LWLockRelease(plock);

    return found;
}

/* -------------------------------------------------------------------------
 *                               Core Lookup
 * ------------------------------------------------------------------------- */

BufferDesc *
CalicoLookupAndPin(SMgrRelation smgr, char relpersistence, ForkNumber forkNum,
                   BlockNumber blockNum, BufferAccessStrategy strategy,
                   bool *foundPtr, IOContext io_context)
{
    const RelFileLocator *rlocator = &smgr->smgr_rlocator.locator;

    ResourceOwnerEnlarge(CurrentResourceOwner);
    ReservePrivateRefCountEntry();

    if (unlikely(blockNum >= MAX_BLOCKS_PER_REL)){
        *(int*)(NULL) = 1;
        memcpy(NULL, foundPtr, blockNum);
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("block number %u is out of range for Calico", blockNum)));
    }
        
    /* Try TLS cache first - check if we have the translation path cached */
    uint32 relation_id;
    SecondLevelTable second_level_table;
    bool tls_hit = CalicoTLSCacheLookup(rlocator->spcOid, rlocator->dbOid,
                                        rlocator->relNumber, forkNum,
                                        &relation_id, &second_level_table);
    
    if (!tls_hit)
    {
        /* TLS cache miss - compute relation_id and fetch L2 table */
        relation_id = CalicoGetRelationIDInternal(rlocator->spcOid, rlocator->dbOid,
                                                   rlocator->relNumber, forkNum);

        /* fetch/create second-level table pointer */
        bool found_shmem;
        char name[64];
        SecondLevelTable new_tab;
        uint64 expected;
        uint64 desired;
        uint64 p;
        
        p = pg_atomic_read_u64(&CalicoTopLevelTable[relation_id]);
        if (unlikely(p == 0))
        {
            snprintf(name, sizeof(name), "Calico 2nd-Level Table %u", relation_id);

            new_tab = (SecondLevelTable)
                ShmemInitStruct(name,
                                (Size)MAX_BLOCKS_PER_REL * sizeof(CalicoTranslationEntry),
                                &found_shmem);
            if (!found_shmem)
            {
                /* atomic initialize all entries to 0 */
                // for (uint32 i = 0; i < MAX_BLOCKS_PER_REL; i++)
                //     pg_atomic_init_u64(&new_tab[i], 0);
                //madvise huge page
                madvise(new_tab,
                         (Size)MAX_BLOCKS_PER_REL * sizeof(CalicoTranslationEntry),
                         MADV_NOHUGEPAGE);
            }

            expected = 0;
            desired  = (uint64)(uintptr_t)new_tab;
            if (!pg_atomic_compare_exchange_u64(&CalicoTopLevelTable[relation_id], &expected, desired))
            {
                /* someone else installed it */
                new_tab = (SecondLevelTable)(uintptr_t) expected;
            }
            second_level_table = new_tab;
            fprintf(stderr, "Created Calico L2 table for %s of size %zu\n", name, sizeof(CalicoTranslationEntry) * MAX_BLOCKS_PER_REL);
        }
        else
        {
            second_level_table = (SecondLevelTable)(uintptr_t)p;
        }
        
        /* Update TLS cache with the translation path */
        CalicoTLSCacheUpdate(rlocator->spcOid, rlocator->dbOid, rlocator->relNumber, forkNum,
                             relation_id, second_level_table);
    }
    
    /* Now we have the L2 table (either from cache or freshly fetched) */
    CalicoTranslationEntry *entry_ptr = &second_level_table[blockNum];


    /* Build the canonical BufferTag and choose the SAME mapping partition lock as hashtable */
    BufferTag tag;
    InitBufferTag(&tag, rlocator, forkNum, blockNum);
    uint64 entry_val = pg_atomic_read_u64(entry_ptr);

    if (CalicoIsValid(entry_val))
    {
        int buf_id = (int) CalicoGetFrameId(entry_val);
        
        //mm_prefetch the header
        Page * page = BufferGetPage(buf_id + 1);
        __builtin_prefetch(page, 0, 3);
        BufferDesc *buf = GetBufferDescriptor(buf_id);
        bool valid = PinBuffer(buf, strategy);


        /* Validate tag while we still hold the SAME partition lock */
        if (valid)
        {
            if (unlikely(!BufferTagsEqual(&buf->tag, &tag)))
            {
                /* mapping changed under us, treat as miss */
                UnpinBuffer(buf);
                goto pessimistic_path;
            }
        }
        *foundPtr = true;
        if (!valid)
            *foundPtr = false;
        else{
            
            *foundPtr = true;
        }
            

        return buf;
    }

pessimistic_path:
    uint32 newHash = BufTableHashCode(&tag);
    LWLock *plock = BufMappingPartitionLock(newHash);

    /* ------------- HIT PATH ------------- */
    LWLockAcquire(plock, LW_SHARED);
    entry_val = pg_atomic_read_u64(entry_ptr);

    if (CalicoIsValid(entry_val))
    {
        int buf_id = (int) CalicoGetFrameId(entry_val);
        BufferDesc *buf = GetBufferDescriptor(buf_id);
        bool valid = PinBuffer(buf, strategy);

        /* Validate tag while we still hold the SAME partition lock */
        if (valid)
        {
            if (unlikely(!BufferTagsEqual(&buf->tag, &tag)))
            {
                /* mapping changed under us, treat as miss */
                UnpinBuffer(buf);
                LWLockRelease(plock);
                goto miss_path;
            }
        }

        LWLockRelease(plock);

        *foundPtr = true;
        if (!valid)
            *foundPtr = false;

        return buf;
    }
    LWLockRelease(plock);
    // EVICTED STATE

miss_path:
    /* ------------- MISS PATH ------------- */
    {
        Buffer victim_buf = GetVictimBuffer(strategy, io_context);
        BufferDesc *victim = GetBufferDescriptor(victim_buf - 1);

        LWLockAcquire(plock, LW_EXCLUSIVE);

        entry_val = pg_atomic_read_u64(entry_ptr);
        // check invariant: evicted or (locked and valid)
        if (CalicoIsLocked(entry_val) && !CalicoIsValid(entry_val))
        {
            elog(ERROR, "Calico buffer mapping invariant violated: entry is LOCKED but not VALID");
        }
        if (CalicoIsValid(entry_val))
        {
            /* lost race: give up victim, use existing one */
            UnpinBuffer(victim);

            int buf_id = (int) CalicoGetFrameId(entry_val);
            BufferDesc *exist = GetBufferDescriptor(buf_id);
            bool valid = PinBuffer(exist, strategy);

            if (valid)
            {
                if (unlikely(!BufferTagsEqual(&exist->tag, &tag)))
                {
                    /* Extremely unlikely; fallback to miss again */
                    UnpinBuffer(exist);
                    LWLockRelease(plock);
                    goto miss_path;
                }
            }

            LWLockRelease(plock);

            *foundPtr = true;
            if (!valid) *foundPtr = false;
            return exist;
        }
        
        /*
         * Publish mapping via CAS under the SAME partition lock.
         * desired = {VALID, version+1, victim_id}
         */
        uint32 old_ver = CalicoGetVersion(entry_val);
        uint8  old_st  = CalicoGetState(entry_val) | CALICO_STATE_VALID;

        uint64 expect  = entry_val;
        uint64 desired = CalicoPackEntry(victim->buf_id, old_ver + 1, old_st);

        if (!pg_atomic_compare_exchange_u64(entry_ptr, &expect, desired))
        {
            /* someone else inserted concurrently: use theirs */
            UnpinBuffer(victim);
            uint64 now = pg_atomic_read_u64(entry_ptr);
            int buf_id = (int) CalicoGetFrameId(now);
            BufferDesc *exist = GetBufferDescriptor(buf_id);
            bool valid = PinBuffer(exist, strategy);
            LWLockRelease(plock);
            *foundPtr = valid;
            return exist;
        }

        /*
         * We "won" publishing. Now set the buffer header tag/state.
         * Keep the exact order: directory first, then buffer header.
         */
        uint32 vstate = LockBufHdr(victim);

        /* These asserts mirror hashtable path’s invariants */
        Assert(BUF_STATE_GET_REFCOUNT(vstate) == 1);
        Assert(!(vstate & (BM_TAG_VALID | BM_VALID | BM_DIRTY | BM_IO_IN_PROGRESS)));

        victim->tag = tag;

        vstate |= (BM_TAG_VALID | BUF_USAGECOUNT_ONE);
        if (relpersistence == RELPERSISTENCE_PERMANENT || forkNum == INIT_FORKNUM)
            vstate |= BM_PERMANENT;

        UnlockBufHdr(victim, vstate);

        LWLockRelease(plock);

        *foundPtr = false;
        return victim;
    }
}

/* -------------------------------------------------------------------------
 *                                 Prefetch
 * ------------------------------------------------------------------------- */

PrefetchBufferResult
CalicoPrefetchBuffer(SMgrRelation smgr, ForkNumber forkNum, BlockNumber blockNum)
{
    PrefetchBufferResult result = {InvalidBuffer, false};
    const RelFileLocator *rlocator = &smgr->smgr_rlocator.locator;

    if (unlikely(blockNum >= MAX_BLOCKS_PER_REL))
        return result;

    /* Build BufferTag and compute SAME partition lock */
    BufferTag tag;
    InitBufferTag(&tag, rlocator, forkNum, blockNum);
    uint32 newHash = BufTableHashCode(&tag);
    LWLock *plock = BufMappingPartitionLock(newHash);

    /*
     * IMPORTANT: Prefetch is find-only. It MUST NOT create relation-id or
     * second-level tables. Only check if already present; otherwise try OS
     * prefetch.
     */
    /* Try to locate top-level L2 pointer without creating it */
    uint32 rid_tmp;
    if (!CalicoFindRelationIDIfExists(&tag, &rid_tmp))
    {
#ifdef USE_PREFETCH
        if ((io_direct_flags & IO_DIRECT_DATA) == 0 &&
            smgrprefetch(smgr, forkNum, blockNum, 1))
            result.initiated_io = true;
#endif
        return result;
    }

    uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rid_tmp]);
    if (p == 0)
    {
#ifdef USE_PREFETCH
        if ((io_direct_flags & IO_DIRECT_DATA) == 0 &&
            smgrprefetch(smgr, forkNum, blockNum, 1))
            result.initiated_io = true;
#endif
        return result;
    }

    SecondLevelTable second = (SecondLevelTable)(uintptr_t)p;
    CalicoTranslationEntry *entry_ptr = &second[blockNum];

    LWLockAcquire(plock, LW_SHARED);
    uint64 entry_val = pg_atomic_read_u64(entry_ptr);

    if (CalicoIsValid(entry_val))
        result.recent_buffer = CalicoGetFrameId(entry_val) + 1;
    else
    {
#ifdef USE_PREFETCH
        if ((io_direct_flags & IO_DIRECT_DATA) == 0 &&
            smgrprefetch(smgr, forkNum, blockNum, 1))
            result.initiated_io = true;
#endif
    }
    LWLockRelease(plock);

    return result;
}


/* -------------------------------------------------------------------------
 *                        Safe Invalidate for Victim
 * ------------------------------------------------------------------------- */

bool
CalicoInvalidateVictimBuffer(BufferDesc *buf)
{
    Assert(GetPrivateRefCount(BufferDescriptorGetBuffer(buf)) == 1);

    /* have buffer pinned, so it's safe to read tag without lock */
    BufferTag candidate = buf->tag;

    /* find relation id without creating new one */
    uint32 rel_id;
    if (!CalicoFindRelationIDIfExists(&candidate, &rel_id))
        return false; /* directory didn't have it; match hashtable semantics */

    uint32 block_num = candidate.blockNum;

    /* SAME partition lock as hashtable path */
    uint32 newHash = BufTableHashCode(&candidate);
    LWLock *plock = BufMappingPartitionLock(newHash);

    LWLockAcquire(plock, LW_EXCLUSIVE);

    uint32 bstate = LockBufHdr(buf);

    if (!(bstate & BM_TAG_VALID) ||
        !BufferTagsEqual(&buf->tag, &candidate) ||
        BUF_STATE_GET_REFCOUNT(bstate) != 1 ||
        (bstate & BM_DIRTY))
    {
        UnlockBufHdr(buf, bstate);
        LWLockRelease(plock);
        return false;
    }

    /* 1) CAS directory to EVICTED FIRST - increment version to invalidate optimistic readers */
    uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rel_id]);
    if (p != 0)
    {
        SecondLevelTable second = (SecondLevelTable)(uintptr_t)p;
        CalicoTranslationEntry *entry_ptr = &second[block_num];
        uint64 cur = pg_atomic_read_u64(entry_ptr);

        if (CalicoIsValid(cur))
        {
            /* Check that LOCKED bit is not set - if it is, someone holds exclusive lock */
            if (CalicoIsLocked(cur))
            {
                /* Buffer has exclusive lock, should not evict */
                UnlockBufHdr(buf, bstate);
                LWLockRelease(plock);
                elog(ERROR, "CalicoInvalidateVictimBuffer: attempting to evict locked buffer");
                return false;
            }
            
            uint64 desired = CalicoPackEntry(0, CalicoGetVersion(cur) + 1, CALICO_STATE_EVICTED);
            /* CAS will fail if LOCKED bit gets set between check and CAS */
            if (!pg_atomic_compare_exchange_u64(entry_ptr, &cur, desired))
            {
                /* Entry changed - check if LOCKED bit was set */
                uint64 new_val = pg_atomic_read_u64(entry_ptr);
                if (CalicoIsLocked(new_val))
                {
                    UnlockBufHdr(buf, bstate);
                    LWLockRelease(plock);
                    elog(ERROR, "CalicoInvalidateVictimBuffer: buffer locked during eviction");
                    return false;
                }
                /* Otherwise, entry was modified in some other way, treat as failure */
                UnlockBufHdr(buf, bstate);
                LWLockRelease(plock);
                return false;
            }
        }
    }

    /* 2) Now clear local header AFTER version increment - optimistic readers are now invalidated */
    ClearBufferTag(&buf->tag);
    bstate &= ~(BUF_FLAG_MASK | BUF_USAGECOUNT_MASK);
    UnlockBufHdr(buf, bstate);

    LWLockRelease(plock);
    return true;
}

/* -------------------------------------------------------------------------
 *                        Forceful Invalidate (error paths)
 * ------------------------------------------------------------------------- */

void
CalicoForceInvalidateBuffer(BufferDesc *buf)
{
    uint32 bstate = pg_atomic_read_u32(&buf->state);
    uint32 rel_id;
    Assert(bstate & BM_LOCKED);

    bool   has_tag = (bstate & BM_TAG_VALID) != 0;
    BufferTag oldTag;
    if (has_tag)
        oldTag = buf->tag;

    UnlockBufHdr(buf, bstate);

    if (!has_tag)
        return;

retry:
    /* find relation id only if exists */
    if (!CalicoFindRelationIDIfExists(&oldTag, &rel_id))
        return; /* nothing to do in directory */

    /* SAME partition lock as hashtable path */
    uint32 newHash = BufTableHashCode(&oldTag);
    LWLock *plock = BufMappingPartitionLock(newHash);

    LWLockAcquire(plock, LW_EXCLUSIVE);
    bstate = LockBufHdr(buf);

    if (!BufferTagsEqual(&buf->tag, &oldTag))
    {
        UnlockBufHdr(buf, bstate);
        LWLockRelease(plock);
        return;
    }

    if (BUF_STATE_GET_REFCOUNT(bstate) != 0)
    {
        UnlockBufHdr(buf, bstate);
        LWLockRelease(plock);

        if (GetPrivateRefCount(BufferDescriptorGetBuffer(buf)) > 0)
            elog(ERROR, "buffer is pinned by self in CalicoForceInvalidateBuffer");

        WaitIO(buf);
        goto retry;
    }

    /* 1) CAS directory to EVICTED FIRST - increment version to invalidate optimistic readers */
    uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rel_id]);
    if (p != 0)
    {
        SecondLevelTable second = (SecondLevelTable)(uintptr_t)p;
        CalicoTranslationEntry *entry_ptr = &second[oldTag.blockNum];
        uint64 cur = pg_atomic_read_u64(entry_ptr);

        if (CalicoIsValid(cur))
        {
            // if locked
            if (CalicoIsLocked(cur))
            {
                UnlockBufHdr(buf, bstate);
                LWLockRelease(plock);
                elog(ERROR, "CalicoForceInvalidateBuffer: attempting to evict locked buffer" );
                return;
            }
            uint64 desired = CalicoPackEntry(0, CalicoGetVersion(cur) + 1, CALICO_STATE_EVICTED);
            /* 
             * CAS to evict - this is force invalidate so we clear both VALID and LOCKED bits.
             * This is only called in error paths where we need to clean up regardless of lock state.
             */
            if (!pg_atomic_compare_exchange_u64(entry_ptr, &cur, desired))
            {
               UnlockBufHdr(buf, bstate);
               LWLockRelease(plock);
               elog(ERROR, "CalicoForceInvalidateBuffer: entry changed during eviction");
               return;
            }
        }
    }

    /* 2) Now clear local header AFTER version increment - optimistic readers are now invalidated */
    ClearBufferTag(&buf->tag);
    bstate &= ~(BUF_FLAG_MASK | BUF_USAGECOUNT_MASK);
    UnlockBufHdr(buf, bstate);

    LWLockRelease(plock);
}


static inline bool
CalicoMap_TagToRidBlk(const BufferTag *tag, uint32 *out_rid, uint32 *out_blk)
{
    uint32 rid;
    if (!CalicoFindRelationIDIfExists(tag, &rid))
        return false;
    *out_rid = rid;
    *out_blk = tag->blockNum;
    return true;
}

int
CalicoShim_BufTableLookup(const BufferTag *tag, uint32 hashcode)
{
    uint32 rid, blk;
    SecondLevelTable second;
    CalicoTranslationEntry *entry_ptr;
    uint64 p;
    uint64 v;
    int ret;
    
    if (!CalicoMap_TagToRidBlk(tag, &rid, &blk))
        return -1;

    p = pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
    if (p == 0)
        return -1;

    second = (SecondLevelTable)(uintptr_t)p;
    entry_ptr = &second[blk];
    v = pg_atomic_read_u64(entry_ptr);
    ret = CalicoIsValid(v) ? (int)CalicoGetFrameId(v) : -1;
    return ret;
}

int
CalicoShim_BufTableInsert(const BufferTag *tag, uint32 hashcode, int buf_id)
{
    uint32 rid;
    SecondLevelTable second;
    bool found_shmem;
    char name[64];
    SecondLevelTable new_tab;
    uint64 p;
    uint64 expected;
    uint64 desired;
    uint32 blk;
    CalicoTranslationEntry *entry_ptr;
    uint64 cur;
    uint32 i;
    
    rid = CalicoGetRelationIDInternal(tag->spcOid, tag->dbOid, tag->relNumber, tag->forkNum);

    p = pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
    if (p == 0)
    {
        snprintf(name, sizeof(name), "Calico 2nd-Level Table %u", rid);
        new_tab = (SecondLevelTable)
            ShmemInitStruct(name,
                            (Size)MAX_BLOCKS_PER_REL * sizeof(CalicoTranslationEntry),
                            &found_shmem);
        if (!found_shmem)
        {
            for (i = 0; i < MAX_BLOCKS_PER_REL; i++)
                pg_atomic_init_u64(&new_tab[i], 0);
        }
        expected = 0;
        desired = (uint64)(uintptr_t)new_tab;
        if (!pg_atomic_compare_exchange_u64(&CalicoTopLevelTable[rid], &expected, desired))
            new_tab = (SecondLevelTable)(uintptr_t) expected;
        second = new_tab;
    }
    else
        second = (SecondLevelTable)(uintptr_t)p;

    blk = tag->blockNum;
    entry_ptr = &second[blk];

    cur = pg_atomic_read_u64(entry_ptr);
    if (CalicoIsValid(cur))
    {
        int existing = (int) CalicoGetFrameId(cur);
        return existing;
    }

    
    desired = CalicoPackEntry((uint32)buf_id, CalicoGetVersion(cur)+1, CalicoGetState(cur) | CALICO_STATE_VALID);
    if (!pg_atomic_compare_exchange_u64(entry_ptr, &cur, desired))
    {
        int existing = (int) CalicoGetFrameId(pg_atomic_read_u64(entry_ptr));
        return existing;
    }

    return -1;
}

void
CalicoShim_BufTableDelete(const BufferTag *tag, uint32 hashcode)
{
    uint32 rid, blk;
    uint64 p;
    SecondLevelTable second;
    CalicoTranslationEntry *entry_ptr;
    uint64 cur;
    uint64 desired;
    
    if (!CalicoMap_TagToRidBlk(tag, &rid, &blk))
        return;

    p = pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
    if (p == 0)
        return;

    second = (SecondLevelTable)(uintptr_t)p;
    entry_ptr = &second[blk];

    cur = pg_atomic_read_u64(entry_ptr);
    if (CalicoIsValid(cur))
    {
        /* If LOCKED bit is set, we should not delete - this would be a bug */
        if (CalicoIsLocked(cur))
        {
            elog(WARNING, "CalicoShim_BufTableDelete: attempting to delete locked entry");
            return;
        }
        
        desired = CalicoPackEntry(0, CalicoGetVersion(cur) + 1, CALICO_STATE_EVICTED);
        if (!pg_atomic_compare_exchange_u64(entry_ptr, &cur, desired))
        {
            /* CAS failed - check if LOCKED bit was set */
            uint64 new_val = pg_atomic_read_u64(entry_ptr);
            if (CalicoIsLocked(new_val))
                elog(WARNING, "CalicoShim_BufTableDelete: entry became locked during delete");
        }
    }
}

/* -------------------------------------------------------------------------
 *                    Optimistic Read Interfaces
 * ------------------------------------------------------------------------- */

/*
 * CalicoLookupAndOptimisticPin - Lookup a buffer without taking locks
 * 
 * This function performs an optimistic read of the buffer mapping:
 * 1. Reads the version number from the translation entry
 * 2. Returns the frame ID (buf_id) directly to avoid BufferDesc dereference
 * 3. Returns entry_ptr pointer for fast validation without re-lookup
 * 4. Caller must validate using CalicoLookupAndOptimisticValidate with entry_ptr
 * 
 * Returns NULL if the page is not in the buffer pool
 */
static BufferDesc *
CalicoLookupAndOptimisticPinInternal(SMgrRelation smgr,
                                      ForkNumber forkNum,
                                      BlockNumber blockNum,
                                      bool *foundPtr,
                                      uint64 *versionOut,
                                      uint32 *frameIdOut,
                                      void **entryPtrOut)
{
    RelFileLocator *rlocator = &smgr->smgr_rlocator.locator;
    uint32 rid;
    SecondLevelTable second;
    CalicoTranslationEntry *entry_ptr;
    uint64 entry_val;
    uint32 buf_id;
    BufferDesc *buf;
    
    *foundPtr = false;
    *versionOut = 0;
    *frameIdOut = 0;
    *entryPtrOut = NULL;
    
    if (unlikely(blockNum >= MAX_BLOCKS_PER_REL)){
        *(int*)(NULL) = 1;
        return NULL;
    }

    
    /* Try TLS cache first if relation_id not provided */
    bool tls_hit = CalicoTLSCacheLookup(rlocator->spcOid, rlocator->dbOid,
                                        rlocator->relNumber, forkNum,
                                        &rid, &second);
    
    if (!tls_hit) {
        /* TLS cache miss - get relation ID */
        rid = CalicoGetRelationIDInternal(rlocator->spcOid, rlocator->dbOid,
                                        rlocator->relNumber, forkNum);
        /* Get second-level table */
        uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
        if (p == 0)
            return NULL;
        
        second = (SecondLevelTable)(uintptr_t)p;
        
        /* Update TLS cache with the translation path */
        CalicoTLSCacheUpdate(rlocator->spcOid, rlocator->dbOid, rlocator->relNumber, forkNum,
                                rid, second);
    }

    entry_ptr = &second[blockNum];
    
    /* Read entry atomically */
    entry_val = pg_atomic_read_u64(entry_ptr);

    /* Check if entry is valid or locked */
    if (!CalicoIsValid(entry_val) || CalicoIsLocked(entry_val))
        return NULL;
    
    buf_id = CalicoGetFrameId(entry_val);
    buf = GetBufferDescriptor(buf_id);
    
    /*
     * CRITICAL: Memory barrier to ensure the entry read completes BEFORE
     * the caller reads buffer content. This prevents the CPU from reordering
     * buffer content reads before the entry read, which would break the
     * optimistic read protocol.
     */
    //pg_memory_barrier();
    
    /* Return the version number, frame ID, and entry pointer for validation */
    *versionOut = entry_val;
    *frameIdOut = buf_id;
    *entryPtrOut = (void *)entry_ptr;
    *foundPtr = true;
    
    return buf;
}

/*
 * CalicoLookupAndOptimisticPin - Public API for optimistic buffer lookup
 * 
 * Attempts to pin a buffer without taking locks. Returns the buffer descriptor
 * and version information for later validation.
 */
BufferDesc *
CalicoLookupAndOptimisticPin(SMgrRelation smgr,
							ForkNumber forkNum,
							BlockNumber blockNum,
							bool *foundPtr,
							uint64 *versionOut,
							uint32 *frameIdOut,
							void **entryPtrOut)
{
    return CalicoLookupAndOptimisticPinInternal(smgr, forkNum, blockNum,
                                                 foundPtr, versionOut, frameIdOut,
                                                 entryPtrOut);
}

/*
 * CalicoLookupAndOptimisticValidate - Validate an optimistic read
 * 
 * Checks if the version number in the translation entry has changed
 * since the optimistic read. Takes entry_ptr directly to avoid re-querying
 * CalicoGetRelationID and second-level table lookup.
 * Returns true if the data is still valid.
 */
bool
CalicoLookupAndOptimisticValidate(void *entryPtr,
                                   uint64 expectedVersion)
{
    CalicoTranslationEntry *entry_ptr = (CalicoTranslationEntry *)entryPtr;
    uint64 current_val;
    
    if (entry_ptr == NULL)
        return false;
    
    // /*
    //  * CRITICAL: Memory barrier to ensure all buffer content reads complete
    //  * BEFORE we read the version for validation. Without this, the CPU could
    //  * reorder the validation read before the buffer content reads, causing
    //  * us to validate against a stale version while reading modified data.
    //  */
    // pg_memory_barrier();
    
    /* Read current entry value */
    current_val = pg_atomic_read_u64(entry_ptr);
    
    /* Compare with expected version - exact match required */
    return (current_val == expectedVersion);
}

/*
 * CalicoLookupAndOptimisticPinBatch - Batch lookup for improved ILP/MLP
 * 
 * Looks up multiple blocks in parallel to maximize instruction and memory
 * level parallelism. Uses loop unrolling to expose more parallelism to
 * the CPU and reduce loop overhead.
 * 
 * Parameters:
 *   smgr - SMgrRelation for the relation
 *   forkNum - Fork number
 *   blockNums - Array of block numbers to lookup
 *   count - Number of blocks to lookup
 *   frameIdsOut - Output array for frame IDs (must be pre-allocated)
 *   foundOut - Output array for found flags (must be pre-allocated)
 */
void
CalicoLookupAndOptimisticPinBatch(SMgrRelation smgr,
                                   ForkNumber forkNum,
                                   const BlockNumber *blockNums,
                                   int count,
                                   uint32 *frameIdsOut,
                                   bool *foundOut)
{
    const RelFileLocator *rlocator = &smgr->smgr_rlocator.locator;
    uint32 relation_id;
    SecondLevelTable second_level_table;
    int i;
    
    /* Initialize outputs */
    for (i = 0; i < count; i++)
    {
        foundOut[i] = false;
    }
    
    /* Try TLS cache first */
    bool tls_hit = CalicoTLSCacheLookup(rlocator->spcOid, rlocator->dbOid,
                                        rlocator->relNumber, forkNum,
                                        &relation_id, &second_level_table);
    
    if (!tls_hit)
    {
        /* TLS cache miss - get relation ID and L2 table */
        relation_id = CalicoGetRelationIDInternal(rlocator->spcOid, rlocator->dbOid,
                                                   rlocator->relNumber, forkNum);
        
        /* Get second-level table */
        uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[relation_id]);
        if (p == 0)
            return; /* No second level table, all lookups fail */
        
        second_level_table = (SecondLevelTable)(uintptr_t)p;
        
        /* Update TLS cache with the translation path */
        CalicoTLSCacheUpdate(rlocator->spcOid, rlocator->dbOid, rlocator->relNumber, forkNum,
                             relation_id, second_level_table);
    }
    
    /* 
     * Main loop with 4-way unrolling for better ILP/MLP.
     * Processing 4 entries at a time allows:
     * - Multiple memory loads in flight simultaneously
     * - Better instruction scheduling by the CPU
     * - Reduced loop overhead
     */
    i = 0;
    for (; i + 3 < count; i += 4)
    {
        CalicoTranslationEntry *entry_ptr0, *entry_ptr1, *entry_ptr2, *entry_ptr3;
        uint64 entry_val0, entry_val1, entry_val2, entry_val3;
        uint8 state0, state1, state2, state3;
        
        /* Compute entry pointers - these can be done in parallel */
        entry_ptr0 = &second_level_table[blockNums[i + 0]];
        entry_ptr1 = &second_level_table[blockNums[i + 1]];
        entry_ptr2 = &second_level_table[blockNums[i + 2]];
        entry_ptr3 = &second_level_table[blockNums[i + 3]];
        
        /* Read entries - these loads can execute in parallel */
        entry_val0 = pg_atomic_read_u64(entry_ptr0);
        entry_val1 = pg_atomic_read_u64(entry_ptr1);
        entry_val2 = pg_atomic_read_u64(entry_ptr2);
        entry_val3 = pg_atomic_read_u64(entry_ptr3);
        
        /* Extract states - independent operations */
        state0 = CalicoGetState(entry_val0);
        state1 = CalicoGetState(entry_val1);
        state2 = CalicoGetState(entry_val2);
        state3 = CalicoGetState(entry_val3);
        
        /* Process results */
        if (state0 & CALICO_STATE_VALID)
        {
            frameIdsOut[i + 0] = CalicoGetFrameId(entry_val0);
            foundOut[i + 0] = true;
        }
        
        if (state1 & CALICO_STATE_VALID)
        {
            frameIdsOut[i + 1] = CalicoGetFrameId(entry_val1);
            foundOut[i + 1] = true;
        }
        
        if (state2 & CALICO_STATE_VALID)
        {
            frameIdsOut[i + 2] = CalicoGetFrameId(entry_val2);
            foundOut[i + 2] = true;
        }
        
        if (state3 & CALICO_STATE_VALID)
        {
            frameIdsOut[i + 3] = CalicoGetFrameId(entry_val3);
            foundOut[i + 3] = true;
        }
    }
    
    /* Handle remaining elements (0-3 items) */
    for (; i < count; i++)
    {
        CalicoTranslationEntry *entry_ptr;
        uint64 entry_val;
        uint8 state;
        
        entry_ptr = &second_level_table[blockNums[i]];
        entry_val = pg_atomic_read_u64(entry_ptr);
        state = CalicoGetState(entry_val);
        
        if (state & CALICO_STATE_VALID)
        {
            frameIdsOut[i] = CalicoGetFrameId(entry_val);
            foundOut[i] = true;
        }
    }
}