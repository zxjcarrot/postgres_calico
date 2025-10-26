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

#define CALICO_STATE_EVICTED    0
#define CALICO_STATE_VALID      1
#define CALICO_STATE_LOCKED     2 /* reserved */

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

#define MAX_RELATION_IDS        (1U << 20) /* ~1M relation forks */
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

/* 全局 ready 标志 */
static volatile bool CalicoReady = false;

/* 对外查询：Calico 是否可以接管 BufTable */
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
                                 131072,                /* initial */
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
        for (uint32 i = 0; i < MAX_RELATION_IDS; i++)
            pg_atomic_init_u64(&CalicoTopLevelTable[i], 0);
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
    InitBufTable(NBuffers + NUM_BUFFER_PARTITIONS);
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

/* Backend-local cache to avoid shared-table lookups */
static HTAB *
CalicoLocalCache(void)
{
    static HTAB *local_cache = NULL;
    if (local_cache == NULL)
    {
        HASHCTL ctl;
        MemSet(&ctl, 0, sizeof(ctl));
        ctl.keysize   = sizeof(CalicoRelFork);
        ctl.entrysize = sizeof(CalicoRelMapEntry);
        local_cache = hash_create("Calico Local RelID Cache", 128, &ctl, HASH_ELEM | HASH_BLOBS);
    }
    return local_cache;
}

/* create-or-find */
static uint32
CalicoGetRelationID(Oid spcOid, Oid dbOid, RelFileNumber relNumber, ForkNumber forkNum)
{
    CalicoRelFork key;
    bool found;
    CalicoRelMapEntry *entry;
    HTAB *local = CalicoLocalCache();

    key.spcOid    = spcOid;
    key.dbOid     = dbOid;
    key.relNumber = relNumber;
    key.forkNum   = forkNum;

    /* local cache first */
    entry = (CalicoRelMapEntry *) hash_search(local, &key, HASH_FIND, &found);
    if (found)
        return entry->relation_id;

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
        uint32 rid = entry->relation_id;
        LWLockRelease(plock);

        bool lf;
        entry = (CalicoRelMapEntry *) hash_search(local, &key, HASH_ENTER, &lf);
        if (!lf) entry->relation_id = rid;
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
    uint32 rid = entry->relation_id;
    LWLockRelease(plock);

    bool lf;
    entry = (CalicoRelMapEntry *) hash_search(local, &key, HASH_ENTER, &lf);
    if (!lf) entry->relation_id = rid;
    return rid;
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

    if (unlikely(blockNum >= MAX_BLOCKS_PER_REL))
        ereport(ERROR, (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                        errmsg("block number %u is out of range for Calico", blockNum)));

    /* compute relation_id (create-or-find, like hashtable path implies by use) */
    uint32 relation_id = CalicoGetRelationID(rlocator->spcOid, rlocator->dbOid,
                                             rlocator->relNumber, forkNum);

    /* fetch/create second-level table pointer */
    SecondLevelTable second_level_table;
    {
        uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[relation_id]);
        if (unlikely(p == 0))
        {
            bool found_shmem;
            char name[64];
            snprintf(name, sizeof(name), "Calico 2nd-Level Table %u", relation_id);

            SecondLevelTable new_tab = (SecondLevelTable)
                ShmemInitStruct(name,
                                (Size)MAX_BLOCKS_PER_REL * sizeof(CalicoTranslationEntry),
                                &found_shmem);
            if (!found_shmem)
            {
                /* atomic initialize all entries to 0 */
                // for (uint32 i = 0; i < MAX_BLOCKS_PER_REL; i++)
                //     pg_atomic_init_u64(&new_tab[i], 0);
            }

            uint64 expected = 0;
            uint64 desired  = (uint64)(uintptr_t)new_tab;
            if (!pg_atomic_compare_exchange_u64(&CalicoTopLevelTable[relation_id], &expected, desired))
            {
                /* someone else installed it */
                new_tab = (SecondLevelTable)(uintptr_t) expected;
            }
            second_level_table = new_tab;
            fprintf(stderr, "Created Calico L2 table for %s of size %zu\n", name, sizeof(CalicoTranslationEntry) * MAX_BLOCKS_PER_REL);
        }
            second_level_table = (SecondLevelTable)(uintptr_t)p;
    CalicoTranslationEntry *entry_ptr = &second_level_table[blockNum];

    /* Build the canonical BufferTag and choose the SAME mapping partition lock as hashtable */
    BufferTag tag;
    InitBufferTag(&tag, rlocator, forkNum, blockNum);
    uint32 newHash = BufTableHashCode(&tag);
    LWLock *plock = BufMappingPartitionLock(newHash);

    /* ------------- HIT PATH ------------- */
    LWLockAcquire(plock, LW_SHARED);
    uint64 entry_val = pg_atomic_read_u64(entry_ptr);

    if (CalicoGetState(entry_val) == CALICO_STATE_VALID)
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

miss_path:
    /* ------------- MISS PATH ------------- */
    {
        Buffer victim_buf = GetVictimBuffer(strategy, io_context);
        BufferDesc *victim = GetBufferDescriptor(victim_buf - 1);

        LWLockAcquire(plock, LW_EXCLUSIVE);

        entry_val = pg_atomic_read_u64(entry_ptr);
        if (CalicoGetState(entry_val) == CALICO_STATE_VALID)
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
        uint8  old_st  = CalicoGetState(entry_val);

        if (old_st == CALICO_STATE_VALID)
        {
            /* theoretically handled above, but keep it robust */
            UnpinBuffer(victim);
            int buf_id = (int) CalicoGetFrameId(entry_val);
            BufferDesc *exist = GetBufferDescriptor(buf_id);
            bool valid = PinBuffer(exist, strategy);
            LWLockRelease(plock);
            *foundPtr = valid; /* follows hashtable semantics */
            return exist;
        }

        uint64 expect  = entry_val;
        uint64 desired = CalicoPackEntry(victim->buf_id, old_ver + 1, CALICO_STATE_VALID);

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

    if (CalicoGetState(entry_val) == CALICO_STATE_VALID)
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

    /* 1) clear local header first */
    ClearBufferTag(&buf->tag);
    bstate &= ~(BUF_FLAG_MASK | BUF_USAGECOUNT_MASK);
    UnlockBufHdr(buf, bstate);

    /* 2) CAS directory to EVICTED only if it still points to this frame */
    uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rel_id]);
    if (p != 0)
    {
        SecondLevelTable second = (SecondLevelTable)(uintptr_t)p;
        CalicoTranslationEntry *entry_ptr = &second[block_num];
        uint64 cur = pg_atomic_read_u64(entry_ptr);

        if (CalicoGetState(cur) == CALICO_STATE_VALID &&
            CalicoGetFrameId(cur) == (uint32)buf->buf_id)
        {
            uint64 desired = CalicoPackEntry(0, CalicoGetVersion(cur) + 1, CALICO_STATE_EVICTED);
            (void) pg_atomic_compare_exchange_u64(entry_ptr, &cur, desired);
            /* on failure: somebody changed it; match hashtable delete-fail */
        }
    }

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

    /* 1) clear local header first */
    ClearBufferTag(&buf->tag);
    bstate &= ~(BUF_FLAG_MASK | BUF_USAGECOUNT_MASK);
    UnlockBufHdr(buf, bstate);

    /* 2) CAS directory to EVICTED if it still points to this frame */
    uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rel_id]);
    if (p != 0)
    {
        SecondLevelTable second = (SecondLevelTable)(uintptr_t)p;
        CalicoTranslationEntry *entry_ptr = &second[oldTag.blockNum];
        uint64 cur = pg_atomic_read_u64(entry_ptr);

        if (CalicoGetState(cur) == CALICO_STATE_VALID &&
            CalicoGetFrameId(cur) == (uint32)buf->buf_id)
        {
            uint64 desired = CalicoPackEntry(0, CalicoGetVersion(cur) + 1, CALICO_STATE_EVICTED);
            (void) pg_atomic_compare_exchange_u64(entry_ptr, &cur, desired);
        }
    }

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
    if (!CalicoMap_TagToRidBlk(tag, &rid, &blk))
        return -1;

    uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
    if (p == 0)
        return -1;

    SecondLevelTable second = (SecondLevelTable)(uintptr_t)p;
    CalicoTranslationEntry *entry_ptr = &second[blk];
    uint64 v = pg_atomic_read_u64(entry_ptr);
    int ret = (CalicoGetState(v) == CALICO_STATE_VALID) ? (int)CalicoGetFrameId(v) : -1;
    return ret;
}

int
CalicoShim_BufTableInsert(const BufferTag *tag, uint32 hashcode, int buf_id)
{
    uint32 rid = CalicoGetRelationID(tag->spcOid, tag->dbOid, tag->relNumber, tag->forkNum);

    SecondLevelTable second;
    {
        uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
        if (p == 0)
        {
            bool found_shmem;
            char name[64];
            snprintf(name, sizeof(name), "Calico 2nd-Level Table %u", rid);
            SecondLevelTable new_tab = (SecondLevelTable)
                ShmemInitStruct(name,
                                (Size)MAX_BLOCKS_PER_REL * sizeof(CalicoTranslationEntry),
                                &found_shmem);
            if (!found_shmem)
            {
                for (uint32 i = 0; i < MAX_BLOCKS_PER_REL; i++)
                    pg_atomic_init_u64(&new_tab[i], 0);
            }
            uint64 expected = 0, desired = (uint64)(uintptr_t)new_tab;
            if (!pg_atomic_compare_exchange_u64(&CalicoTopLevelTable[rid], &expected, desired))
                new_tab = (SecondLevelTable)(uintptr_t) expected;
            second = new_tab;
        }
        else
            second = (SecondLevelTable)(uintptr_t)p;
    }

    uint32 blk = tag->blockNum;
    CalicoTranslationEntry *entry_ptr = &second[blk];

    uint64 cur = pg_atomic_read_u64(entry_ptr);
    if (CalicoGetState(cur) == CALICO_STATE_VALID)
    {
        int existing = (int) CalicoGetFrameId(cur);
        return existing;
    }

    uint64 desired = CalicoPackEntry((uint32)buf_id, CalicoGetVersion(cur)+1, CALICO_STATE_VALID);
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
    if (!CalicoMap_TagToRidBlk(tag, &rid, &blk))
        return;

    uint64 p = pg_atomic_read_u64(&CalicoTopLevelTable[rid]);
    if (p == 0)
        return;

    SecondLevelTable second = (SecondLevelTable)(uintptr_t)p;
    CalicoTranslationEntry *entry_ptr = &second[blk];

    uint64 cur = pg_atomic_read_u64(entry_ptr);
    if (CalicoGetState(cur) == CALICO_STATE_VALID)
    {
        uint64 desired = CalicoPackEntry(0, CalicoGetVersion(cur)+1, CALICO_STATE_EVICTED);
        (void) pg_atomic_compare_exchange_u64(entry_ptr, &cur, desired);
    }
}