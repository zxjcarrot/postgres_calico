/* calico.h — Calico buffer manager declarations */
#ifndef CALICO_H
#define CALICO_H

#include "storage/buf.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"

/* Calico as BufTable adapter — maintains BufTable semantics and return values */
extern int  CalicoShim_BufTableLookup(const BufferTag *tag, uint32 hashcode);
extern int  CalicoShim_BufTableInsert(const BufferTag *tag, uint32 hashcode, int buf_id);
extern void CalicoShim_BufTableDelete(const BufferTag *tag, uint32 hashcode);

/* Check if Calico is ready to take over from BufTable */
extern bool Calico_IsReady(void);

#endif /* CALICO_H */
