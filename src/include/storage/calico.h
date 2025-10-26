/* calico.h — 新增声明 */
#ifndef CALICO_H
#define CALICO_H

#include "storage/buf.h"
#include "storage/bufmgr.h"
#include "storage/buf_internals.h"

/* Calico 作为 BufTable 适配器的 3 个操作（保持 BufTable 语义与返回值） */
extern int  CalicoShim_BufTableLookup(const BufferTag *tag, uint32 hashcode);
extern int  CalicoShim_BufTableInsert(const BufferTag *tag, uint32 hashcode, int buf_id);
/* Delete 在 PG 里是 void；CalicoShim 返回 void 亦可 */
extern void CalicoShim_BufTableDelete(const BufferTag *tag, uint32 hashcode);

extern bool Calico_IsReady(void);

#endif /* CALICO_H */
