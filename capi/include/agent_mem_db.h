#ifndef AGENT_MEM_DB_H
#define AGENT_MEM_DB_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* AgentMemDBHandle;

AgentMemDBHandle agent_mem_db_new(size_t dim);
void agent_mem_db_free(AgentMemDBHandle h);
size_t agent_mem_db_dim(AgentMemDBHandle h);

int agent_mem_db_store(AgentMemDBHandle h, const char* task_id,
                       const float* embedding, size_t dim, float reward);

char* agent_mem_db_query(AgentMemDBHandle h, const float* embedding, size_t dim,
                         float min_reward, size_t top_k);

int agent_mem_db_save(AgentMemDBHandle h, const char* path);
AgentMemDBHandle agent_mem_db_load(const char* path);

size_t agent_mem_db_prune_older_than(AgentMemDBHandle h, int64_t timestamp_cutoff_ms);
size_t agent_mem_db_prune_keep_newest(AgentMemDBHandle h, size_t n);
size_t agent_mem_db_prune_keep_highest_reward(AgentMemDBHandle h, size_t n);

char* agent_mem_db_last_error(void);
void agent_mem_db_free_string(char* s);

/* AgentMemDBDisk — disk-backed storage */
typedef void* AgentMemDBDiskHandle;

AgentMemDBDiskHandle agent_mem_db_disk_open(const char* path, size_t dim);
AgentMemDBDiskHandle agent_mem_db_disk_open_exact_with_checkpoint(const char* path, size_t dim);
void agent_mem_db_disk_free(AgentMemDBDiskHandle h);

int agent_mem_db_disk_store(AgentMemDBDiskHandle h, const char* task_id,
                            const float* embedding, size_t dim, float reward);

char* agent_mem_db_disk_query(AgentMemDBDiskHandle h, const float* embedding, size_t dim,
                              float min_reward, size_t top_k);

int agent_mem_db_disk_checkpoint(AgentMemDBDiskHandle h);

int agent_mem_db_disk_prune_older_than(AgentMemDBDiskHandle h, int64_t timestamp_cutoff_ms);
int agent_mem_db_disk_prune_keep_newest(AgentMemDBDiskHandle h, size_t n);
int agent_mem_db_disk_prune_keep_highest_reward(AgentMemDBDiskHandle h, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* AGENT_MEM_DB_H */
