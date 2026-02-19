# Design: Configurable Retention and Pruning

**Status:** First slice implemented  
**Related:** [design_roadmap.md](design_roadmap.md), [design_on_disk_storage.md](design_on_disk_storage.md)

## Goal

Allow agents to limit memory growth by pruning episodes based on time, count, or future: importance. Episodes without timestamps are handled explicitly.

## Use Cases

- **Time-based retention:** "Keep only episodes from the last 7 days"
- **Count-based retention:** "Keep only the 1000 most recent episodes"
- **Importance-based (future):** "Evict low-reward episodes when over capacity"

## Current State

- Episodes have optional `timestamp` (Unix ms)
- No pruning API; DB grows unbounded
- AgentMemDBDisk has append-only log; no compaction

## Approach

### First Slice: Time-Based Pruning

**AgentMemDB** and **AgentMemDBDisk**:

- `prune_older_than(&mut self, timestamp_cutoff_ms: i64) -> usize`
  - Remove episodes with `timestamp < cutoff`
  - Episodes without timestamp: **keep** (conservative; user can filter at store time)
  - Returns number of episodes removed
  - Implementation: rebuild index with kept episodes (HNSW/Exact don't support in-place removal)

**Python:**

- `db.prune_older_than(timestamp_cutoff_ms) -> int`

### Second Slice: Count-Based Pruning (Implemented)

- `prune_keep_newest(n: usize)` — keep n most recent by timestamp
- Episodes without timestamp treated as oldest (pruned first)
- AgentMemDBDisk compacts log

### Third Slice: Importance-Based Pruning (Implemented)

- `prune_keep_highest_reward(n: usize)` — keep n episodes with highest reward
- Ties: prefer more recent (higher timestamp); episodes without timestamp sort last
- AgentMemDBDisk compacts log

### Future Slices

- **AgentMemDBDisk compaction:** Rewrite log to remove pruned episodes (currently log stays append-only; replay filters at load time or we add compaction)

## Backward Compatibility

- Additive API. No breaking changes.
- Episodes without timestamp are never pruned by time-based retention.

## Implementation (First Slice)

- **AgentMemDB::prune_older_than(timestamp_cutoff_ms) -> usize** — rebuilds index with kept episodes
- **AgentMemDBDisk::prune_older_than(timestamp_cutoff_ms) -> Result<usize>** — rebuilds index and compacts log
- **Python:** `db.prune_older_than(timestamp_cutoff_ms)`; `AgentMemDBAsync.prune_older_than_async`
- Tests: `tests/basic.rs::test_prune_older_than`, `tests/disk.rs::test_disk_prune_older_than`, `test_basic.py::test_prune_older_than`
- **prune_keep_newest(n):** AgentMemDB, AgentMemDBDisk, Python, AgentMemDBAsync; tests in basic.rs, disk.rs, test_basic.py
- **Node:** pruneOlderThan, pruneKeepNewest; **Go:** PruneOlderThan, PruneKeepNewest (via C API)
- **prune_keep_highest_reward(n):** AgentMemDB, AgentMemDBDisk, Python, AgentMemDBAsync, Node, Go, HTTP

## Out of Scope

- Automatic/scheduled pruning (caller invokes explicitly)
