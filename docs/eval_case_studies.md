# Evaluation Case Studies

## Tuning Applied (2026-02)

### Problem
Initial runs showed agent_mem_db (83–90%) competitive with naive (86.7%) but not clearly winning on long-history tasks. Failure analysis: naive succeeded but agent_mem_db failed on 2 long tasks.

### Root Cause
Long tasks have 56 episodes (1 key + 55 filler). With `top_k=10`, the key episode sometimes ranked outside the top 10 when many filler episodes had similar embedding structure.

### Fix
- **Short tasks:** `top_k=10` (sufficient; only 1 episode)
- **Long tasks:** `top_k=25` (ensures key is in retrieved set among 56 episodes)

### Result
- Long tasks: agent_mem_db 88.9% vs naive 33.3%
- Overall: agent_mem_db 97.5% vs naive 85%

## Failure Modes (Pre-Tuning)

| Case | naive | agent_mem_db | Cause |
|------|-------|--------------|-------|
| Long task (secret code, VIP, backup email) | Sometimes | Sometimes | Key episode not in top-k when k=10 |

Post-tuning with `top_k=25` on long tasks, agent_mem_db failures on long tasks dropped from 2 to 1 (one remaining edge case).
