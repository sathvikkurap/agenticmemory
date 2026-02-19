# Design: Combine Similarity + Constraints

**Status:** First slice done (recency tie-breaker)  
**Related:** [design_roadmap.md](design_roadmap.md), [design_rich_metadata.md](design_rich_metadata.md)

## Goal

Enable queries that combine vector similarity with constraint-based ranking, e.g. "recent + similar" — prioritize episodes that are both semantically similar and recently stored.

## Current Behavior

Today we:
1. Run vector search (HNSW or exact) to get candidates ordered by similarity.
2. Apply filters (tags, time, source, user_id, etc.) — exclude non-matching.
3. Return top_k that pass all filters.

Result: **pure similarity order** among filtered episodes. Recency does not affect ranking.

## Use Cases

- **"Recent + similar"** — Agent recalls similar past experiences but prefers recent ones (e.g. user preferences may have changed).
- **Recency decay** — Older episodes contribute less to the final score.
- **Time-weighted similarity** — When two episodes are equally similar, prefer the more recent.

## Options

### Option A: Recency Boost (Score = similarity × recency_weight)

Add `recency_weight: Option<f32>` to QueryOptions. When set (e.g. 0.9), multiply similarity by `recency_weight^age` where age = (now - timestamp) in some unit. Recent episodes get higher effective score.

- **Pros:** Simple, one extra param, backward compatible.
- **Cons:** Requires timestamp on all episodes; need to define age scale (seconds? days?).

### Option B: Recency Tie-Breaker

When two episodes have (approximately) equal similarity, prefer the one with more recent timestamp. No new score formula; just change sort key to `(similarity, -timestamp)`.

- **Pros:** Minimal change, no new params.
- **Cons:** Only helps when similarities are very close; weak effect.

### Option C: Filter-First Strategy

When `time_after` is set and the filter is highly selective, run filter first (scan episodes by timestamp) then do similarity within the filtered set. Useful when e.g. "last 24h" returns 100 episodes and we want top 5 by similarity.

- **Pros:** Can be more efficient when filter dominates.
- **Cons:** Requires secondary index on timestamp; different code path; complexity.

### Option D: Hybrid Score Parameter

Add `recency_factor: Option<f32>` (0.0 = pure similarity, 1.0 = recency dominates). Final score = `(1 - recency_factor) * similarity + recency_factor * recency_score`, where recency_score is normalized 0–1 by time.

- **Pros:** Explicit control over similarity vs recency tradeoff.
- **Cons:** More params; need to define recency_score normalization.

## Recommendation

**First slice (done):** Option B (recency tie-breaker). Zero new API surface; when episodes have equal similarity (same L2 distance), sort by `-timestamp` so recent ones come first. Episodes without timestamp sort last. Implemented in `query_similar_with_options`.

**Future:** Option A or D if users need stronger recency influence.

## Implementation Sketch (Option B)

In `query_similar_with_options`, after filtering, instead of `.take(opts.top_k)` on the iterator (which preserves index order), we could:
- Request more candidates (e.g. top_k * 4) to have a larger pool.
- Sort the filtered results by `(similarity_desc, timestamp_desc)` — episodes with no timestamp go to end.
- Take top_k.

The index returns `(key, distance)` — we have distance. For tie-breaking we need timestamp. We already have episodes. So: collect filtered episodes with their distances, sort by (distance_asc, -timestamp), take top_k.

## Out of Scope (This Design)

- Secondary indexes for time/tags (see design_rich_metadata.md).
- Async APIs.
- PQ/IVF backends.
