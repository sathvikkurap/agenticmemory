# Design: Rich Metadata and Query Filters

**Status:** First slice in progress  
**Related:** [design_roadmap.md](design_roadmap.md)

## Goal

Enable structured metadata on episodes and expressive query filters so agents can:
- Filter by time ("recent episodes")
- Filter by tags ("coding", "support")
- Filter by task_id prefix or exact match
- Combine similarity search with constraints

## Backward Compatibility

All new fields are **optional**. Existing `Episode::new(...)` and `query_similar` remain unchanged. New constructors and query options extend the API.

## Episode Extensions

### Optional Fields (First Slice)

| Field       | Type              | Default   | Use case                    |
|------------|-------------------|-----------|-----------------------------|
| `timestamp` | `Option<i64>`     | None      | Unix ms; time-based filter  |
| `tags`     | `Option<Vec<String>>` | None  | Categorical filter          |
| `source`   | `Option<String>`  | None      | Origin (e.g., "api", "cli") |
| `user_id`  | `Option<String>`  | None      | Multi-tenant isolation      |

Existing `metadata` (JSON) remains for arbitrary structured data. The new fields enable fast filtering without JSON parsing.

### Constructor Extensions

```rust
Episode::new(task_id, embedding, reward)  // unchanged
Episode::with_timestamp(task_id, embedding, reward, ts)
Episode::with_tags(task_id, embedding, reward, tags)
// Or builder pattern for multiple optional fields
```

## Query Filters

### QueryOptions (First Slice)

```rust
pub struct QueryOptions {
    pub min_reward: f32,
    pub top_k: usize,
    pub tags_any: Option<Vec<String>>,   // episode has any of these tags
    pub tags_all: Option<Vec<String>>,   // episode has all of these tags
    pub time_after: Option<i64>,          // timestamp >= (Unix ms)
    pub time_before: Option<i64>,         // timestamp <= (Unix ms)
    pub task_id_prefix: Option<String>,  // task_id.starts_with(prefix)
}
```

### Filtering Strategy

1. Run vector search (existing index) to get candidate episodes.
2. Apply filters in order: min_reward, tags, time range, task_id.
3. Request `top_k * 2` or more from index to account for filtered-out results.
4. Stop when we have `top_k` that pass all filters.

For large tag/time filters, we may need to increase the candidate multiplier. Future: secondary indexes for tags/time if filtering dominates.

## API Evolution

### Phase 1 (This Slice)

- Add `timestamp`, `tags` to Episode (optional).
- Add `query_similar_with_options(query, QueryOptions)` — new method; `query_similar` unchanged.
- Python: `Episode(..., timestamp=..., tags=...)`, `query_similar(..., tags_any=..., time_after=...)`.

### Phase 2 (Done)

- `source`, `user_id` — optional Episode fields; QueryOptions filters.
- `tags_all` — episode must have all of these tags.
- `task_id_prefix` — task_id.starts_with(prefix).
- Secondary indexes for time/tags when needed.

## Implementation Notes

- Serde: `#[serde(default)]` for optional fields so old persisted JSON still deserializes.
- Python: Add optional kwargs to Episode constructor and query_similar.
