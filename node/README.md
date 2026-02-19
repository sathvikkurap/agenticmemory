# agent_mem_db — Node.js bindings

Episodic memory for LLM agents with HNSW vector search. Native bindings via napi-rs.

## Install

```bash
cd node && npm install && npm run build
```

## Usage

```javascript
const { AgentMemDb, createEpisode } = require('agent_mem_db');

const db = new AgentMemDb(768);
const ep = createEpisode('task1', embeddingArray, 0.9);
db.storeEpisode(ep);
const hits = db.querySimilar(queryEmbedding, 0.0, 5);
```

### TypeScript

```typescript
import { AgentMemDb, createEpisode, Episode } from 'agent_mem_db';

const db = new AgentMemDb(768);
const ep = createEpisode('task1', embedding, 0.9, null, null, ['coding']);
db.storeEpisode(ep);
const hits = db.querySimilar(embedding, 0.0, 5, {
  tagsAny: ['coding'],
  timeAfter: 1000,
});
```

### API

- `AgentMemDb(dim)` — in-memory DB, HNSW backend
- `AgentMemDb.exact(dim)` — exact (brute-force) search
- `AgentMemDb.withMaxElements(dim, maxElements)` — scale
- `AgentMemDb.loadFromFile(path)` — load from JSON
- `AgentMemDbDisk.open(path, dim)` — disk-backed DB (HNSW)
- `AgentMemDbDisk.openExactWithCheckpoint(path, dim)` — disk-backed with checkpoint for fast restart
- `createEpisode(taskId, embedding, reward, metadata?, timestamp?, tags?)` — create episode
- `db.storeEpisode(episode)` — store
- `db.querySimilar(embedding, minReward, topK, opts?)` — query
- `db.saveToFile(path)` — persist (in-memory only)
- `diskDb.checkpoint()` — persist checkpoint (disk, ExactIndex only)

## Example

```bash
node example.js
```

## Test

```bash
npm run build && npm test
```

## Build

Requires Rust toolchain. From `node/`:

```bash
npm run build
```
