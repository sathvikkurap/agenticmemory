/**
 * Basic tests for agent_mem_db Node bindings.
 * Run: node test.js (after npm run build)
 */
const { AgentMemDb, AgentMemDbDisk, createEpisode } = require('./index.js');

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'Assertion failed');
}

const dim = 8;
const db = new AgentMemDb(dim);

// Store episodes
const ep1 = createEpisode('task_1', Array(dim).fill(0.1), 1.0);
const ep2 = createEpisode('task_2', Array(dim).fill(0.2), 0.5);
db.storeEpisode(ep1);
db.storeEpisode(ep2);

// Query
const hits = db.querySimilar(Array(dim).fill(0.1), 0.0, 2);
assert(hits.length === 2, `expected 2 hits, got ${hits.length}`);
assert(hits[0].reward >= hits[1].reward, 'results should be ordered by similarity');

// Save/load
const path = require('path').join(require('os').tmpdir(), 'agent_mem_db_test.json');
db.saveToFile(path);
const db2 = AgentMemDb.loadFromFile(path);
const loaded = db2.querySimilar(Array(dim).fill(0.1), 0.0, 1);
assert(loaded.length === 1, `expected 1 hit after load, got ${loaded.length}`);
assert(loaded[0].taskId === 'task_1', `expected task_1, got ${loaded[0].taskId}`);
require('fs').unlinkSync(path);

// Prune tests
const db3 = new AgentMemDb(dim);
db3.storeEpisode(createEpisode('old', Array(dim).fill(0.1), 0.9, null, 1000));
db3.storeEpisode(createEpisode('new', Array(dim).fill(0.1), 0.8, null, 3000));
const removed = db3.pruneOlderThan(2000);
assert(removed === 1, `prune_older_than: expected 1 removed, got ${removed}`);
const afterPrune = db3.querySimilar(Array(dim).fill(0.1), 0.0, 5);
assert(afterPrune.length === 1 && afterPrune[0].taskId === 'new', 'prune should keep only new');

const db4 = new AgentMemDb(dim);
db4.storeEpisode(createEpisode('a', Array(dim).fill(0.1), 0.9, null, 1000));
db4.storeEpisode(createEpisode('b', Array(dim).fill(0.1), 0.8, null, 2000));
db4.storeEpisode(createEpisode('c', Array(dim).fill(0.1), 0.7, null, 3000));
const removed2 = db4.pruneKeepNewest(2);
assert(removed2 === 1, `prune_keep_newest: expected 1 removed, got ${removed2}`);
const afterNewest = db4.querySimilar(Array(dim).fill(0.1), 0.0, 5);
assert(afterNewest.length === 2, `prune_keep_newest: expected 2, got ${afterNewest.length}`);

const db5 = new AgentMemDb(dim);
db5.storeEpisode(createEpisode('low', Array(dim).fill(0.1), 0.3));
db5.storeEpisode(createEpisode('high', Array(dim).fill(0.1), 0.9));
db5.storeEpisode(createEpisode('mid', Array(dim).fill(0.1), 0.5));
const removed3 = db5.pruneKeepHighestReward(2);
assert(removed3 === 1, `prune_keep_highest_reward: expected 1 removed, got ${removed3}`);
const afterReward = db5.querySimilar(Array(dim).fill(0.1), 0.0, 5);
assert(afterReward.length === 2, `prune_keep_highest_reward: expected 2, got ${afterReward.length}`);

// AgentMemDBDisk with checkpoint
const path2 = require('path').join(require('os').tmpdir(), `agent_mem_db_disk_test_${Date.now()}`);
const diskDb = AgentMemDbDisk.openExactWithCheckpoint(path2, dim);
diskDb.storeEpisode(createEpisode('t1', Array(dim).fill(0.1), 0.7));
diskDb.storeEpisode(createEpisode('t2', Array(dim).fill(0.1), 0.8));
diskDb.checkpoint();
const diskDb2 = AgentMemDbDisk.openExactWithCheckpoint(path2, dim);
const diskHits = diskDb2.querySimilar(Array(dim).fill(0.1), 0.5, 5);
assert(diskHits.length === 2, `disk checkpoint: expected 2 hits, got ${diskHits.length}`);
require('fs').rmSync(path2, { recursive: true, force: true });

console.log('All Node tests passed.');
