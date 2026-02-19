const { AgentMemDb, createEpisode } = require('./index.js');

const dim = 8;
const db = new AgentMemDb(dim);

// Store a few episodes
for (let i = 0; i < 5; i++) {
  const embedding = Array(dim).fill(0.1).map((x, j) => x + j * 0.01);
  const ep = createEpisode(`task_${i}`, embedding, 0.5 + i * 0.1);
  db.storeEpisode(ep);
}

// Query
const query = Array(dim).fill(0.1);
const hits = db.querySimilar(query, 0.0, 3);
console.log('Top 3 similar episodes:', hits.length);
hits.forEach((h, i) => console.log(`  ${i + 1}. ${h.taskId} reward=${h.reward}`));
