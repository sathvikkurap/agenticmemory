"""
AgentMemDB-backed agent: episodic memory with HNSW retrieval.
"""

import time
import agent_mem_db_py as agent_mem_db
from .base import Task, EvalResult, embed, stub_policy, grade_response

DIM = 16


def run_task(task: Task) -> EvalResult:
    """AgentMemDB: store episodes, retrieve before answering."""
    start = time.perf_counter()
    db = agent_mem_db.AgentMemDB(DIM)
    total_tokens = 0
    successes = 0
    turns = 0
    for stmt in task.store_statements:
        emb = embed(stmt, DIM)
        ep = agent_mem_db.Episode(
            task_id=task.task_id,
            state_embedding=emb,
            reward=1.0,
            metadata={"text": stmt},
        )
        db.store_episode(ep)
        turns += 1
        total_tokens += 50
    for q, expected in zip(task.recall_questions, task.expected_answers):
        turns += 1
        emb = embed(q, DIM)
        # Long tasks have 56 episodes; use higher k to ensure key is retrieved
        top_k = 25 if task.task_type == "long" else 10
        similar = db.query_similar(emb, min_reward=0.0, top_k=top_k)
        context_parts = []
        for ep in similar:
            meta = ep.metadata
            if isinstance(meta, dict) and "text" in meta:
                context_parts.append(str(meta["text"]))
            else:
                context_parts.append(f"Recall: {ep.task_id}")
        context = "\n".join(context_parts) if context_parts else ""
        resp, tok = stub_policy(context, q)
        total_tokens += tok
        if grade_response(resp, expected):
            successes += 1
    elapsed_ms = (time.perf_counter() - start) * 1000
    success_rate = successes / len(task.recall_questions) if task.recall_questions else 0
    return EvalResult(
        task_id=task.task_id,
        variant="agent_mem_db",
        success=success_rate == 1.0,
        total_tokens=total_tokens,
        total_latency_ms=elapsed_ms,
        num_turns=turns,
        task_type=task.task_type,
        num_recall_correct=successes,
    )
