"""
No-memory baseline agent. Each turn is stateless; cannot answer recall questions.
"""

import time
from .base import Task, EvalResult, stub_policy, grade_response


def run_task(task: Task) -> EvalResult:
    """No-memory: context is empty for recall."""
    start = time.perf_counter()
    total_tokens = 0
    successes = 0
    turns = 0
    for _ in task.store_statements:
        turns += 1
        total_tokens += 50
    for q, expected in zip(task.recall_questions, task.expected_answers):
        turns += 1
        resp, tok = stub_policy("", q)
        total_tokens += tok
        if grade_response(resp, expected):
            successes += 1
    elapsed_ms = (time.perf_counter() - start) * 1000
    success_rate = successes / len(task.recall_questions) if task.recall_questions else 0
    return EvalResult(
        task_id=task.task_id,
        variant="no_memory",
        success=success_rate == 1.0,
        total_tokens=total_tokens,
        total_latency_ms=elapsed_ms,
        num_turns=turns,
        task_type=task.task_type,
        num_recall_correct=successes,
    )
