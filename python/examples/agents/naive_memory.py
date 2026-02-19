"""
Naive memory baseline: full conversation history with truncation.
"""

import time
from .base import Task, EvalResult, stub_policy, grade_response

MAX_CONTEXT_TOKENS = 256  # Simulated context window (naive truncation)


def run_task(task: Task) -> EvalResult:
    """Naive: keep full history, truncate to last N tokens."""
    start = time.perf_counter()
    history: list[str] = []
    total_tokens = 0
    successes = 0
    turns = 0
    for stmt in task.store_statements:
        history.append(f"User: {stmt}")
        turns += 1
        total_tokens += 50
    for q, expected in zip(task.recall_questions, task.expected_answers):
        turns += 1
        context = "\n".join(history)
        if len(context) > MAX_CONTEXT_TOKENS * 4:
            context = context[-(MAX_CONTEXT_TOKENS * 4):]
        resp, tok = stub_policy(context, q)
        total_tokens += tok
        if grade_response(resp, expected):
            successes += 1
        history.append(f"User: {q}\nAssistant: {resp}")
    elapsed_ms = (time.perf_counter() - start) * 1000
    success_rate = successes / len(task.recall_questions) if task.recall_questions else 0
    return EvalResult(
        task_id=task.task_id,
        variant="naive",
        success=success_rate == 1.0,
        total_tokens=total_tokens,
        total_latency_ms=elapsed_ms,
        num_turns=turns,
        task_type=task.task_type,
        num_recall_correct=successes,
    )
