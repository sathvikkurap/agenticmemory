"""
Shared base for agent variants: task definitions, stub policy, embedding.
"""

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any


STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "to", "of", "for", "in", "on", "at", "by", "with", "user", "user's", "remember", "what", "how", "when", "which", "that", "this", "it", "and", "or"}


def embed(text: str, dim: int = 16) -> list[float]:
    """Bag-of-words style: words map to deterministic vectors; texts with shared words have similar embeddings."""
    words = set(w.lower().strip(".,?!") for w in text.split() if len(w) > 1)
    words -= STOPWORDS
    vec = [0.0] * dim
    for w in sorted(words):
        random.seed(hash(w) & 0xFFFFFFFF)
        for i in range(dim):
            vec[i] += random.uniform(-1, 1)
    if sum(abs(x) for x in vec) < 1e-6:
        vec[0] = 1.0
    # Normalize so Euclidean distance reflects cosine-like similarity
    norm = (sum(x * x for x in vec)) ** 0.5
    if norm > 1e-6:
        vec = [x / norm for x in vec]
    return vec


@dataclass
class Task:
    """A single evaluation task: store phase + recall phase."""
    task_id: str
    store_statements: list[str]  # "Remember: user prefers X"
    recall_questions: list[str]  # "What does user prefer?"
    expected_answers: list[list[str]]  # For each question, list of acceptable answers
    task_type: str = "short"  # "short" or "long" (long = key at start + many filler; naive truncation drops key)


def make_tasks(n: int = 40) -> list[Task]:
    """Generate n tasks where memory matters. Some have long histories to stress naive truncation."""
    templates: list[tuple[list[str], list[str], list[list[str]]]] = [
        (["Remember: user prefers dark mode for the display."], ["What does user prefer for display?"], [["dark mode"]]),
        (["Remember: user's favorite color is blue."], ["What is user's favorite color?"], [["blue"]]),
        (["Remember: user wants notifications disabled."], ["Should we send notifications?"], [["disabled", "no"]]),
        (["Remember: user prefers Python over JavaScript."], ["Which language does user prefer?"], [["Python"]]),
        (["Remember: user's timezone is PST."], ["What timezone is user in?"], [["PST"]]),
        (["Remember: user likes coffee with oat milk."], ["How does user like their coffee?"], [["oat milk"]]),
        (["Remember: user's name is Alex."], ["What is the user's name?"], [["Alex"]]),
        (["Remember: user prefers meetings before noon."], ["When does user prefer meetings?"], [["noon", "before noon"]]),
        (["Remember: user's project is called ProjectX."], ["What is user's project called?"], [["ProjectX"]]),
        (["Remember: user wants weekly summaries on Monday."], ["When does user want summaries?"], [["Monday", "weekly"]]),
    ]
    # Long-history tasks: key statement FIRST, then filler. Naive truncation keeps last N chars and DROPS the key.
    filler = "This is filler statement number {} to pad the context and test truncation."
    long_templates: list[tuple[list[str], list[str], list[list[str]]]] = [
        (["Remember: user's secret code is 42."] + [filler.format(j) for j in range(1, 55)], ["What is the secret code?"], [["42"]]),
        (["Remember: user's VIP status is Gold."] + [filler.format(j) for j in range(1, 55)], ["What is user's VIP status?"], [["Gold"]]),
        (["Remember: user's backup email is backup@example.com."] + [filler.format(j) for j in range(1, 55)], ["What is the backup email?"], [["backup@example.com"]]),
    ]
    all_templates = templates + long_templates
    task_types = ["short"] * len(templates) + ["long"] * len(long_templates)
    tasks = []
    for i in range(n):
        idx = i % len(all_templates)
        t = all_templates[idx]
        tt = task_types[idx]
        tasks.append(Task(
            task_id=f"task_{i}",
            store_statements=list(t[0]),
            recall_questions=list(t[1]),
            expected_answers=list(t[2]),
            task_type=tt,
        ))
    return tasks


def stub_policy(context: str, question: str) -> tuple[str, int]:
    """
    Simulated policy: if expected answer appears in context, return it; else guess.
    Returns (response, token_count).
    """
    # Simulate token count: rough chars/4
    context_lower = context.lower()
    question_lower = question.lower()
    # Simple extraction: look for "prefer X", "favorite X", "name is X", etc.
    if "prefer" in question_lower or "preference" in question_lower:
        for word in ["dark mode", "light mode", "oat milk", "python", "javascript", "noon", "monday"]:
            if word in context_lower:
                return word, len(context) // 4 + len(question) // 4 + 20
    if "color" in question_lower and "blue" in context_lower:
        return "blue", len(context) // 4 + len(question) // 4 + 10
    if "notification" in question_lower and "disabled" in context_lower:
        return "disabled", len(context) // 4 + len(question) // 4 + 10
    if "timezone" in question_lower and "pst" in context_lower:
        return "PST", len(context) // 4 + len(question) // 4 + 10
    if "name" in question_lower and "alex" in context_lower:
        return "Alex", len(context) // 4 + len(question) // 4 + 10
    if "project" in question_lower and "projectx" in context_lower:
        return "ProjectX", len(context) // 4 + len(question) // 4 + 10
    if "coffee" in question_lower and "oat milk" in context_lower:
        return "oat milk", len(context) // 4 + len(question) // 4 + 15
    if "meeting" in question_lower and "noon" in context_lower:
        return "before noon", len(context) // 4 + len(question) // 4 + 15
    if "summar" in question_lower and "monday" in context_lower:
        return "Monday", len(context) // 4 + len(question) // 4 + 10
    if "secret code" in question_lower and "42" in context_lower:
        return "42", len(context) // 4 + len(question) // 4 + 5
    if "vip" in question_lower and "gold" in context_lower:
        return "Gold", len(context) // 4 + len(question) // 4 + 5
    if "backup email" in question_lower and "backup@example.com" in context_lower:
        return "backup@example.com", len(context) // 4 + len(question) // 4 + 20
    # No match
    return "I don't know", len(context) // 4 + len(question) // 4 + 5


def grade_response(response: str, expected: list[str]) -> bool:
    """Check if response matches any expected answer (case-insensitive substring)."""
    r = response.lower().strip()
    for exp in expected:
        if exp.lower() in r:
            return True
    return False


@dataclass
class EvalResult:
    """Result for one task run."""
    task_id: str
    variant: str
    success: bool
    total_tokens: int
    total_latency_ms: float
    num_turns: int
    task_type: str = "short"
    num_recall_correct: int = 0  # For partial success tracking


