"""
Async API for AgentMemDB — non-blocking wrappers using asyncio.to_thread.

Use these methods when calling from async contexts (e.g. async web servers,
LangChain/LangGraph async chains) to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:
    from .agent_mem_db_py import AgentMemDB, Episode


def _to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run sync function in thread pool. Python 3.9+ uses asyncio.to_thread."""
    if sys.version_info >= (3, 9):
        return asyncio.to_thread(func, *args, **kwargs)
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


class AgentMemDBAsync:
    """
    Async wrapper for AgentMemDB. Delegates to sync methods via asyncio.to_thread.
    """

    def __init__(self, db: "AgentMemDB") -> None:
        self._db = db

    async def store_episode_async(self, episode: "Episode") -> None:
        """Store an episode without blocking the event loop."""
        await _to_thread(self._db.store_episode, episode)

    async def query_similar_async(
        self,
        state_embedding: List[float],
        min_reward: float = 0.0,
        top_k: int = 5,
        *,
        tags_any: Optional[List[str]] = None,
        tags_all: Optional[List[str]] = None,
        task_id_prefix: Optional[str] = None,
        time_after: Optional[int] = None,
        time_before: Optional[int] = None,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List["Episode"]:
        """Query similar episodes without blocking the event loop."""
        return await _to_thread(
            self._db.query_similar,
            state_embedding,
            min_reward,
            top_k,
            tags_any=tags_any,
            tags_all=tags_all,
            task_id_prefix=task_id_prefix,
            time_after=time_after,
            time_before=time_before,
            source=source,
            user_id=user_id,
        )

    async def save_to_file_async(self, path: str) -> None:
        """Save DB to file without blocking the event loop."""
        await _to_thread(self._db.save_to_file, path)

    async def prune_older_than_async(self, timestamp_cutoff_ms: int) -> int:
        """Prune episodes older than cutoff (Unix ms) without blocking the event loop."""
        return await _to_thread(
            self._db.prune_older_than, timestamp_cutoff_ms
        )

    async def prune_keep_newest_async(self, n: int) -> int:
        """Prune to keep only n most recent episodes without blocking the event loop."""
        return await _to_thread(self._db.prune_keep_newest, n)

    async def prune_keep_highest_reward_async(self, n: int) -> int:
        """Prune to keep only n highest-reward episodes without blocking the event loop."""
        return await _to_thread(self._db.prune_keep_highest_reward, n)

    @staticmethod
    async def load_from_file_async(path: str) -> "AgentMemDB":
        """Load DB from file without blocking the event loop."""
        from .agent_mem_db_py import AgentMemDB

        return await _to_thread(AgentMemDB.load_from_file, path)
