"""AgentMemDBStore: LangGraph BaseStore backed by AgentMemDB for episodic memory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import agent_mem_db_py as agent_mem_db

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        IndexConfig,
        Item,
        ListNamespacesOp,
        MatchCondition,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
        ensure_embeddings,
        get_text_at_path,
        tokenize_path,
    )
except ImportError as e:
    raise ImportError(
        "LangGraph integration requires langgraph. Install with: pip install langgraph"
    ) from e


def _compare_values(item_value: Any, filter_value: Any) -> bool:
    """Compare values for filter matching (simplified JSONB-like)."""
    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(
                _apply_operator(item_value, op_key, op_value)
                for op_key, op_value in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(
            _compare_values(item_value.get(k), v) for k, v in filter_value.items()
        )
    elif isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(
                _compare_values(iv, fv)
                for iv, fv in zip(item_value, filter_value, strict=False)
            )
        )
    return item_value == filter_value


def _apply_operator(value: Any, operator: str, op_value: Any) -> bool:
    """Apply comparison operator."""
    if operator == "$eq":
        return value == op_value
    if operator == "$ne":
        return value != op_value
    if operator == "$gt":
        return float(value) > float(op_value)
    if operator == "$gte":
        return float(value) >= float(op_value)
    if operator == "$lt":
        return float(value) < float(op_value)
    if operator == "$lte":
        return float(value) <= float(op_value)
    raise ValueError(f"Unsupported operator: {operator}")


def _does_match(match_condition: MatchCondition, key: tuple[str, ...]) -> bool:
    """Whether a namespace matches the match condition."""
    match_type = match_condition.match_type
    path = match_condition.path
    if len(key) < len(path):
        return False
    if match_type == "prefix":
        for k_elem, p_elem in zip(key, path, strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    if match_type == "suffix":
        for k_elem, p_elem in zip(reversed(key), reversed(path), strict=False):
            if p_elem == "*":
                continue
            if k_elem != p_elem:
                return False
        return True
    raise ValueError(f"Unsupported match type: {match_type}")


def _namespace_prefix_match(namespace: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    """True if namespace starts with prefix."""
    if len(namespace) < len(prefix):
        return False
    return namespace[: len(prefix)] == prefix


class AgentMemDBStore(BaseStore):
    """LangGraph BaseStore backed by AgentMemDB for episodic long-term memory.

    Supports put/get/delete and semantic search when configured with embeddings.
    Uses AgentMemDB (HNSW) for vector similarity; key-value for direct access.
    """

    __slots__ = ("_data", "_task_to_key", "_db", "index_config", "embeddings")

    def __init__(self, *, index: IndexConfig | None = None) -> None:
        if not index or "dims" not in index:
            raise ValueError("index with 'dims' is required for AgentMemDBStore")
        dims = index["dims"]
        self.index_config = index.copy()
        self.embeddings = ensure_embeddings(self.index_config.get("embed"))
        self.index_config["__tokenized_fields"] = [
            (p, tokenize_path(p)) if p != "$" else (p, p)
            for p in (self.index_config.get("fields") or ["$"])
        ]
        self._db = agent_mem_db.AgentMemDB.with_max_elements(dims, 50_000)
        self._data: dict[tuple[str, ...], dict[str, Item]] = defaultdict(dict)
        self._task_to_key: dict[str, tuple[tuple[str, ...], str]] = {}

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        put_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        search_ops: list[tuple[int, SearchOp]] = []

        for i, op in enumerate(ops):
            if isinstance(op, GetOp):
                item = self._data.get(op.namespace, {}).get(op.key)
                results.append(item)
            elif isinstance(op, SearchOp):
                search_ops.append((i, op))
                results.append(None)
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            elif isinstance(op, PutOp):
                put_ops[(op.namespace, op.key)] = op
                results.append(None)
            else:
                raise ValueError(f"Unknown operation: {type(op)}")

        if put_ops:
            self._apply_puts(put_ops)

        for idx, op in search_ops:
            results[idx] = self._search(op)

        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return self.batch(ops)

    def _task_id(self, namespace: tuple[str, ...], key: str) -> str:
        return "|".join(namespace) + "|" + key

    def _apply_puts(self, put_ops: dict[tuple[tuple[str, ...], str], PutOp]) -> None:
        to_embed: dict[str, list[tuple[tuple[str, ...], str, str]]] = defaultdict(list)
        now = datetime.now(timezone.utc)

        for (namespace, key), op in put_ops.items():
            if op.value is None:
                self._task_to_key.pop(self._task_id(namespace, key), None)
                self._data[namespace].pop(key, None)
                continue

            self._task_to_key.pop(self._task_id(namespace, key), None)
            item = Item(
                value=op.value,
                key=key,
                namespace=namespace,
                created_at=now,
                updated_at=now,
            )
            self._data[namespace][key] = item

            if op.index is not False and self.embeddings:
                paths = (
                    self.index_config["__tokenized_fields"]
                    if op.index is None
                    else [(p, tokenize_path(p)) for p in op.index]
                )
                for path, field in paths:
                    texts = get_text_at_path(op.value, field)
                    if texts:
                        for j, text in enumerate(texts):
                            to_embed[text].append(
                                (namespace, key, f"{path}.{j}" if len(texts) > 1 else path)
                            )

        if to_embed and self.embeddings:
            texts_ordered = list(to_embed.keys())
            embeddings = self.embeddings.embed_documents(texts_ordered)
            for text, emb in zip(texts_ordered, embeddings, strict=False):
                for (namespace, key, _path) in to_embed[text]:
                    meta = {"__namespace": list(namespace), "__key": key}
                    task_id = "|".join(namespace) + "|" + key
                    ep = agent_mem_db.Episode(
                        task_id=task_id,
                        state_embedding=[float(x) for x in emb],
                        reward=1.0,
                        metadata=meta,
                    )
                    self._db.store_episode(ep)
                    self._task_to_key[task_id] = (namespace, key)

    def _search(self, op: SearchOp) -> list[SearchItem]:
        namespace_prefix = op.namespace_prefix

        def filter_fn(item: Item) -> bool:
            if not _namespace_prefix_match(item.namespace, namespace_prefix):
                return False
            if not op.filter:
                return True
            return all(
                _compare_values(item.value.get(k), v)
                for k, v in op.filter.items()
            )

        if op.query and self.embeddings:
            query_emb = self.embeddings.embed_query(op.query)
            hits = self._db.query_similar(
                [float(x) for x in query_emb],
                min_reward=0.0,
                top_k=op.limit + op.offset + 100,
            )
            seen: set[tuple[tuple[str, ...], str]] = set()
            candidates: list[tuple[Item, float | None]] = []
            for ep in hits:
                key_info = self._task_to_key.get(ep.task_id)
                if not key_info:
                    continue
                namespace, key = key_info
                if (namespace, key) in seen:
                    continue
                item = self._data.get(namespace, {}).get(key)
                if not item or not filter_fn(item):
                    continue
                seen.add((namespace, key))
                candidates.append((item, None))
            items = candidates[op.offset : op.offset + op.limit]
        else:
            all_items: list[Item] = []
            for ns in self._data:
                if not _namespace_prefix_match(ns, namespace_prefix):
                    continue
                for item in self._data[ns].values():
                    if filter_fn(item):
                        all_items.append(item)
            items = [
                (it, None)
                for it in all_items[op.offset : op.offset + op.limit]
            ]

        return [
            SearchItem(
                namespace=item.namespace,
                key=item.key,
                value=item.value,
                created_at=item.created_at,
                updated_at=item.updated_at,
                score=score,
            )
            for item, score in items
        ]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces = list(self._data.keys())
        if op.match_conditions:
            namespaces = [
                ns
                for ns in namespaces
                if all(_does_match(c, ns) for c in op.match_conditions)
            ]
        if op.max_depth is not None:
            namespaces = sorted({ns[: op.max_depth] for ns in namespaces})
        else:
            namespaces = sorted(namespaces)
        return namespaces[op.offset : op.offset + op.limit]
