"""AgentMemDB as a LangChain VectorStore backend."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import agent_mem_db_py as agent_mem_db

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. Install with: pip install langchain-core"
    ) from e


class AgentMemDBVectorStore(VectorStore):
    """VectorStore backed by AgentMemDB for episodic memory with HNSW retrieval."""

    def __init__(
        self,
        embedding: Embeddings,
        dim: Optional[int] = None,
        use_exact: bool = False,
        max_elements: int = 20_000,
    ):
        """Initialize with an embedding model.

        Args:
            embedding: LangChain Embeddings model (e.g., OpenAIEmbeddings).
            dim: Embedding dimension. If None, inferred from embedding.embed_documents([""]).
            use_exact: Use exact (brute-force) search instead of HNSW.
            max_elements: Max episodes for HNSW (ignored if use_exact).
        """
        self.embedding = embedding
        if dim is None:
            # Infer dim from embedding
            test_emb = embedding.embed_documents([""])
            dim = len(test_emb[0])
        self.dim = dim
        if use_exact:
            self._db = agent_mem_db.AgentMemDB.exact(dim)
        else:
            self._db = agent_mem_db.AgentMemDB.with_max_elements(dim, max_elements)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        dim: Optional[int] = None,
        use_exact: bool = False,
        **kwargs: Any,
    ) -> "AgentMemDBVectorStore":
        """Create store from texts. Convenience constructor."""
        store = cls(embedding=embedding, dim=dim, use_exact=use_exact, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the store. Returns list of episode IDs."""
        texts = list(texts)
        if metadatas is None:
            metadatas = [{}] * len(texts)
        if len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts")
        embeddings = self.embedding.embed_documents(texts)
        if len(embeddings) != len(texts):
            raise ValueError("embedding count must match texts")
        for i, emb in enumerate(embeddings):
            if len(emb) != self.dim:
                raise ValueError(
                    f"Embedding dim {len(emb)} != expected {self.dim}"
                )
        result_ids = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            meta = dict(metadatas[i]) if metadatas else {}
            meta["text"] = text
            ep = agent_mem_db.Episode(
                task_id=ids[i] if ids and i < len(ids) else f"doc_{i}",
                state_embedding=emb,
                reward=1.0,
                metadata=meta,
            )
            self._db.store_episode(ep)
            result_ids.append(ep.id)
        return result_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the query."""
        emb = self.embedding.embed_query(query)
        if len(emb) != self.dim:
            raise ValueError(f"Query embedding dim {len(emb)} != expected {self.dim}")
        tags_any = None
        time_after = None
        time_before = None
        if filter:
            tags_any = filter.get("tags_any") or filter.get("tags")
            time_after = filter.get("time_after")
            time_before = filter.get("time_before")
        hits = self._db.query_similar(
            emb,
            min_reward=0.0,
            top_k=k,
            tags_any=tags_any,
            time_after=time_after,
            time_before=time_before,
        )
        docs = []
        for ep in hits:
            meta = ep.metadata if isinstance(ep.metadata, dict) else {}
            text = meta.get("text", str(meta))
            doc_meta = {"id": ep.id}
            if isinstance(meta, dict):
                doc_meta.update(meta)
            docs.append(Document(page_content=text, metadata=doc_meta))
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the embedding vector."""
        if len(embedding) != self.dim:
            raise ValueError(f"Embedding dim {len(embedding)} != expected {self.dim}")
        hits = self._db.query_similar(embedding, min_reward=0.0, top_k=k)
        docs = []
        for ep in hits:
            meta = ep.metadata if isinstance(ep.metadata, dict) else {}
            text = meta.get("text", str(meta))
            doc_meta = {"id": ep.id}
            if isinstance(meta, dict):
                doc_meta.update(meta)
            docs.append(Document(page_content=text, metadata=doc_meta))
        return docs
