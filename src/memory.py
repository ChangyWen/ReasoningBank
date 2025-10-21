import numpy as np
import faiss
from typing import List, Optional, Callable, Tuple, Dict


class MemoryBank:
    def __init__(self, embedding_dim: int = 3072):
        self.embedding_dim = embedding_dim
        # a base index that performs inner-product (dot product) similarity search
        base = faiss.IndexFlatIP(self.embedding_dim)
        # a wrapper index that maps memory IDs to the base index
        self._index: faiss.IndexIDMap2 = faiss.IndexIDMap2(base)
        self._meta: Dict[int, Tuple[str, str]] = {}
        self._next_id: int = 0


    def _format_embedding(self, embedding: np.array) -> np.array:
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        embedding = embedding.astype(np.float32)
        if embedding.shape != (1, self.embedding_dim):
            embedding = embedding.reshape(1, -1)
            assert embedding.shape[1] == self.embedding_dim
        return np.ascontiguousarray(embedding)


    def add_memory(self, title: str, content: str, embedding: np.array):
        embedding = self._format_embedding(embedding)
        mem_id = self._next_id

        self._index.add_with_ids(embedding, np.array([mem_id], dtype=np.int64))
        self._meta[mem_id] = (title, content)
        self._next_id += 1


    def get_memory(self, query_embedding: np.array, top_k: int = 10) -> List[Tuple[str, str]]:
        if self._index.ntotal == 0 or top_k <= 0:
            return []
        query_embedding = self._format_embedding(query_embedding)
        _, ids = self._index.search(query_embedding, min(top_k, self._index.ntotal))
        valid_ids = [int(i) for i in ids[0] if i != -1]
        return [self._meta[i] for i in valid_ids]
