"""
FAISS semantic retriever for the SHL catalog.

Implements:
- Load catalog items from `catalog.json`
- Build a combined text per assessment for embedding:
  name + description + keys + job_levels
- Embed using `sentence-transformers` locally (no API calls)
- Build a FAISS index (cosine similarity via normalized inner product)
- Persist:
  - `faiss_index.pkl` (serialized FAISS index bytes + metadata)
  - `catalog_data.pkl` (the full catalog list; used to return full objects)
- Search:
  - input: query string
  - output: top 10 most relevant *full assessment objects*

Expected catalog item fields (per user):
entity_id, name, link, description, job_levels, duration, remote, adaptive,
keys, languages
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class RetrieverConfig:
    """
    Configuration for retrieval.
    """

    catalog_path: Path = Path("catalog.json")
    top_k: int = 10

    # Persistence targets (requested file names)
    faiss_index_path: Path = Path("faiss_index.pkl")
    catalog_data_path: Path = Path("catalog_data.pkl")

    # Embeddings model
    embedding_model: str = "paraphrase-MiniLM-L3-v2"


def _as_list(value: Any) -> List[str]:
    """
    Normalize catalog fields that may be missing / string / list into a list of strings.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return [str(value).strip()] if str(value).strip() else []


def _combine_text(assessment: Dict[str, Any]) -> str:
    """
    Create a combined text for embedding:
    name + description + keys + job_levels
    """
    name = str(assessment.get("name", "")).strip()
    description = str(assessment.get("description", "")).strip()
    keys = ", ".join(_as_list(assessment.get("keys")))
    job_levels = ", ".join(_as_list(assessment.get("job_levels")))

    parts = [
        name,
        description,
        f"keys: {keys}" if keys else "",
        f"job_levels: {job_levels}" if job_levels else "",
    ]
    return "\n".join([p for p in parts if p])


_EMBED_MODEL: Optional[SentenceTransformer] = None


def _get_embedder(model_name: str) -> SentenceTransformer:
    """
    Lazy-initialize a local sentence-transformers model.

    Notes:
    - Model weights download once (~80MB) and then run locally.
    - No API keys, batching, retries, or sleeps required.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print("Loading embedding model...")
        _EMBED_MODEL = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    return _EMBED_MODEL


def _l2_normalize(vectors: "faiss.Float32Matrix") -> None:
    """
    In-place L2 normalization for cosine similarity with IndexFlatIP.
    """
    faiss.normalize_L2(vectors)


def _to_faiss_matrix(vectors: Sequence[Sequence[float]]) -> "faiss.Float32Matrix":
    """
    Convert python vectors to a FAISS float32 matrix.
    """
    # faiss requires float32 contiguous arrays; this helper keeps dependencies minimal
    import numpy as np

    mat = np.asarray(vectors, dtype="float32")
    if mat.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")
    return mat


class CatalogRetriever:
    """
    FAISS-backed catalog retriever.

    Notes:
    - Uses cosine similarity by normalizing embeddings and searching with inner product.
    - Stores full catalog objects separately so search can return full records.
    """

    def __init__(self, config: Optional[RetrieverConfig] = None) -> None:
        self.config = config or RetrieverConfig()

        self._index: Optional[faiss.Index] = None
        self._catalog: Optional[List[Dict[str, Any]]] = None
        self._dimension: Optional[int] = None

    # --------- Catalog / index IO ---------

    def load_catalog(self) -> List[Dict[str, Any]]:
        """
        Load the catalog JSON array from disk.
        """
        if not self.config.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog file not found at `{self.config.catalog_path}`."
            )
        # Some real-world catalog exports can contain unescaped control characters.
        # Use `strict=False` to tolerate these while parsing.
        text = self.config.catalog_path.read_text(encoding="utf-8")
        data = json.loads(text, strict=False)
        if not isinstance(data, list):
            raise ValueError("`catalog.json` must contain a JSON array of objects.")
        # Defensive normalization: ensure each item is a dict
        catalog: List[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict):
                catalog.append(item)
        if not catalog:
            raise ValueError("Catalog is empty or has no valid objects.")
        return catalog

    def build_and_save_index(self) -> None:
        """
        Build the FAISS index from `catalog.json` and save to disk.
        """
        catalog = self.load_catalog()
        texts = [_combine_text(a) for a in catalog]

        embedder = _get_embedder(self.config.embedding_model)

        # For building index (exactly as requested)
        vectors = embedder.encode(texts, show_progress_bar=True)

        mat = _to_faiss_matrix(vectors)
        _l2_normalize(mat)

        dim = int(mat.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(mat)

        # Persist index bytes + basic metadata
        index_bytes = faiss.serialize_index(index)
        payload = {
            "dimension": dim,
            "index_bytes": index_bytes,
            "metric": "cosine_via_ip",
            "embedding_model": self.config.embedding_model,
        }

        self.config.faiss_index_path.write_bytes(pickle.dumps(payload))
        self.config.catalog_data_path.write_bytes(pickle.dumps(catalog))

        # Keep in-memory copies too
        self._index = index
        self._catalog = catalog
        self._dimension = dim

    def load_index(self) -> None:
        """
        Load FAISS index + catalog data from disk.
        """
        if not self.config.faiss_index_path.exists() or not self.config.catalog_data_path.exists():
            raise FileNotFoundError(
                "Missing index files. Run `python retriever.py` to build "
                f"`{self.config.faiss_index_path.name}` and `{self.config.catalog_data_path.name}`."
            )

        payload = pickle.loads(self.config.faiss_index_path.read_bytes())
        catalog = pickle.loads(self.config.catalog_data_path.read_bytes())

        if not isinstance(payload, dict) or "index_bytes" not in payload:
            raise ValueError("Invalid `faiss_index.pkl` payload.")
        if not isinstance(catalog, list):
            raise ValueError("Invalid `catalog_data.pkl` payload.")

        index = faiss.deserialize_index(payload["index_bytes"])

        # Dimension safety check: if the stored FAISS index was built with a different
        # embedding model/dimension, rebuild automatically instead of crashing later.
        embedder = _get_embedder(self.config.embedding_model)
        try:
            model_dim = int(embedder.get_sentence_embedding_dimension())
        except Exception:
            # Fallback: derive dimension from a tiny encode if needed.
            v0 = embedder.encode(["dimension probe"])[0]
            model_dim = int(len(v0))

        if int(getattr(index, "d", 0)) != model_dim:
            print(
                "FAISS index dimension mismatch detected. "
                f"index_dim={getattr(index, 'd', None)} model_dim={model_dim}. "
                "Rebuilding index now..."
            )
            self.build_and_save_index()
            return

        self._index = index
        self._catalog = catalog
        self._dimension = int(payload.get("dimension") or index.d)

    # --------- Search ---------

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Semantic search over the catalog.

        Args:
            query: user query string
            top_k: number of results (defaults to config.top_k; requested default is 10)

        Returns:
            Top-k full assessment objects (dicts) from the catalog.
        """
        if not query or not query.strip():
            return []

        if self._index is None or self._catalog is None:
            # Prefer load from disk (fast path) rather than always rebuilding
            self.load_index()

        assert self._index is not None
        assert self._catalog is not None

        embedder = _get_embedder(self.config.embedding_model)

        # For search query embedding (exactly as requested)
        query_vector = embedder.encode([query.strip()])[0]

        import numpy as np

        q = np.asarray([query_vector], dtype="float32")
        _l2_normalize(q)

        k = int(top_k or self.config.top_k)
        k = max(1, min(k, len(self._catalog)))

        scores, ids = self._index.search(q, k)
        # ids shape: (1, k)
        result: List[Dict[str, Any]] = []
        for idx in ids[0].tolist():
            if idx == -1:
                continue
            if 0 <= idx < len(self._catalog):
                result.append(self._catalog[idx])
        return result

    # Backward-compatible alias (older name from scaffold)
    def semantic_search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.search(query=query, top_k=top_k)


def main() -> None:
    """
    Build and save the FAISS index files next to `catalog.json`.

    Output files:
    - faiss_index.pkl
    - catalog_data.pkl
    """
    retriever = CatalogRetriever()
    retriever.build_and_save_index()
    print(
        "Index build complete. Wrote "
        f"`{retriever.config.faiss_index_path.name}` and `{retriever.config.catalog_data_path.name}`."
    )


if __name__ == "__main__":
    main()

