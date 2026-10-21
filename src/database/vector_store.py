import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, List, Tuple, Union

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import settings


class VectorStore:
    """Manage local embeddings and vector search using Qdrant."""

    def __init__(self) -> None:
        self.settings = settings
        self.embedding_model = self.settings.embedding_model
        qdrant_path: Path | None = None

        if self.settings.qdrant_mode == "embedded":
            qdrant_path = Path(self.settings.qdrant_path).expanduser()
            if not qdrant_path.is_absolute():
                qdrant_path = Path(__file__).resolve().parents[2] / qdrant_path
            qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        elif self.settings.qdrant_mode == "server":
            if not self.settings.qdrant_url.strip():
                raise ValueError("QDRANT_URL must be set when QDRANT_MODE=server.")
        else:
            raise ValueError("QDRANT_MODE must be 'embedded' or 'server'.")

        self.embedder = SentenceTransformer(self.embedding_model)
        self.embedding_dimensions = self.embedder.get_embedding_dimension()
        if qdrant_path is not None:
            self.vec_client = QdrantClient(path=str(qdrant_path))
        else:
            self.vec_client = QdrantClient(url=self.settings.qdrant_url)

    def get_embedding(self, text: str) -> List[float]:
        """Generate a local embedding for the supplied text."""
        start_time = time.time()
        embedding = self.embedder.encode(text.replace("\n", " "))
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        logging.info("Embedding generated in %.3f seconds", time.time() - start_time)
        return embedding

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate local embeddings for a batch of texts."""
        if not texts:
            raise ValueError("No text was provided for embedding.")

        start_time = time.time()
        embeddings = self.embedder.encode(
            [text.replace("\n", " ") for text in texts],
            show_progress_bar=True,
        )
        result = [
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            for embedding in embeddings
        ]
        logging.info(
            "Generated %d embeddings in %.3f seconds",
            len(result),
            time.time() - start_time,
        )
        return result

    def ensure_collection(self, collection_name: str) -> None:
        """Create a Qdrant collection if needed and verify its vector dimensions."""
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        if self.vec_client.collection_exists(collection_name):
            collection = self.vec_client.get_collection(collection_name)
            vectors = collection.config.params.vectors
            current_size = getattr(vectors, "size", None)
            if current_size != self.embedding_dimensions:
                raise ValueError(
                    f"Collection '{collection_name}' has vector size "
                    f"{current_size}; the configured embedding model produces "
                    f"vectors of size {self.embedding_dimensions}. The collection "
                    "was not changed."
                )
            return

        self.vec_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    def create_tables(self, collection_name: str) -> None:
        """Ensure the named collection exists."""
        self.ensure_collection(collection_name)

    def upsert(
        self,
        df: pd.DataFrame,
        *,
        collection_name: str,
    ) -> None:
        """Insert or update records with columns id, metadata, contents, embedding."""
        required_columns = {"id", "metadata", "contents", "embedding"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required upsert columns: {', '.join(sorted(missing_columns))}"
            )
        if df.empty:
            return

        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        points = []
        for record in df.to_dict(orient="records"):
            metadata = record["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("Each record's metadata must be a dictionary.")

            embedding = record["embedding"]
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            if len(embedding) != self.embedding_dimensions:
                raise ValueError(
                    f"Embedding has dimension {len(embedding)}; expected "
                    f"{self.embedding_dimensions}."
                )

            points.append(
                models.PointStruct(
                    id=record["id"],
                    vector=embedding,
                    payload={
                        "metadata": metadata,
                        "content": record["contents"],
                    },
                )
            )

        self.ensure_collection(collection_name)
        self.vec_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logging.info("Inserted %d records into %s", len(points), collection_name)

    @staticmethod
    def _make_metadata_filter(
        metadata_filter: Union[dict, List[dict]],
    ) -> models.Filter:
        def make_conditions(metadata: dict) -> list[models.FieldCondition]:
            if not metadata:
                raise ValueError("Metadata filters cannot be empty.")
            return [
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value),
                )
                for key, value in metadata.items()
            ]

        if isinstance(metadata_filter, dict):
            return models.Filter(must=make_conditions(metadata_filter))
        if isinstance(metadata_filter, list) and metadata_filter:
            return models.Filter(
                should=[
                    models.Filter(must=make_conditions(item))
                    for item in metadata_filter
                ]
            )
        raise ValueError("metadata_filter must be a dictionary or a non-empty list.")

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        return_dataframe: bool = True,
        *,
        collection_name: str | Sequence[str],
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """Search explicitly selected collections and return the best matches."""
        if limit < 1:
            raise ValueError("limit must be greater than zero.")

        if isinstance(collection_name, str):
            target_collections = [collection_name]
        else:
            target_collections = list(dict.fromkeys(collection_name))

        if not target_collections or any(not name.strip() for name in target_collections):
            raise ValueError("Provide at least one non-empty collection name.")

        missing_collections = [
            name
            for name in target_collections
            if not self.vec_client.collection_exists(name)
        ]
        if missing_collections:
            raise ValueError(
                "Qdrant collection(s) do not exist: "
                f"{', '.join(missing_collections)}. Index documents before querying."
            )

        query_embedding = self.get_embedding(query_text)
        start_time = time.time()
        search_args: dict[str, Any] = {}
        if metadata_filter is not None:
            search_args["query_filter"] = self._make_metadata_filter(metadata_filter)

        results: List[Tuple[Any, ...]] = []
        for target_collection in target_collections:
            response = self.vec_client.query_points(
                collection_name=target_collection,
                query=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=True,
                **search_args,
            )
            results.extend(
                (
                    point.id,
                    {
                        **(point.payload or {}).get("metadata", {}),
                        "collection_name": target_collection,
                    },
                    (point.payload or {}).get("content"),
                    point.vector,
                    1 - point.score,
                )
                for point in response.points
            )

        results.sort(key=lambda result: result[4])
        results = results[:limit]
        logging.info(
            "Vector search across %d collection(s) completed in %.3f seconds",
            len(target_collections),
            time.time() - start_time,
        )
        return self._create_dataframe_from_results(results) if return_dataframe else results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """Convert Qdrant search results to the existing tabular result shape."""
        df = pd.DataFrame(
            results,
            columns=["id", "metadata", "content", "embedding", "distance"],
        )
        if df.empty:
            return df
        metadata = pd.json_normalize(df.pop("metadata"))
        df = pd.concat([df, metadata], axis=1)
        df["id"] = df["id"].astype(str)
        return df

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: Union[dict, List[dict]] = None,
        delete_all: bool = False,
        *,
        collection_name: str,
    ) -> None:
        """Delete by point IDs, metadata filter, or all points in the collection."""
        if sum(bool(value) for value in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")
        if not self.vec_client.collection_exists(collection_name):
            raise ValueError(f"Qdrant collection '{collection_name}' does not exist.")

        if delete_all:
            selector = models.FilterSelector(filter=models.Filter())
        elif ids:
            selector = models.PointIdsList(points=ids)
        else:
            selector = models.FilterSelector(
                filter=self._make_metadata_filter(metadata_filter)
            )

        self.vec_client.delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=True,
        )

    def close(self) -> None:
        self.vec_client.close()
