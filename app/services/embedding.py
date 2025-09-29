# Copyright 2024
# Directory: yt-rag/app/services/embedding.py

"""
Embedding service for generating text embeddings using HuggingFace.
Focused solely on vector embeddings for RAG retrieval.
"""

import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using HuggingFace."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """Initialize HuggingFace embedding model."""
        self.embed_model = HuggingFaceEmbeddings(model_name=model_name)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (1024 dimensions for BAAI/bge-m3)
        """
        try:
            embeddings = self.embed_model.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        embeddings = self.embed_model.embed_query(query)
        return embeddings


# Global service instance
embedding_service = EmbeddingService()
