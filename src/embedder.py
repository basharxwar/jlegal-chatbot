"""
embedder.py — Arabic semantic embedding with switchable model.

Default model: UBC-NLP/MARBERT (handles both MSA and Levantine/Jordanian dialect).
Override via environment variable: EMBEDDING_MODEL=aubmindlab/bert-base-arabertv02

Both models use explicit mean pooling via the sentence-transformers pipeline.
"""

import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, models as st_models

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "aubmindlab/bert-base-arabertv02")

_model = None


def get_model() -> SentenceTransformer:
    """Load the configured model with mean pooling. Cached after first call."""
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {MODEL_NAME}")
        word_embedding = st_models.Transformer(MODEL_NAME, max_seq_length=512)
        pooling = st_models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        _model = SentenceTransformer(modules=[word_embedding, pooling])
        print(f"[embedder] Model loaded OK  (dim={word_embedding.get_word_embedding_dimension()})")
    return _model


def embed_text(text: str) -> list:
    model = get_model()
    vector = model.encode([text], convert_to_numpy=True)[0]
    return vector.tolist()


def embed_texts(texts: list) -> np.ndarray:
    model = get_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=16,
        show_progress_bar=True,
    )
    return vectors
