"""DINNNER: Distribution-Informed Neural Network for Named Entity Recognition (demo package)."""

from .config import DEFAULT_MODEL, MODELS, ModelSpec
from .inference import Entity, SentenceResult, predict, split_words
from .model import DINNNER, load_model

__all__ = [
    "DEFAULT_MODEL", "MODELS", "ModelSpec",
    "Entity", "SentenceResult", "predict", "split_words",
    "DINNNER", "load_model",
]
