from pathlib import Path

from src.config import MODEL_PATH, PREPROCESSOR_PATH
from src.utils import save_pickle, load_pickle


def save_model(model) -> None:
    """
    Save trained model.
    """
    save_pickle(model, MODEL_PATH)


def load_model():
    """
    Load trained model.
    """
    return load_pickle(MODEL_PATH)


def save_preprocessor(
    numeric_cols,
    medians,
    mean,
    std,
) -> None:
    """
    Save preprocessing parameters.
    """

    preprocessing = {
        "numeric_cols": numeric_cols,
        "medians": medians,
        "mean": mean,
        "std": std,
    }

    save_pickle(preprocessing, PREPROCESSOR_PATH)


def load_preprocessor():
    """
    Load preprocessing parameters.
    """
    return load_pickle(PREPROCESSOR_PATH)