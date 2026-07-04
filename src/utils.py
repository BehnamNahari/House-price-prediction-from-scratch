import json
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    OUTPUTS_DIR,
    MODELS_DIR,
    FIGURES_DIR,
)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def create_directories() -> None:
    """
    Create project output directories if they do not exist.
    """

    directories = [
        OUTPUTS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def save_pickle(obj: Any, path: Path) -> None:
    """
    Save a Python object using pickle.
    """

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """
    Load a Python object from a pickle file.
    """

    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data: dict, path: Path) -> None:
    """
    Save dictionary as JSON.
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: Path) -> dict:
    """
    Load dictionary from JSON.
    """

    with open(path, "r") as f:
        return json.load(f)


def timer() -> float:
    """
    Return current time.
    """

    return time.perf_counter()