"""Weight loading and safetensors I/O utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
from safetensors.numpy import load_file, save_file


def load_weights(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load weights from safetensors file(s).

    Args:
        path: Path to a single .safetensors file or a directory containing
              sharded safetensors files.
    Returns:
        Dictionary mapping weight names to numpy arrays (float32).
    """
    path = Path(path)
    if path.is_file():
        return load_file(str(path))
    if path.is_dir():
        weights: Dict[str, np.ndarray] = {}
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors files found in {path}")
        for shard in shards:
            weights.update(load_file(str(shard)))
        return weights
    raise FileNotFoundError(f"Path not found: {path}")


def save_weights(weights: Dict[str, np.ndarray], path: Union[str, Path]) -> None:
    """Save weights to a safetensors file.

    Args:
        weights: Dictionary mapping weight names to numpy arrays.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean: Dict[str, np.ndarray] = {}
    for k, v in weights.items():
        arr = np.asarray(v, dtype=np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        clean[k] = arr
    save_file(clean, str(path))
