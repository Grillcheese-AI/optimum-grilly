"""
grilly.experimental.vsa - Vector Symbolic Architecture operations.

Core VSA operations that form the foundation for all experimental features.
Provides binary (bipolar), holographic (continuous), and block code vector operations.

Classes:
    BinaryOps: Operations for bipolar (+1/-1) vectors
    HolographicOps: Operations for continuous vectors using FFT-based binding
    BlockCodeOps: Operations for sparse block code vectors (IBM NVSA)
    ResonatorNetwork: Factorization of composite vectors
"""

from .block_ops import BlockCodeOps

__all__ = [
    "BlockCodeOps",
]
