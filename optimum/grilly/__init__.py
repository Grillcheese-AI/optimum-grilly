"""
optimum-grilly: HuggingFace Optimum backend for Grilly Vulkan inference.

Enables running transformer models on any GPU (AMD, NVIDIA, Intel) via
Vulkan compute shaders — no CUDA dependency.
"""

from .version import __version__


def __getattr__(name):
    if name == "GrillyModel":
        from .modeling import GrillyModel
        return GrillyModel
    if name == "GrillyModelForCausalLM":
        from .modeling import GrillyModelForCausalLM
        return GrillyModelForCausalLM
    if name == "GrillyModelForFeatureExtraction":
        from .modeling import GrillyModelForFeatureExtraction
        return GrillyModelForFeatureExtraction
    if name == "GrillyModelForSequenceClassification":
        from .modeling import GrillyModelForSequenceClassification
        return GrillyModelForSequenceClassification
    if name == "GrillyConfig":
        from .configuration import GrillyConfig
        return GrillyConfig
    raise AttributeError(f"module 'optimum.grilly' has no attribute {name!r}")


__all__ = [
    "__version__",
    "GrillyModel",
    "GrillyModelForCausalLM",
    "GrillyModelForFeatureExtraction",
    "GrillyModelForSequenceClassification",
    "GrillyConfig",
]
