"""Export HuggingFace PyTorch models to .grilly format.

This is the only module in optimum-grilly that requires PyTorch.
It converts HF model weights to safetensors (numpy) and generates
the grilly_config.json needed by the Vulkan inference backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .configuration import GrillyConfig

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


_TASK_MODEL_MAP = {
    "causal-lm": "AutoModelForCausalLM",
    "feature-extraction": "AutoModel",
    "sequence-classification": "AutoModelForSequenceClassification",
    "auto": "AutoModel",
}


def export_to_grilly(
    model_name_or_path: str,
    output_dir: str | Path,
    task: str = "causal-lm",
    dtype: str = "float32",
    include_tokenizer: bool = True,
) -> Path:
    """Convert a HuggingFace model to .grilly format.

    Downloads (or loads from local path) a HuggingFace PyTorch model,
    extracts all parameters and buffers as numpy float32 arrays, and
    writes them to a safetensors file alongside the grilly config.

    Args:
        model_name_or_path: HuggingFace model ID (e.g. "meta-llama/Llama-3.2-1B")
            or a local directory containing a pretrained model.
        output_dir: Directory to write the exported .grilly model files.
        task: Model task type. One of "causal-lm", "feature-extraction",
            "sequence-classification", or "auto" (uses AutoModel).
        dtype: Weight data type. Currently only "float32" is supported.
        include_tokenizer: Whether to copy tokenizer files into the output
            directory so they are bundled with the model.

    Returns:
        Path to the output directory containing the exported model.

    Raises:
        RuntimeError: If PyTorch is not installed.
        ValueError: If task type is not recognized.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for model export. "
            "Install with: pip install optimum-grilly[export]"
        )

    import transformers
    from safetensors.numpy import save_file
    from transformers import AutoConfig, AutoTokenizer

    # Resolve model class from task
    auto_cls_name = _TASK_MODEL_MAP.get(task)
    if auto_cls_name is None:
        raise ValueError(
            f"Unknown task {task!r}. Choose from: {', '.join(_TASK_MODEL_MAP)}"
        )
    auto_cls = getattr(transformers, auto_cls_name)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load and save the HuggingFace config ---
    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    hf_config_dict = hf_config.to_dict()

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config_dict, f, indent=2, default=str)

    # --- 2. Build and save the grilly config ---
    grilly_config = GrillyConfig.from_hf_config(hf_config_dict)
    grilly_config.save(output_dir)

    # --- 3. Load PyTorch model and extract weights ---
    model = auto_cls.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
    )

    weights: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        weights[name] = param.detach().cpu().numpy().astype(np.float32)
    for name, buf in model.named_buffers():
        weights[name] = buf.detach().cpu().numpy().astype(np.float32)

    # --- 4. Save weights as safetensors (numpy format) ---
    save_file(weights, str(output_dir / "model.safetensors"))

    # --- 5. Optionally bundle the tokenizer ---
    if include_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            tokenizer.save_pretrained(str(output_dir))
        except Exception:
            # Tokenizer is best-effort; some models may not have one
            pass

    # --- 6. Clean up any PyTorch-specific artifacts ---
    for pattern in ["*.bin", "*.pt", "pytorch_*"]:
        for f in output_dir.glob(pattern):
            f.unlink()

    return output_dir


def _cli_main():
    """CLI entry point for ``python -m optimum.grilly.export``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a HuggingFace model to .grilly format",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or path to local model directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the exported .grilly model",
    )
    parser.add_argument(
        "--task",
        default="causal-lm",
        choices=list(_TASK_MODEL_MAP),
        help="Model task type (default: causal-lm)",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32"],
        help="Weight data type (default: float32)",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip bundling the tokenizer",
    )

    args = parser.parse_args()
    result_dir = export_to_grilly(
        args.model,
        args.output,
        task=args.task,
        dtype=args.dtype,
        include_tokenizer=not args.no_tokenizer,
    )
    print(f"Exported to {result_dir}")


if __name__ == "__main__":
    _cli_main()
