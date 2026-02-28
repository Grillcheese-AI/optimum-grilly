# Optimum Grilly

<p align="center">
  <em>HuggingFace Optimum backend for Grilly — Vulkan GPU inference on any GPU</em>
</p>

[![PyPI](https://img.shields.io/pypi/v/optimum-grilly)](https://pypi.org/project/optimum-grilly/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/downloads/)

> **Alpha software.** APIs may change. We welcome early adopters and feedback.

**optimum-grilly** bridges [HuggingFace Transformers](https://huggingface.co/docs/transformers) to [Grilly](https://github.com/grillcheese-ai/grilly)'s Vulkan compute backend. Load any supported model with `from_pretrained`, run inference on AMD, NVIDIA, or Intel GPUs — no CUDA required.

## Features

- **Any GPU**: AMD, NVIDIA, Intel — anything with Vulkan drivers
- **HuggingFace compatible**: Same `from_pretrained` / `generate` API you already know
- **Zero PyTorch runtime**: Export once, run forever without PyTorch installed
- **Automatic CPU fallback**: Works without a GPU (slower, but functional)
- **Supported architectures**: LLaMA, Mistral, BERT, GPT-2 (T5 planned)

## Installation

```bash
# Core package (CPU fallback only)
pip install optimum-grilly

# With Vulkan GPU acceleration
pip install optimum-grilly[gpu]

# With export support (requires PyTorch)
pip install optimum-grilly[export]

# Everything
pip install optimum-grilly[all]
```

### Requirements

- Python >= 3.10
- [grilly](https://github.com/grillcheese-ai/grilly) >= 0.4.5 (for GPU acceleration)
- Vulkan drivers installed on your system
- For export: PyTorch >= 2.0

## Quick Start

### 1. Export a HuggingFace model

Convert a HuggingFace model to `.grilly` format (safetensors + config):

```python
from optimum.grilly import export_to_grilly

# Export a causal LM
export_to_grilly(
    "meta-llama/Llama-3.2-1B",
    output_dir="./llama-1b-grilly",
)

# Export a BERT model for feature extraction
export_to_grilly(
    "bert-base-uncased",
    output_dir="./bert-grilly",
    task="feature-extraction",
)
```

Or from the command line:

```bash
optimum-grilly-export --model meta-llama/Llama-3.2-1B --output ./llama-1b-grilly
optimum-grilly-export --model bert-base-uncased --output ./bert-grilly --task feature-extraction
```

### 2. Run inference

```python
from optimum.grilly import GrillyModelForCausalLM
from transformers import AutoTokenizer

# Load model and tokenizer
model = GrillyModelForCausalLM.from_pretrained("./llama-1b-grilly")
tokenizer = AutoTokenizer.from_pretrained("./llama-1b-grilly")

# Generate text
input_ids = tokenizer("The meaning of life is", return_tensors="np")["input_ids"]
output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### 3. Feature extraction (embeddings)

```python
from optimum.grilly import GrillyModelForFeatureExtraction
from optimum.grilly.pipelines import grilly_feature_extraction_pipeline
from transformers import AutoTokenizer

model = GrillyModelForFeatureExtraction.from_pretrained("./bert-grilly")
tokenizer = AutoTokenizer.from_pretrained("./bert-grilly")

# Get sentence embeddings
embedding = grilly_feature_extraction_pipeline(
    model, tokenizer, "Hello world", pooling="mean"
)
print(embedding.shape)  # (1, 768)
```

## API Reference

### Configuration

```python
from optimum.grilly import GrillyConfig

# From a HuggingFace config dict
config = GrillyConfig.from_hf_config(hf_config_dict)

# Save / load
config.save("./model-dir")
config = GrillyConfig.load("./model-dir")

# Inspect
print(config)  # GrillyConfig(model_type='llama', hidden_size=4096, ...)
print(config.get_layer_map())  # Layer descriptors for weight loading
```

### Models

| Class | Description |
|-------|-------------|
| `GrillyModel` | Base class — embed + transformer blocks + final norm |
| `GrillyModelForCausalLM` | + LM head + `generate()` for text generation |
| `GrillyModelForFeatureExtraction` | Returns `last_hidden_state` for embeddings |
| `GrillyModelForSequenceClassification` | + classifier head for classification tasks |

All models support:
- `from_pretrained(path)` — Load from a `.grilly` directory
- `save_pretrained(path)` — Save config + weights
- `forward(input_ids, attention_mask=None)` — Run inference

### Export

```python
from optimum.grilly import export_to_grilly

export_to_grilly(
    model_name_or_path="meta-llama/Llama-3.2-1B",
    output_dir="./output",
    task="causal-lm",         # "causal-lm", "feature-extraction",
                               # "sequence-classification", "auto"
    dtype="float32",
    include_tokenizer=True,
)
```

### Pipelines

```python
from optimum.grilly.pipelines import (
    grilly_text_generation_pipeline,
    grilly_feature_extraction_pipeline,
)

# Text generation
text = grilly_text_generation_pipeline(model, tokenizer, "Once upon a time")

# Feature extraction with pooling
embedding = grilly_feature_extraction_pipeline(
    model, tokenizer, "Hello", pooling="mean"  # "mean", "cls", "last"
)
```

## Architecture

```
optimum-grilly
├── optimum/grilly/
│   ├── __init__.py          # Lazy imports
│   ├── configuration.py     # GrillyConfig (HF config mapping)
│   ├── modeling.py           # GrillyModel + task subclasses
│   ├── export.py             # HF PyTorch → .grilly converter
│   ├── pipelines.py          # Pipeline helpers
│   ├── utils.py              # safetensors I/O
│   └── version.py
├── tests/
│   ├── test_configuration.py
│   ├── test_modeling.py
│   ├── test_export.py
│   ├── test_pipelines.py
│   └── test_utils.py
└── pyproject.toml
```

### How it works

1. **Export** (`export.py`): Downloads a HuggingFace PyTorch model, extracts all `named_parameters()` and `named_buffers()` as float32 numpy arrays, saves them as safetensors alongside a `grilly_config.json` that maps the HF architecture to grilly ops.

2. **Load** (`modeling.py`): Reads the safetensors weights and config, builds a graph of `_TransformerBlock` objects that hold numpy weight arrays. Each block dispatches linear/norm/attention/FFN operations to `grilly_core` (the C++ Vulkan extension) with automatic CPU numpy fallbacks.

3. **Inference**: All computation happens in float32. The Vulkan backend handles GPU upload/download transparently. When `grilly_core` is not available, all ops fall back to numpy — slower but correct.

### Supported architectures

| Architecture | Status | Notes |
|-------------|--------|-------|
| LLaMA / LLaMA 2 / LLaMA 3 | Supported | Pre-norm, SwiGLU, RoPE, GQA |
| Mistral | Supported | Same as LLaMA (sliding window not yet implemented) |
| BERT | Supported | Post-norm, standard FFN |
| GPT-2 | Supported | Pre-norm, fused QKV, Conv1D weight handling |
| T5 | Planned | Encoder-decoder not yet implemented |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VK_GPU_INDEX` | Select GPU by index (default: 0) |
| `GRILLY_DEBUG` | Set to `1` for debug logging |
| `ALLOW_CPU_VULKAN` | Set to `1` to allow llvmpipe CPU fallback |

## Known Limitations

- **No KV-cache**: `generate()` recomputes the full forward pass per token (O(n²)). KV-cache support is planned.
- **Float32 only**: No fp16/bf16/int8 quantization yet.
- **No beam search**: Only greedy and top-k sampling.
- **No streaming**: `generate()` returns the full sequence.
- **T5 not supported**: Encoder-decoder architectures are not yet implemented.

## Development

```bash
git clone https://github.com/grillcheese-ai/optimum-grilly.git
cd optimum-grilly
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Links

- [Grilly](https://github.com/grillcheese-ai/grilly) — The GPU framework
- [HuggingFace Optimum](https://huggingface.co/docs/optimum) — HF's optimization toolkit
- [GrillCheese AI](https://github.com/grillcheese-ai) — Research lab
