"""
GrillyConfig: maps HuggingFace model configs to grilly layer configurations.

Standalone module -- does not import grilly or optimum at module level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_CONFIG_FILENAME = "grilly_config.json"

# ---- architecture defaults ------------------------------------------------

_ARCH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "llama": {
        "norm_type": "rmsnorm",
        "activation": "silu",
        "has_bias": False,
    },
    "mistral": {
        "norm_type": "rmsnorm",
        "activation": "silu",
        "has_bias": False,
    },
    "bert": {
        "norm_type": "layernorm",
        "activation": "gelu",
        "has_bias": True,
    },
    "gpt2": {
        "norm_type": "layernorm",
        "activation": "gelu",
        "has_bias": True,
    },
    "t5": {
        "norm_type": "rmsnorm",
        "activation": "relu",
        "has_bias": False,
    },
}

# ---- HF key remapping -----------------------------------------------------

_HF_KEY_MAP: Dict[str, str] = {
    "rms_norm_eps": "norm_eps",
    "layer_norm_eps": "norm_eps",
    "layer_norm_epsilon": "norm_eps",
    "hidden_act": "activation",
    "n_embd": "hidden_size",
    "n_layer": "num_hidden_layers",
    "n_head": "num_attention_heads",
    "n_inner": "intermediate_size",
    "n_positions": "max_position_embeddings",
}

# ---- weight-name patterns per architecture ---------------------------------

_WEIGHT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "llama": {
        "embed": "model.embed_tokens.weight",
        "layer_prefix": "model.layers.{i}",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    },
    "mistral": {
        "embed": "model.embed_tokens.weight",
        "layer_prefix": "model.layers.{i}",
        "final_norm": "model.norm.weight",
        "lm_head": "lm_head.weight",
    },
    "bert": {
        "embed": "bert.embeddings",
        "layer_prefix": "bert.encoder.layer.{i}",
        "final_norm": None,
        "lm_head": None,
    },
    "gpt2": {
        "embed": "wte.weight",
        "layer_prefix": "h.{i}",
        "final_norm": "ln_f",
        "lm_head": None,  # tied to embed
    },
    "t5": {
        "embed": "shared.weight",
        "layer_prefix": "encoder.block.{i}",
        "final_norm": "encoder.final_layer_norm.weight",
        "lm_head": "lm_head.weight",
    },
}


@dataclass
class GrillyConfig:
    """Configuration that bridges a HuggingFace model config to grilly layers."""

    # -- core architecture params --
    model_type: str = "llama"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None  # None => MHA (== num_attention_heads)
    intermediate_size: int = 11008
    vocab_size: int = 32000
    max_position_embeddings: int = 4096

    # -- normalization & activation --
    norm_eps: float = 1e-5
    norm_type: str = "rmsnorm"
    activation: str = "silu"

    # -- misc architecture flags --
    has_bias: bool = False
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False

    # -- backend metadata --
    backend: str = "vulkan"
    optimum_grilly_version: str = "0.1.0"

    # -- extra / passthrough fields --
    extra: Dict[str, Any] = field(default_factory=dict)

    # ---- constructors ------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GrillyConfig":
        """Create a GrillyConfig from an explicit dictionary of parameters."""
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for k, v in d.items():
            if k in known_keys:
                kwargs[k] = v
            else:
                extra[k] = v
        if extra:
            kwargs.setdefault("extra", {}).update(extra)
        return cls(**kwargs)

    @classmethod
    def from_hf_config(cls, hf_dict: Dict[str, Any]) -> "GrillyConfig":
        """Map a HuggingFace model config dict to a GrillyConfig.

        Applies HF key remapping, then fills architecture-specific defaults
        for any fields not explicitly present.
        """
        # 1. remap HF keys to grilly keys
        remapped: Dict[str, Any] = {}
        for k, v in hf_dict.items():
            target = _HF_KEY_MAP.get(k, k)
            remapped[target] = v

        # 2. determine model_type
        model_type = remapped.get("model_type", "llama")
        arch_defaults = _ARCH_DEFAULTS.get(model_type, {})

        # 3. apply arch defaults for missing fields
        for key, default_val in arch_defaults.items():
            remapped.setdefault(key, default_val)

        # 4. num_key_value_heads defaults to num_attention_heads (MHA)
        if "num_key_value_heads" not in remapped and "num_attention_heads" in remapped:
            remapped["num_key_value_heads"] = remapped["num_attention_heads"]

        return cls.from_dict(remapped)

    # ---- serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dict (JSON-friendly)."""
        d: Dict[str, Any] = {
            "model_type": self.model_type,
            "architecture": {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.effective_kv_heads,
                "intermediate_size": self.intermediate_size,
                "vocab_size": self.vocab_size,
                "max_position_embeddings": self.max_position_embeddings,
                "norm_eps": self.norm_eps,
                "norm_type": self.norm_type,
                "activation": self.activation,
                "has_bias": self.has_bias,
                "rope_theta": self.rope_theta,
                "tie_word_embeddings": self.tie_word_embeddings,
            },
            "backend": self.backend,
            "optimum_grilly_version": self.optimum_grilly_version,
            "layer_map": self.get_layer_map(),
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    def save(self, directory: str | Path) -> Path:
        """Write ``grilly_config.json`` into *directory* and return the path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / _CONFIG_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "GrillyConfig":
        """Read ``grilly_config.json`` from *directory*."""
        path = Path(directory) / _CONFIG_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        # Flatten the nested "architecture" dict back to top-level keys
        flat: Dict[str, Any] = {}
        flat["model_type"] = data.get("model_type", "llama")
        flat["backend"] = data.get("backend", "vulkan")
        flat["optimum_grilly_version"] = data.get("optimum_grilly_version", "0.1.0")
        arch = data.get("architecture", {})
        flat.update(arch)
        if "extra" in data:
            flat["extra"] = data["extra"]
        return cls.from_dict(flat)

    # ---- layer map ---------------------------------------------------------

    @property
    def effective_kv_heads(self) -> int:
        """Return num_key_value_heads, defaulting to num_attention_heads (MHA)."""
        return self.num_key_value_heads if self.num_key_value_heads is not None else self.num_attention_heads

    def get_layer_map(self) -> List[Dict[str, Any]]:
        """Generate a list of layer descriptors for weight loading.

        Each entry describes one logical layer with its HF weight name
        pattern and the grilly op it maps to.
        """
        patterns = _WEIGHT_PATTERNS.get(self.model_type, _WEIGHT_PATTERNS["llama"])
        layers: List[Dict[str, Any]] = []

        # embedding
        layers.append({
            "name": "embedding",
            "hf_pattern": patterns["embed"],
            "grilly_op": "embedding",
            "params": {"vocab_size": self.vocab_size, "hidden_size": self.hidden_size},
        })

        # transformer blocks
        for i in range(self.num_hidden_layers):
            prefix = patterns["layer_prefix"].format(i=i)
            layers.append({
                "name": f"layer.{i}",
                "hf_pattern": prefix,
                "grilly_op": "transformer_block",
                "params": {
                    "hidden_size": self.hidden_size,
                    "num_attention_heads": self.num_attention_heads,
                    "num_key_value_heads": self.effective_kv_heads,
                    "intermediate_size": self.intermediate_size,
                    "norm_type": self.norm_type,
                    "norm_eps": self.norm_eps,
                    "activation": self.activation,
                    "has_bias": self.has_bias,
                },
            })

        # final norm
        if patterns.get("final_norm") is not None:
            layers.append({
                "name": "final_norm",
                "hf_pattern": patterns["final_norm"],
                "grilly_op": self.norm_type,
                "params": {"hidden_size": self.hidden_size, "eps": self.norm_eps},
            })

        # lm_head
        if patterns.get("lm_head") is not None:
            layers.append({
                "name": "lm_head",
                "hf_pattern": patterns["lm_head"],
                "grilly_op": "linear",
                "params": {
                    "in_features": self.hidden_size,
                    "out_features": self.vocab_size,
                    "bias": self.has_bias,
                },
            })

        return layers

    # ---- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GrillyConfig(model_type={self.model_type!r}, "
            f"hidden_size={self.hidden_size}, "
            f"layers={self.num_hidden_layers}, "
            f"heads={self.num_attention_heads}, "
            f"kv_heads={self.effective_kv_heads})"
        )
