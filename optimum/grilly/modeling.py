"""
Core modeling module for optimum-grilly.

Provides GrillyModel and task-specific subclasses for Vulkan-accelerated
transformer inference. Each class wraps numpy weight arrays and dispatches
operations to the grilly C++ backend (Vulkan GPU) with automatic CPU fallbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .configuration import GrillyConfig
from .utils import load_weights, save_weights

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bridge detection -- grilly_core is the C++ pybind11 module
# ---------------------------------------------------------------------------

_bridge = None
_BRIDGE = False
_device = None

_SHADER_DIR = None


def _find_shader_dir() -> Optional[str]:
    """Locate the grilly shaders/spv directory."""
    # Try common locations
    candidates = [
        Path(__file__).resolve().parents[3] / "grilly" / "shaders" / "spv",
    ]
    # Try to find via grilly package
    try:
        import grilly
        pkg = Path(grilly.__file__).resolve().parent
        candidates.insert(0, pkg / "shaders" / "spv")
        candidates.insert(1, pkg.parent / "shaders" / "spv")
    except ImportError:
        pass
    for p in candidates:
        if p.is_dir() and any(p.glob("*.spv")):
            return str(p)
    return None


def _get_device():
    """Lazily initialise and return the grilly_core Device singleton."""
    global _device, _bridge, _BRIDGE, _SHADER_DIR
    if _device is not None:
        return _device
    if _bridge is None:
        return None
    try:
        _device = _bridge.Device()
        shader_dir = _SHADER_DIR or _find_shader_dir()
        if shader_dir:
            _device.load_shaders(shader_dir)
            _SHADER_DIR = shader_dir
        return _device
    except Exception:
        _BRIDGE = False
        return None


try:
    import grilly_core as _bridge  # C++ pybind11 Vulkan backend (optional)
    _BRIDGE = True
except ImportError:
    _bridge = None
    _BRIDGE = False


# ---------------------------------------------------------------------------
# CPU fallback helpers
# ---------------------------------------------------------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis (CPU fallback)."""
    x_max = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _rmsnorm_np(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)  (CPU fallback)."""
    variance = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps).astype(np.float32)
    return (x_normed * weight).astype(np.float32)


def _layernorm_np(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float
) -> np.ndarray:
    """LayerNorm (CPU fallback)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_normed = (x - mean) / np.sqrt(var + eps)
    return (gamma * x_normed + beta).astype(np.float32)


def _silu_np(x: np.ndarray) -> np.ndarray:
    return (x * (1.0 / (1.0 + np.exp(-x)))).astype(np.float32)


def _gelu_np(x: np.ndarray) -> np.ndarray:
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))).astype(
        np.float32
    )


def _relu_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.float32)


def _tanh_np(x: np.ndarray) -> np.ndarray:
    return np.tanh(x).astype(np.float32)


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACT_MAP_NP: Dict[str, Callable] = {
    "silu": _silu_np,
    "gelu": _gelu_np,
    "relu": _relu_np,
    "tanh": _tanh_np,
}

_ACT_MAP_BRIDGE: Dict[str, str] = {
    "silu": "silu",
    "gelu": "gelu",
    "relu": "relu",
    "tanh": "tanh_act",
}


def _get_act_fn(name: str) -> Callable:
    """Return GPU activation if bridge is available, else numpy fallback."""

    def _act(x: np.ndarray) -> np.ndarray:
        dev = _get_device()
        bridge_name = _ACT_MAP_BRIDGE.get(name)
        if dev is not None and bridge_name is not None:
            fn = getattr(_bridge, bridge_name, None)
            if fn is not None:
                try:
                    result = fn(dev, x)
                    if result is not None:
                        return result
                except Exception as e:
                    logger.debug("GPU op failed, falling back to CPU: %s", e)
        # CPU fallback
        return _ACT_MAP_NP.get(name, _silu_np)(x)

    return _act


# ---------------------------------------------------------------------------
# RoPE helper
# ---------------------------------------------------------------------------

def _build_rope_tables(
    seq_len: int, head_dim: int, theta: float = 10000.0
) -> tuple:
    """Build cosine and sine tables for rotary position embeddings."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, freqs)  # (seq_len, head_dim // 2)
    cos_table = np.cos(freqs).astype(np.float32)
    sin_table = np.sin(freqs).astype(np.float32)
    return cos_table, sin_table


def _apply_rope_np(
    x: np.ndarray, cos_table: np.ndarray, sin_table: np.ndarray
) -> np.ndarray:
    """Apply rotary embeddings (CPU fallback).

    x: (batch, heads, seq, head_dim)
    cos/sin tables: (seq, head_dim // 2)
    """
    head_dim = x.shape[-1]
    half = head_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Broadcast cos/sin: (1, 1, seq, half)
    c = cos_table[np.newaxis, np.newaxis, :, :]
    s = sin_table[np.newaxis, np.newaxis, :, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return np.concatenate([o1, o2], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# _TransformerBlock
# ---------------------------------------------------------------------------

class _TransformerBlock:
    """One transformer layer. Holds weight numpy arrays and dispatches ops."""

    def __init__(self, config: GrillyConfig, layer_idx: int, weights: dict):
        self.config = config
        self.layer_idx = layer_idx
        self.model_type = config.model_type
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.effective_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.norm_eps = config.norm_eps
        self.norm_type = config.norm_type
        self.act_fn = _get_act_fn(config.activation)
        self.rope_theta = config.rope_theta
        self.has_bias = config.has_bias

        # Weight placeholders
        self.q_weight = None
        self.k_weight = None
        self.v_weight = None
        self.o_weight = None
        self.q_bias = None
        self.k_bias = None
        self.v_bias = None
        self.o_bias = None

        self.gate_weight = None
        self.up_weight = None
        self.down_weight = None
        self.gate_bias = None
        self.up_bias = None
        self.down_bias = None

        self.input_norm_weight = None
        self.input_norm_bias = None
        self.post_attn_norm_weight = None
        self.post_attn_norm_bias = None

        self._load_weights(weights)

    # ---- weight loading ---------------------------------------------------

    def _load_weights(self, weights: dict):
        """Extract this layer's weights from the full weight dict."""
        i = self.layer_idx

        if self.model_type in ("llama", "mistral"):
            pfx = f"model.layers.{i}"
            self.q_weight = weights.get(f"{pfx}.self_attn.q_proj.weight")
            self.k_weight = weights.get(f"{pfx}.self_attn.k_proj.weight")
            self.v_weight = weights.get(f"{pfx}.self_attn.v_proj.weight")
            self.o_weight = weights.get(f"{pfx}.self_attn.o_proj.weight")

            self.gate_weight = weights.get(f"{pfx}.mlp.gate_proj.weight")
            self.up_weight = weights.get(f"{pfx}.mlp.up_proj.weight")
            self.down_weight = weights.get(f"{pfx}.mlp.down_proj.weight")

            self.input_norm_weight = weights.get(f"{pfx}.input_layernorm.weight")
            self.post_attn_norm_weight = weights.get(f"{pfx}.post_attention_layernorm.weight")

        elif self.model_type in ("bert", "xlm-roberta"):
            # Try both prefixed and unprefixed keys (export may strip "bert.")
            pfx = f"bert.encoder.layer.{i}"
            if f"{pfx}.attention.self.query.weight" not in weights:
                pfx = f"encoder.layer.{i}"
            self.q_weight = weights.get(f"{pfx}.attention.self.query.weight")
            self.k_weight = weights.get(f"{pfx}.attention.self.key.weight")
            self.v_weight = weights.get(f"{pfx}.attention.self.value.weight")
            self.o_weight = weights.get(f"{pfx}.attention.output.dense.weight")
            self.q_bias = weights.get(f"{pfx}.attention.self.query.bias")
            self.k_bias = weights.get(f"{pfx}.attention.self.key.bias")
            self.v_bias = weights.get(f"{pfx}.attention.self.value.bias")
            self.o_bias = weights.get(f"{pfx}.attention.output.dense.bias")

            self.up_weight = weights.get(f"{pfx}.intermediate.dense.weight")
            self.up_bias = weights.get(f"{pfx}.intermediate.dense.bias")
            self.down_weight = weights.get(f"{pfx}.output.dense.weight")
            self.down_bias = weights.get(f"{pfx}.output.dense.bias")

            self.input_norm_weight = weights.get(
                f"{pfx}.attention.output.LayerNorm.weight"
            )
            self.input_norm_bias = weights.get(
                f"{pfx}.attention.output.LayerNorm.bias"
            )
            self.post_attn_norm_weight = weights.get(
                f"{pfx}.output.LayerNorm.weight"
            )
            self.post_attn_norm_bias = weights.get(
                f"{pfx}.output.LayerNorm.bias"
            )

        elif self.model_type == "gpt2":
            pfx = f"h.{i}"
            # GPT-2 uses HF Conv1D which stores weights as (in, out) —
            # transpose to standard (out, in) convention for _linear().
            w = weights.get(f"{pfx}.attn.c_attn.weight")
            self.q_weight = w.T if w is not None else None
            self.q_bias = weights.get(f"{pfx}.attn.c_attn.bias")
            w = weights.get(f"{pfx}.attn.c_proj.weight")
            self.o_weight = w.T if w is not None else None
            self.o_bias = weights.get(f"{pfx}.attn.c_proj.bias")

            w = weights.get(f"{pfx}.mlp.c_fc.weight")
            self.up_weight = w.T if w is not None else None
            self.up_bias = weights.get(f"{pfx}.mlp.c_fc.bias")
            w = weights.get(f"{pfx}.mlp.c_proj.weight")
            self.down_weight = w.T if w is not None else None
            self.down_bias = weights.get(f"{pfx}.mlp.c_proj.bias")

            self.input_norm_weight = weights.get(f"{pfx}.ln_1.weight")
            self.input_norm_bias = weights.get(f"{pfx}.ln_1.bias")
            self.post_attn_norm_weight = weights.get(f"{pfx}.ln_2.weight")
            self.post_attn_norm_bias = weights.get(f"{pfx}.ln_2.bias")

        elif self.model_type == "t5":
            raise NotImplementedError(
                "T5 encoder-decoder inference is not yet supported. "
                "Supported architectures: llama, mistral, bert, gpt2."
            )

    # ---- norm helper ------------------------------------------------------

    def _norm(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply RMSNorm or LayerNorm via bridge, with CPU fallback."""
        dev = _get_device()
        if self.norm_type == "rmsnorm":
            if dev is not None:
                try:
                    result = _bridge.rmsnorm(dev, x, weight, self.norm_eps)
                    if result is not None:
                        return result
                except Exception as e:
                    logger.debug("GPU op failed, falling back to CPU: %s", e)
            return _rmsnorm_np(x, weight, self.norm_eps)
        else:
            # layernorm
            b = bias if bias is not None else np.zeros_like(weight)
            if dev is not None:
                try:
                    result = _bridge.layernorm(dev, x, weight, b, self.norm_eps)
                    if result is not None:
                        return result
                except Exception as e:
                    logger.debug("GPU op failed, falling back to CPU: %s", e)
            return _layernorm_np(x, weight, b, self.norm_eps)

    # ---- linear helper ----------------------------------------------------

    def _linear(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Matrix multiply via bridge, with CPU fallback."""
        dev = _get_device()
        if dev is not None:
            try:
                result = _bridge.linear(dev, x, weight, bias)
                if result is not None:
                    return result
            except Exception:
                pass
        # CPU fallback: weight is (out_features, in_features)
        out = x @ weight.T
        if bias is not None:
            out = out + bias
        return out.astype(np.float32)

    # ---- attention --------------------------------------------------------

    def _attention(
        self, hidden_states: np.ndarray, attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Multi-head (grouped query) attention with RoPE for LLaMA/Mistral."""
        batch, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        if self.model_type == "gpt2" and self.q_weight is not None:
            # GPT-2: fused QKV projection
            qkv = self._linear(hidden_states, self.q_weight, self.q_bias)
            q, k, v = np.split(qkv, 3, axis=-1)
        else:
            q = self._linear(hidden_states, self.q_weight, self.q_bias)
            k = self._linear(hidden_states, self.k_weight, self.k_bias)
            v = self._linear(hidden_states, self.v_weight, self.v_bias)

        # Reshape to multi-head: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE for LLaMA / Mistral
        if self.model_type in ("llama", "mistral"):
            cos_table, sin_table = _build_rope_tables(
                seq_len, self.head_dim, self.rope_theta
            )
            dev = _get_device()
            if dev is not None:
                try:
                    q_rope = _bridge.rope(dev, q, cos_table, sin_table, self.rope_theta, 1.0)
                    k_rope = _bridge.rope(dev, k, cos_table, sin_table, self.rope_theta, 1.0)
                    if q_rope is not None and k_rope is not None:
                        q, k = q_rope, k_rope
                    else:
                        q = _apply_rope_np(q, cos_table, sin_table)
                        k = _apply_rope_np(k, cos_table, sin_table)
                except Exception:
                    q = _apply_rope_np(q, cos_table, sin_table)
                    k = _apply_rope_np(k, cos_table, sin_table)
            else:
                q = _apply_rope_np(q, cos_table, sin_table)
                k = _apply_rope_np(k, cos_table, sin_table)

        # GQA: repeat KV heads if necessary
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = np.repeat(k, n_rep, axis=1)
            v = np.repeat(v, n_rep, axis=1)

        # Ensure contiguous arrays for bridge ops
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)

        scale = float(1.0 / np.sqrt(float(self.head_dim)))

        # Determine if causal masking is needed
        is_causal = self.model_type in ("llama", "mistral", "gpt2")

        # Try flash_attention2 first
        dev = _get_device()
        attn_out = None
        if dev is not None:
            try:
                flash_mask = None
                if attention_mask is not None and not is_causal:
                    flash_mask = attention_mask
                result = _bridge.flash_attention2(
                    dev, q, k, v, flash_mask, scale, 64, 64
                )
                if result is not None:
                    attn_out = result
            except Exception:
                pass

        # Manual attention fallback
        if attn_out is None:
            if dev is not None:
                try:
                    scores = _bridge.attention_scores(dev, q, k, scale)
                    if scores is not None:
                        if is_causal or attention_mask is not None:
                            mask_arg = attention_mask.astype(np.float32) if attention_mask is not None else None
                            scores = _bridge.attention_mask(
                                dev, scores, mask_arg, is_causal, -1e9
                            )
                        weights = _bridge.softmax(dev, scores, -1)
                        if weights is not None:
                            attn_out = _bridge.attention_output(dev, weights, v)
                except Exception as e:
                    logger.debug("GPU op failed, falling back to CPU: %s", e)

        # Pure numpy fallback
        if attn_out is None:
            scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
            if is_causal:
                causal_mask = np.triu(
                    np.full((seq_len, seq_len), -1e9, dtype=np.float32), k=1
                )
                scores = scores + causal_mask[np.newaxis, np.newaxis, :, :]
            if attention_mask is not None:
                scores = scores + attention_mask.astype(np.float32)
            weights = _softmax_np(scores)
            attn_out = np.einsum("bhqk,bhkd->bhqd", weights, v)

        # Concat heads
        attn_out = np.ascontiguousarray(attn_out, dtype=np.float32)
        if dev is not None:
            try:
                concat = _bridge.attention_concat_heads(dev, attn_out)
                if concat is not None:
                    attn_out = concat
                else:
                    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
                        batch, seq_len, self.hidden_size
                    )
            except Exception:
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
                    batch, seq_len, self.hidden_size
                )
        else:
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(
                batch, seq_len, self.hidden_size
            )

        # Output projection
        attn_out = self._linear(
            attn_out.astype(np.float32), self.o_weight, self.o_bias
        )
        return attn_out.astype(np.float32)

    # ---- feed-forward network ---------------------------------------------

    def _ffn(self, hidden_states: np.ndarray) -> np.ndarray:
        """Feed-forward: SwiGLU for llama/mistral, standard for bert/gpt2."""
        if self.model_type in ("llama", "mistral"):
            # SwiGLU: gate_proj * act(up_proj) then down_proj
            gate = self._linear(hidden_states, self.gate_weight, self.gate_bias)
            up = self._linear(hidden_states, self.up_weight, self.up_bias)
            hidden = self.act_fn(gate) * up
            return self._linear(hidden.astype(np.float32), self.down_weight, self.down_bias)
        else:
            # Standard: up -> act -> down
            hidden = self._linear(hidden_states, self.up_weight, self.up_bias)
            hidden = self.act_fn(hidden)
            return self._linear(hidden.astype(np.float32), self.down_weight, self.down_bias)

    # ---- forward ----------------------------------------------------------

    def forward(
        self, hidden_states: np.ndarray, attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Full transformer block forward pass.

        Pre-norm for llama/mistral:  norm -> attn -> residual -> norm -> ffn -> residual
        Post-norm for bert:          attn -> residual -> norm -> ffn -> residual -> norm
        """
        if self.model_type in ("llama", "mistral"):
            # Pre-norm architecture
            residual = hidden_states
            hidden_states = self._norm(hidden_states, self.input_norm_weight)
            hidden_states = self._attention(hidden_states, attention_mask)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self._norm(hidden_states, self.post_attn_norm_weight)
            hidden_states = self._ffn(hidden_states)
            hidden_states = residual + hidden_states

        elif self.model_type in ("bert", "xlm-roberta"):
            # Post-norm architecture
            residual = hidden_states
            hidden_states = self._attention(hidden_states, attention_mask)
            hidden_states = residual + hidden_states
            hidden_states = self._norm(
                hidden_states, self.input_norm_weight, self.input_norm_bias
            )

            residual = hidden_states
            hidden_states = self._ffn(hidden_states)
            hidden_states = residual + hidden_states
            hidden_states = self._norm(
                hidden_states, self.post_attn_norm_weight, self.post_attn_norm_bias
            )

        elif self.model_type == "gpt2":
            # Pre-norm (GPT-2 style)
            residual = hidden_states
            hidden_states = self._norm(
                hidden_states, self.input_norm_weight, self.input_norm_bias
            )
            hidden_states = self._attention(hidden_states, attention_mask)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self._norm(
                hidden_states, self.post_attn_norm_weight, self.post_attn_norm_bias
            )
            hidden_states = self._ffn(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states.astype(np.float32)


# ---------------------------------------------------------------------------
# GrillyModel -- base class
# ---------------------------------------------------------------------------

class GrillyModel:
    """Base class for Vulkan-accelerated inference models."""

    def __init__(self, config: GrillyConfig, weights: dict):
        self.config = config
        self._weights = weights
        self._embed_tokens: Optional[np.ndarray] = None
        self._position_embeddings: Optional[np.ndarray] = None
        self._layers: list = []
        self._final_norm_weight: Optional[np.ndarray] = None
        self._final_norm_bias: Optional[np.ndarray] = None
        self._build(config, weights)

    @classmethod
    def from_pretrained(cls, model_path: str | Path) -> "GrillyModel":
        """Load from a local .grilly directory."""
        path = Path(model_path)
        config = GrillyConfig.load(path)
        weights = load_weights(path)
        return cls(config, weights)

    def save_pretrained(self, directory: str | Path):
        """Save config and weights to a directory."""
        save_dir = Path(directory)
        self.config.save(save_dir)
        save_weights(self._weights, save_dir / "model.safetensors")

    def _build(self, config: GrillyConfig, weights: dict):
        """Set up embedding, layers, final norm based on config.model_type."""
        mt = config.model_type

        # Embedding weights
        if mt in ("llama", "mistral"):
            self._embed_tokens = weights.get("model.embed_tokens.weight")
        elif mt in ("bert", "xlm-roberta"):
            self._embed_tokens = (
                weights.get("bert.embeddings.word_embeddings.weight")
                or weights.get("embeddings.word_embeddings.weight")
            )
            self._position_embeddings = (
                weights.get("bert.embeddings.position_embeddings.weight")
                or weights.get("embeddings.position_embeddings.weight")
            )
        elif mt == "gpt2":
            self._embed_tokens = weights.get("wte.weight")
            self._position_embeddings = weights.get("wpe.weight")
        elif mt == "t5":
            self._embed_tokens = weights.get("shared.weight")

        # Transformer layers
        self._layers = [
            _TransformerBlock(config, i, weights)
            for i in range(config.num_hidden_layers)
        ]

        # Final norm
        if mt in ("llama", "mistral"):
            self._final_norm_weight = weights.get("model.norm.weight")
        elif mt == "gpt2":
            self._final_norm_weight = weights.get("ln_f.weight")
            self._final_norm_bias = weights.get("ln_f.bias")
        elif mt == "t5":
            self._final_norm_weight = weights.get("encoder.final_layer_norm.weight")

    def _embed(self, input_ids: np.ndarray) -> np.ndarray:
        """Embed token IDs via bridge or CPU indexing."""
        dev = _get_device()
        ids_u32 = np.asarray(input_ids, dtype=np.uint32)
        if dev is not None and self._embed_tokens is not None:
            try:
                result = _bridge.embedding_lookup(dev, ids_u32, self._embed_tokens)
                if result is not None:
                    hidden = result
                else:
                    hidden = self._embed_tokens[input_ids]
            except Exception:
                hidden = self._embed_tokens[input_ids]
        else:
            hidden = self._embed_tokens[input_ids]

        hidden = hidden.astype(np.float32)

        # Add position embeddings for BERT / GPT-2
        if self._position_embeddings is not None:
            seq_len = input_ids.shape[-1]
            positions = np.arange(seq_len, dtype=np.int64)
            pos_emb = self._position_embeddings[positions]
            hidden = hidden + pos_emb.astype(np.float32)

        return hidden

    def _final_norm(self, hidden_states: np.ndarray) -> np.ndarray:
        """Apply final norm (rmsnorm or layernorm)."""
        if self._final_norm_weight is None:
            return hidden_states

        dev = _get_device()
        if self.config.norm_type == "rmsnorm":
            if dev is not None:
                try:
                    result = _bridge.rmsnorm(
                        dev, hidden_states, self._final_norm_weight, self.config.norm_eps
                    )
                    if result is not None:
                        return result
                except Exception as e:
                    logger.debug("GPU op failed, falling back to CPU: %s", e)
            return _rmsnorm_np(hidden_states, self._final_norm_weight, self.config.norm_eps)
        else:
            bias = (
                self._final_norm_bias
                if self._final_norm_bias is not None
                else np.zeros_like(self._final_norm_weight)
            )
            if dev is not None:
                try:
                    result = _bridge.layernorm(
                        dev,
                        hidden_states,
                        self._final_norm_weight,
                        bias,
                        self.config.norm_eps,
                    )
                    if result is not None:
                        return result
                except Exception as e:
                    logger.debug("GPU op failed, falling back to CPU: %s", e)
            return _layernorm_np(
                hidden_states, self._final_norm_weight, bias, self.config.norm_eps
            )

    def _encode(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Embed -> transformer layers -> final norm."""
        hidden_states = self._embed(input_ids)
        for layer in self._layers:
            hidden_states = layer.forward(hidden_states, attention_mask)
        hidden_states = self._final_norm(hidden_states)
        return hidden_states.astype(np.float32)


# ---------------------------------------------------------------------------
# Task-specific subclasses
# ---------------------------------------------------------------------------

class GrillyModelForFeatureExtraction(GrillyModel):
    """Returns last hidden state (encoder output)."""

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> dict:
        hidden_states = self._encode(input_ids, attention_mask)
        return {"last_hidden_state": hidden_states}


class GrillyModelForSequenceClassification(GrillyModel):
    """Sequence classification with a linear head on top."""

    def __init__(self, config: GrillyConfig, weights: dict):
        super().__init__(config, weights)
        # Load classifier head
        self._classifier_weight: Optional[np.ndarray] = None
        self._classifier_bias: Optional[np.ndarray] = None
        for key_prefix in ("classifier", "score"):
            w = weights.get(f"{key_prefix}.weight")
            if w is not None:
                self._classifier_weight = w
                self._classifier_bias = weights.get(f"{key_prefix}.bias")
                break

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> dict:
        hidden_states = self._encode(input_ids, attention_mask)

        # Pool: [CLS] for BERT, last token for causal models
        if self.config.model_type == "bert":
            pooled = hidden_states[:, 0, :]  # [CLS] token
        else:
            # Last token for causal LMs
            pooled = hidden_states[:, -1, :]

        if self._classifier_weight is not None:
            dev = _get_device()
            if dev is not None:
                try:
                    logits = _bridge.linear(
                        dev, pooled, self._classifier_weight, self._classifier_bias
                    )
                    if logits is not None:
                        return {"logits": logits.astype(np.float32)}
                except Exception as e:
                    logger.debug("GPU op failed, falling back to CPU: %s", e)
            logits = pooled @ self._classifier_weight.T
            if self._classifier_bias is not None:
                logits = logits + self._classifier_bias
            return {"logits": logits.astype(np.float32)}

        return {"logits": pooled.astype(np.float32)}


class GrillyModelForCausalLM(GrillyModel):
    """Causal language model with LM head for text generation."""

    def __init__(self, config: GrillyConfig, weights: dict):
        super().__init__(config, weights)
        # LM head -- may be tied to embedding weights
        self._lm_head_weight: Optional[np.ndarray] = None
        self._lm_head_bias: Optional[np.ndarray] = None

        w = weights.get("lm_head.weight")
        if w is not None:
            self._lm_head_weight = w
        elif config.tie_word_embeddings and self._embed_tokens is not None:
            self._lm_head_weight = self._embed_tokens
        self._lm_head_bias = weights.get("lm_head.bias")

    def _lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """Project hidden states to vocabulary logits."""
        dev = _get_device()
        if dev is not None and self._lm_head_weight is not None:
            try:
                result = _bridge.linear(
                    dev, hidden_states, self._lm_head_weight, self._lm_head_bias
                )
                if result is not None:
                    return result.astype(np.float32)
            except Exception:
                pass
        logits = hidden_states @ self._lm_head_weight.T
        if self._lm_head_bias is not None:
            logits = logits + self._lm_head_bias
        return logits.astype(np.float32)

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> dict:
        hidden_states = self._encode(input_ids, attention_mask)
        logits = self._lm_head(hidden_states)
        return {"logits": logits}

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        """Simple autoregressive generation with top-k sampling.

        Note: No KV-cache — recomputes the full forward pass for each new
        token. This is correct but O(n²) in sequence length. KV-cache
        support is planned for a future release.
        """
        ids = np.array(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]

        for _ in range(max_new_tokens):
            out = self.forward(ids)
            logits = out["logits"][:, -1, :]  # (batch, vocab)

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0 and top_k < logits.shape[-1]:
                # For each batch element, zero out all but top-k
                indices_to_remove = np.argsort(logits, axis=-1)[:, :-top_k]
                for b in range(logits.shape[0]):
                    logits[b, indices_to_remove[b]] = -1e9

            # Sample from distribution
            probs = _softmax_np(logits)
            next_token = np.array(
                [
                    [np.random.choice(probs.shape[-1], p=probs[b])]
                    for b in range(probs.shape[0])
                ],
                dtype=np.int64,
            )
            ids = np.concatenate([ids, next_token], axis=-1)

        return ids
