"""Tests for the core modeling module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from optimum.grilly.configuration import GrillyConfig
from optimum.grilly.modeling import (
    GrillyModelForCausalLM,
    GrillyModelForFeatureExtraction,
    GrillyModelForSequenceClassification,
)

# ---------------------------------------------------------------------------
# Mock model helper
# ---------------------------------------------------------------------------


def _create_llama_mock(tmp_path: Path) -> Path:
    """Create a minimal LLaMA-like model directory for testing.

    Architecture: hidden=64, layers=2, heads=4, kv_heads=2,
                  vocab=100, intermediate=128, head_dim=16
    """
    model_dir = tmp_path / "mock_llama"
    model_dir.mkdir()

    hidden = 64
    num_layers = 2
    num_heads = 4
    num_kv_heads = 2
    intermediate = 128
    vocab = 100
    head_dim = hidden // num_heads  # 16

    # -- create and save GrillyConfig ----------------------------------------
    cfg = GrillyConfig.from_dict({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "max_position_embeddings": 512,
        "norm_eps": 1e-5,
        "norm_type": "rmsnorm",
        "activation": "silu",
        "has_bias": False,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    })
    cfg.save(model_dir)

    # Also write a plain config.json for HF-style loading
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": hidden}),
        encoding="utf-8",
    )

    # -- build weight tensors -----------------------------------------------
    rng = np.random.RandomState(42)
    weights: dict[str, np.ndarray] = {}

    # Embedding + LM head
    weights["model.embed_tokens.weight"] = rng.randn(vocab, hidden).astype(np.float32) * 0.02
    weights["lm_head.weight"] = rng.randn(vocab, hidden).astype(np.float32) * 0.02

    # Final norm
    weights["model.norm.weight"] = np.ones(hidden, dtype=np.float32)

    for i in range(num_layers):
        pfx = f"model.layers.{i}"

        # Attention projections
        weights[f"{pfx}.self_attn.q_proj.weight"] = (
            rng.randn(num_heads * head_dim, hidden).astype(np.float32) * 0.02
        )
        weights[f"{pfx}.self_attn.k_proj.weight"] = (
            rng.randn(num_kv_heads * head_dim, hidden).astype(np.float32) * 0.02
        )
        weights[f"{pfx}.self_attn.v_proj.weight"] = (
            rng.randn(num_kv_heads * head_dim, hidden).astype(np.float32) * 0.02
        )
        weights[f"{pfx}.self_attn.o_proj.weight"] = (
            rng.randn(hidden, hidden).astype(np.float32) * 0.02
        )

        # MLP (SwiGLU)
        weights[f"{pfx}.mlp.gate_proj.weight"] = (
            rng.randn(intermediate, hidden).astype(np.float32) * 0.02
        )
        weights[f"{pfx}.mlp.up_proj.weight"] = (
            rng.randn(intermediate, hidden).astype(np.float32) * 0.02
        )
        weights[f"{pfx}.mlp.down_proj.weight"] = (
            rng.randn(hidden, intermediate).astype(np.float32) * 0.02
        )

        # Norms
        weights[f"{pfx}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        weights[f"{pfx}.post_attention_layernorm.weight"] = np.ones(
            hidden, dtype=np.float32
        )

    # Save weights as safetensors
    save_file(weights, str(model_dir / "model.safetensors"))

    return model_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModeling:
    """Modeling module tests."""

    def test_load_and_forward_causal(self, tmp_path: Path):
        """Load mock LLaMA, run forward, check logits shape."""
        model_dir = _create_llama_mock(tmp_path)
        model = GrillyModelForCausalLM.from_pretrained(model_dir)

        input_ids = np.array([[1, 5, 10, 20, 30]], dtype=np.int64)
        out = model.forward(input_ids)

        logits = out.logits.numpy() if hasattr(out.logits, 'numpy') else out.logits
        # (batch=1, seq_len=5, vocab_size=100)
        assert logits.shape == (1, 5, 100), f"Expected (1, 5, 100), got {logits.shape}"
        assert np.all(np.isfinite(logits))

    def test_generate(self, tmp_path: Path):
        """Generate 5 new tokens and check output shape."""
        model_dir = _create_llama_mock(tmp_path)
        model = GrillyModelForCausalLM.from_pretrained(model_dir)

        input_ids = np.array([[1, 5, 10]], dtype=np.int64)
        generated = model.generate(input_ids, max_new_tokens=5, temperature=1.0, top_k=50)

        # Output should be (1, 3 + 5) = (1, 8)
        assert generated.shape == (1, 8), f"Expected (1, 8), got {generated.shape}"
        # All tokens should be in valid range [0, vocab_size)
        assert np.all(generated >= 0)
        assert np.all(generated < 100)

    def test_feature_extraction(self, tmp_path: Path):
        """Check last_hidden_state shape from feature extraction model."""
        model_dir = _create_llama_mock(tmp_path)
        model = GrillyModelForFeatureExtraction.from_pretrained(model_dir)

        input_ids = np.array([[1, 5, 10, 20]], dtype=np.int64)
        out = model.forward(input_ids)

        hs = out.last_hidden_state
        hs = hs.numpy() if hasattr(hs, 'numpy') else hs
        # (batch=1, seq_len=4, hidden_size=64)
        assert hs.shape == (1, 4, 64), f"Expected (1, 4, 64), got {hs.shape}"
        assert np.all(np.isfinite(hs))

    def test_save_and_reload(self, tmp_path: Path):
        """Save model, reload, check outputs match."""
        model_dir = _create_llama_mock(tmp_path)
        model = GrillyModelForCausalLM.from_pretrained(model_dir)

        input_ids = np.array([[1, 5, 10]], dtype=np.int64)
        out1 = model.forward(input_ids)

        save_dir = tmp_path / "saved_model"
        model.save_pretrained(save_dir)

        model2 = GrillyModelForCausalLM.from_pretrained(save_dir)
        out2 = model2.forward(input_ids)

        l1 = out1.logits.numpy() if hasattr(out1.logits, 'numpy') else out1.logits
        l2 = out2.logits.numpy() if hasattr(out2.logits, 'numpy') else out2.logits
        np.testing.assert_allclose(l1, l2, rtol=1e-5, atol=1e-6,
            err_msg="Outputs should match after save/reload")

    def test_sequence_classification(self, tmp_path: Path):
        """Check logits shape from sequence classification model."""
        model_dir = _create_llama_mock(tmp_path)

        weights_path = model_dir / "model.safetensors"
        from safetensors.numpy import load_file
        weights = load_file(str(weights_path))
        rng = np.random.RandomState(99)
        num_classes = 3
        weights["score.weight"] = rng.randn(num_classes, 64).astype(np.float32) * 0.02
        save_file(weights, str(weights_path))

        model = GrillyModelForSequenceClassification.from_pretrained(model_dir)
        input_ids = np.array([[1, 5, 10]], dtype=np.int64)
        out = model.forward(input_ids)

        logits = out.logits.numpy() if hasattr(out.logits, 'numpy') else out.logits
        assert logits.shape == (1, num_classes), f"Expected (1, 3), got {logits.shape}"
        assert np.all(np.isfinite(logits))

    def test_kv_cache_causal(self, tmp_path: Path):
        """Verify KV cache produces same logits as full recompute."""
        model_dir = _create_llama_mock(tmp_path)
        model = GrillyModelForCausalLM.from_pretrained(model_dir)

        input_ids = np.array([[1, 5, 10, 20, 30]], dtype=np.int64)

        # Full forward (no cache)
        out_full = model.forward(input_ids, return_dict=True)
        logits_full = out_full.logits.numpy() if hasattr(out_full.logits, 'numpy') else out_full.logits

        # Prefill first 3 tokens
        out_prefill = model.forward(input_ids[:, :3], return_dict=True)
        past_kv = out_prefill.past_key_values

        # Decode tokens 4 and 5 with cache
        out_decode = model.forward(
            input_ids[:, 3:], past_key_values=past_kv, return_dict=True)
        logits_decode = out_decode.logits.numpy() if hasattr(out_decode.logits, 'numpy') else out_decode.logits

        # Last 2 tokens' logits should match
        np.testing.assert_allclose(
            logits_full[:, 3:, :], logits_decode, rtol=1e-4, atol=1e-5,
            err_msg="KV cache logits should match full recompute")

    def test_hf_dataclass_returns(self, tmp_path: Path):
        """Verify HF-compliant return types."""
        from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast

        model_dir = _create_llama_mock(tmp_path)

        model_causal = GrillyModelForCausalLM.from_pretrained(model_dir)
        out = model_causal.forward(np.array([[1, 5, 10]], dtype=np.int64))
        assert isinstance(out, CausalLMOutputWithPast)
        assert out.logits is not None
        assert out.past_key_values is not None

        model_feat = GrillyModelForFeatureExtraction.from_pretrained(model_dir)
        out = model_feat.forward(np.array([[1, 5, 10]], dtype=np.int64))
        assert isinstance(out, BaseModelOutput)
        assert out.last_hidden_state is not None
