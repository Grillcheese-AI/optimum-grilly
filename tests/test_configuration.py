"""Tests for GrillyConfig configuration module."""

from __future__ import annotations

import json
from pathlib import Path

from optimum.grilly.configuration import GrillyConfig


class TestGrillyConfig:
    """GrillyConfig unit tests."""

    # -- test_from_dict: create config with explicit args --------------------

    def test_from_dict(self):
        cfg = GrillyConfig.from_dict({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "intermediate_size": 5504,
            "vocab_size": 32000,
            "max_position_embeddings": 2048,
            "norm_eps": 1e-5,
            "norm_type": "rmsnorm",
            "activation": "silu",
            "has_bias": False,
            "rope_theta": 10000.0,
        })
        assert cfg.model_type == "llama"
        assert cfg.hidden_size == 2048
        assert cfg.num_hidden_layers == 16
        assert cfg.num_attention_heads == 16
        assert cfg.num_key_value_heads == 8
        assert cfg.intermediate_size == 5504
        assert cfg.norm_type == "rmsnorm"
        assert cfg.activation == "silu"
        assert cfg.has_bias is False
        assert cfg.backend == "vulkan"

    # -- test_from_hf_config_llama: map a LLaMA HF config -------------------

    def test_from_hf_config_llama(self):
        hf = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "rope_theta": 500000.0,
            "tie_word_embeddings": False,
        }
        cfg = GrillyConfig.from_hf_config(hf)

        assert cfg.model_type == "llama"
        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 32
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.intermediate_size == 11008
        # rms_norm_eps remapped to norm_eps
        assert cfg.norm_eps == 1e-6
        # hidden_act remapped to activation
        assert cfg.activation == "silu"
        # arch defaults applied
        assert cfg.norm_type == "rmsnorm"
        assert cfg.has_bias is False
        assert cfg.rope_theta == 500000.0

    # -- test_from_hf_config_bert: map a BERT HF config ---------------------

    def test_from_hf_config_bert(self):
        hf = {
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 30522,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
        }
        cfg = GrillyConfig.from_hf_config(hf)

        assert cfg.model_type == "bert"
        assert cfg.hidden_size == 768
        assert cfg.num_hidden_layers == 12
        # layer_norm_eps remapped to norm_eps
        assert cfg.norm_eps == 1e-12
        # hidden_act remapped to activation
        assert cfg.activation == "gelu"
        # arch defaults applied
        assert cfg.norm_type == "layernorm"
        assert cfg.has_bias is True
        # MHA: kv_heads == num_heads
        assert cfg.effective_kv_heads == 12

    # -- test_save_and_load: JSON round-trip ---------------------------------

    def test_save_and_load(self, tmp_path: Path):
        original = GrillyConfig.from_dict({
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
            "max_position_embeddings": 32768,
            "norm_eps": 1e-5,
            "norm_type": "rmsnorm",
            "activation": "silu",
            "has_bias": False,
            "rope_theta": 10000.0,
        })

        saved_path = original.save(tmp_path)
        assert saved_path.exists()
        assert saved_path.name == "grilly_config.json"

        # Verify JSON structure has nested architecture dict
        raw = json.loads(saved_path.read_text(encoding="utf-8"))
        assert "model_type" in raw
        assert "architecture" in raw
        assert raw["architecture"]["hidden_size"] == 4096
        assert "layer_map" in raw

        # Round-trip
        loaded = GrillyConfig.load(tmp_path)
        assert loaded.model_type == original.model_type
        assert loaded.hidden_size == original.hidden_size
        assert loaded.num_hidden_layers == original.num_hidden_layers
        assert loaded.num_attention_heads == original.num_attention_heads
        assert loaded.num_key_value_heads == original.num_key_value_heads
        assert loaded.intermediate_size == original.intermediate_size
        assert loaded.vocab_size == original.vocab_size
        assert loaded.norm_eps == original.norm_eps
        assert loaded.norm_type == original.norm_type
        assert loaded.activation == original.activation
        assert loaded.has_bias == original.has_bias
        assert loaded.rope_theta == original.rope_theta
        assert loaded.backend == original.backend

    # -- test_layer_map_generation: verify layer_map structure ---------------

    def test_layer_map_generation(self):
        cfg = GrillyConfig.from_dict({
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 5504,
            "vocab_size": 32000,
        })
        layer_map = cfg.get_layer_map()

        # llama: embed + 4 transformer blocks + final_norm + lm_head = 7
        assert len(layer_map) == 7

        # First entry: embedding
        assert layer_map[0]["name"] == "embedding"
        assert layer_map[0]["grilly_op"] == "embedding"
        assert layer_map[0]["hf_pattern"] == "model.embed_tokens.weight"
        assert layer_map[0]["params"]["vocab_size"] == 32000
        assert layer_map[0]["params"]["hidden_size"] == 2048

        # Transformer blocks
        for i in range(4):
            entry = layer_map[1 + i]
            assert entry["name"] == f"layer.{i}"
            assert entry["grilly_op"] == "transformer_block"
            assert entry["hf_pattern"] == f"model.layers.{i}"
            assert entry["params"]["hidden_size"] == 2048
            assert entry["params"]["num_attention_heads"] == 16
            assert entry["params"]["num_key_value_heads"] == 4
            assert entry["params"]["intermediate_size"] == 5504
            assert entry["params"]["norm_type"] == "rmsnorm"

        # Final norm
        assert layer_map[5]["name"] == "final_norm"
        assert layer_map[5]["grilly_op"] == "rmsnorm"
        assert layer_map[5]["hf_pattern"] == "model.norm.weight"

        # lm_head
        assert layer_map[6]["name"] == "lm_head"
        assert layer_map[6]["grilly_op"] == "linear"
        assert layer_map[6]["hf_pattern"] == "lm_head.weight"

    # -- test_layer_map_bert: BERT has no final_norm / lm_head ---------------

    def test_layer_map_bert(self):
        cfg = GrillyConfig.from_hf_config({
            "model_type": "bert",
            "hidden_size": 768,
            "num_hidden_layers": 2,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 30522,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1e-12,
            "hidden_act": "gelu",
        })
        layer_map = cfg.get_layer_map()

        # bert: embed + 2 transformer blocks = 3 (no final_norm, no lm_head)
        assert len(layer_map) == 3
        assert layer_map[0]["name"] == "embedding"
        assert layer_map[0]["hf_pattern"] == "bert.embeddings"
        assert layer_map[1]["hf_pattern"] == "bert.encoder.layer.0"
        assert layer_map[2]["hf_pattern"] == "bert.encoder.layer.1"
