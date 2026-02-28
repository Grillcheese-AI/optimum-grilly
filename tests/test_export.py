"""Tests for the export module.

Only runs when PyTorch is available (guarded by pytest.importorskip).
"""

from __future__ import annotations

import pytest

from optimum.grilly.configuration import GrillyConfig

torch = pytest.importorskip("torch")


class TestExportHelpers:
    """Verify export module structure and GrillyConfig integration."""

    def test_grilly_config_from_hf_roundtrip(self, tmp_path):
        """GrillyConfig.from_hf_config preserves key fields and round-trips."""
        hf_dict = {
            "model_type": "llama",
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "vocab_size": 1000,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "hidden_act": "silu",
        }
        cfg = GrillyConfig.from_hf_config(hf_dict)

        assert cfg.model_type == "llama"
        assert cfg.hidden_size == 256
        assert cfg.num_hidden_layers == 4
        assert cfg.num_attention_heads == 4
        assert cfg.num_key_value_heads == 2
        assert cfg.intermediate_size == 512
        assert cfg.vocab_size == 1000
        assert cfg.norm_eps == 1e-6
        assert cfg.norm_type == "rmsnorm"
        assert cfg.activation == "silu"
        assert cfg.rope_theta == 500000.0
        assert cfg.has_bias is False

        # Save and reload
        cfg.save(tmp_path)
        loaded = GrillyConfig.load(tmp_path)
        assert loaded.hidden_size == cfg.hidden_size
        assert loaded.model_type == cfg.model_type
        assert loaded.norm_type == cfg.norm_type

    def test_export_function_exists_and_callable(self):
        """export_to_grilly is importable and has the expected signature."""
        from optimum.grilly.export import export_to_grilly

        assert callable(export_to_grilly)

    def test_cli_entry_exists(self):
        """CLI entry point _cli_main is importable."""
        from optimum.grilly.export import _cli_main

        assert callable(_cli_main)
