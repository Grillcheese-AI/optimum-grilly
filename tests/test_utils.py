import numpy as np
import pytest
from safetensors.numpy import save_file

from optimum.grilly.utils import load_weights, save_weights


class TestWeightIO:
    def test_save_and_load(self, tmp_path):
        """Weights round-trip through safetensors."""
        weights = {
            "model.embed_tokens.weight": np.random.randn(100, 64).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32),
            "lm_head.weight": np.random.randn(100, 64).astype(np.float32),
        }
        save_weights(weights, tmp_path / "model.safetensors")
        loaded = load_weights(tmp_path / "model.safetensors")
        for key in weights:
            np.testing.assert_array_equal(loaded[key], weights[key])

    def test_load_weights_dtype(self, tmp_path):
        """Loaded weights are float32."""
        weights = {"w": np.ones((10, 10), dtype=np.float32)}
        save_weights(weights, tmp_path / "model.safetensors")
        loaded = load_weights(tmp_path / "model.safetensors")
        assert loaded["w"].dtype == np.float32

    def test_load_sharded_weights(self, tmp_path):
        """Multiple safetensors shards are merged correctly."""
        shard1 = {"layer.0.weight": np.ones((4, 4), dtype=np.float32)}
        shard2 = {"layer.1.weight": np.zeros((4, 4), dtype=np.float32)}
        save_file(shard1, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file(shard2, str(tmp_path / "model-00002-of-00002.safetensors"))
        loaded = load_weights(tmp_path)
        assert "layer.0.weight" in loaded
        assert "layer.1.weight" in loaded

    def test_load_missing_path(self):
        """FileNotFoundError for nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_weights("/nonexistent/path")

    def test_load_empty_dir(self, tmp_path):
        """FileNotFoundError for directory with no safetensors."""
        with pytest.raises(FileNotFoundError, match="No .safetensors"):
            load_weights(tmp_path)
