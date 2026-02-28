"""Tests for pipeline helpers."""


import numpy as np

from optimum.grilly.pipelines import (
    grilly_feature_extraction_pipeline,
    grilly_text_generation_pipeline,
)


class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def __call__(self, text, return_tensors="np"):
        # Encode each character as a token ID
        ids = np.array([[ord(c) % 50 for c in text[:10]]], dtype=np.int64)
        mask = np.ones_like(ids, dtype=np.int64)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, token_ids, skip_special_tokens=True):
        return "generated text output"


class MockCausalLM:
    """Minimal CausalLM mock."""

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50):
        batch, seq = input_ids.shape
        new_tokens = np.random.randint(0, 50, (batch, max_new_tokens))
        return np.concatenate([input_ids, new_tokens], axis=-1)


class MockFeatureExtractor:
    """Minimal feature extraction mock."""

    def forward(self, input_ids, attention_mask=None):
        batch, seq = input_ids.shape
        hidden = np.random.randn(batch, seq, 64).astype(np.float32)
        return {"last_hidden_state": hidden}


class TestTextGeneration:
    def test_basic_generation(self):
        model = MockCausalLM()
        tokenizer = MockTokenizer()
        result = grilly_text_generation_pipeline(model, tokenizer, "hello", max_new_tokens=5)
        assert isinstance(result, str)
        assert len(result) > 0


class TestFeatureExtraction:
    def test_mean_pooling(self):
        model = MockFeatureExtractor()
        tokenizer = MockTokenizer()
        result = grilly_feature_extraction_pipeline(model, tokenizer, "hello", pooling="mean")
        assert result.shape == (1, 64)

    def test_cls_pooling(self):
        model = MockFeatureExtractor()
        tokenizer = MockTokenizer()
        result = grilly_feature_extraction_pipeline(model, tokenizer, "hello", pooling="cls")
        assert result.shape == (1, 64)

    def test_last_pooling(self):
        model = MockFeatureExtractor()
        tokenizer = MockTokenizer()
        result = grilly_feature_extraction_pipeline(model, tokenizer, "hello", pooling="last")
        assert result.shape == (1, 64)
