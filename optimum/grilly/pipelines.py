"""Pipeline integration for optimum-grilly."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def grilly_text_generation_pipeline(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
) -> str:
    """Simple text generation using a GrillyModelForCausalLM.

    Args:
        model: GrillyModelForCausalLM instance.
        tokenizer: HuggingFace tokenizer.
        prompt: Input text string.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.

    Returns:
        Generated text string.
    """
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def grilly_feature_extraction_pipeline(
    model: Any,
    tokenizer: Any,
    text: str,
    pooling: str = "mean",
) -> np.ndarray:
    """Extract embeddings using a GrillyModelForFeatureExtraction.

    Args:
        model: GrillyModelForFeatureExtraction instance.
        tokenizer: HuggingFace tokenizer.
        text: Input text string.
        pooling: Pooling strategy ("mean", "cls", "last").

    Returns:
        Embedding vector as numpy array of shape (1, hidden_size).
    """
    inputs = tokenizer(text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask: Optional[np.ndarray] = inputs.get("attention_mask")

    # Convert binary mask (1=attend, 0=mask) to additive mask (0=attend, -1e9=mask)
    additive_mask: Optional[np.ndarray] = None
    if attention_mask is not None:
        additive_mask = (1.0 - attention_mask.astype(np.float32)) * -1e9
        additive_mask = additive_mask[:, np.newaxis, np.newaxis, :]

    outputs = model.forward(input_ids, additive_mask)
    hidden_states = outputs["last_hidden_state"]

    if pooling == "cls":
        return hidden_states[:, 0]
    elif pooling == "last":
        return hidden_states[:, -1]
    else:  # mean pooling
        if attention_mask is not None:
            mask = attention_mask[:, :, np.newaxis].astype(np.float32)
            return (hidden_states * mask).sum(axis=1) / mask.sum(axis=1)
        return hidden_states.mean(axis=1)
