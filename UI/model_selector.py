"""
model_selector.py
-----------------
Model selection for the AutoInsight pipeline.

Strategy:
- Always start with Haiku: fast, cheap, sufficient for most EDA tasks.
- Always escalate to Sonnet on the final repair attempt as a hard fallback.
"""

MODEL_HAIKU  = "claude-haiku-4-5-20251001"
MODEL_SONNET = "claude-sonnet-4-6"
MAX_RETRIES  = 3


def select_model_for_attempt(attempt: int) -> str:
    """
    Returns the model for a given repair attempt index (0-based).
    - Attempts 0 to MAX_RETRIES-2 : Haiku.
    - Final attempt (MAX_RETRIES-1): Sonnet as a hard fallback.
    """
    if attempt >= MAX_RETRIES - 1:
        return MODEL_SONNET
    return MODEL_HAIKU


def model_label(model: str) -> str:
    """Human-readable short label for display in the UI."""
    if "haiku" in model:
        return "Haiku"
    if "sonnet" in model:
        return "Sonnet"
    return model
