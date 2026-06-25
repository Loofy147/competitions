import torch
from transformers import BitsAndBytesConfig
from typing import Any, Dict

class OptimizationSkill:
    """
    Skill: Model Performance Optimization.
    Extracted from Gemma-2 4-bit and Unsloth kernels.
    """
    @staticmethod
    def get_4bit_config() -> BitsAndBytesConfig:
        """Returns 4-bit quantization configuration for BitsAndBytes."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    @staticmethod
    def apply_mixed_precision():
        """Helper for enabling mixed precision context."""
        return torch.cuda.amp.autocast()

    @staticmethod
    def optimize_batch_padding(df, length_col="length"):
        """Sorts dataframe by length to minimize padding overhead."""
        return df.sort_values(length_col, ascending=False)
