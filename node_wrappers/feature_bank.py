"""A ComfyUI native node implementation of the Feature Bank of the StreamV2V paper.


References:
    StreamV2V: A Deep Generative Model for High-fidelity View Synthesis - Liang et al.,
    2025, URL: https://arxiv.org/abs/2405.15757
"""

from ..src.feature_bank.feature_bank import (
    FeatureBankAttentionProcessorNode,
    FeatureBankAttentionProcessor,
)

NODE_CLASS_MAPPINGS = {
    "FeatureBankAttentionProcessor": FeatureBankAttentionProcessorNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FeatureBankAttentionProcessor": FeatureBankAttentionProcessor
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
