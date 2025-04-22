"""Source code for the Feature Bank of the StreamV2V paper.

References:
    StreamV2V: Looking Backward: Streaming Video-to-Video Translation with Feature Banks - Liang et al.,
    2025, URL: https://arxiv.org/abs/2405.15757
"""

from typing import Optional, Deque, List, Tuple, Dict, Any
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils import USE_PEFT_BACKEND


def extract_nn_features(
    x: torch.Tensor, y: torch.Tensor, threshold: float = 0.9
) -> torch.Tensor:
    """
    Selects features from x or nearest neighbors from y based on cosine similarity.

    Args:
        x (torch.Tensor): Input features tensor.
        y (torch.Tensor): Neighbor features tensor.
         threshold (float): Cosine similarity threshold for neighbor selection.

    Returns:
        torch.Tensor: Selected features tensor.
    """
    if isinstance(x, deque):
        x = torch.cat(list(x), dim=1)
    if isinstance(y, deque):
        y = torch.cat(list(y), dim=1)

    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)

    cosine_similarity = torch.matmul(x_norm, y_norm.transpose(1, 2))
    max_cosine_values, nearest_neighbors_indices = torch.max(cosine_similarity, dim=-1)
    mask = max_cosine_values < threshold

    indices_expanded = nearest_neighbors_indices.unsqueeze(-1).expand(
        -1, -1, x_norm.size(-1)
    )
    nearest_neighbor_tensor = torch.gather(y, 1, indices_expanded)
    selected_tensor = torch.where(mask.unsqueeze(-1), x, nearest_neighbor_tensor)

    return selected_tensor


class FeatureBank:
    """
    Feature bank to cache and retrieve attention keys, values, and outputs.

    Args:
        max_frames (int): Maximum number of frames to store in the cache.
    """

    def __init__(self, max_frames: int = 4):
        self.cached_key: Deque[torch.FloatTensor] = deque(maxlen=max_frames)
        self.cached_value: Deque[torch.FloatTensor] = deque(maxlen=max_frames)
        self.cached_output: Deque[torch.FloatTensor] = deque(maxlen=max_frames)

    def add(
        self,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        output: torch.FloatTensor,
    ):
        """Add key, value, and output tensors to the feature bank."""
        self.cached_key.append(key.clone())
        self.cached_value.append(value.clone())
        self.cached_output.append(output.clone())

    def get(
        self,
    ) -> Tuple[
        List[torch.FloatTensor], List[torch.FloatTensor], List[torch.FloatTensor]
    ]:
        """Retrieve cached keys, values, and outputs as lists."""
        return list(self.cached_key), list(self.cached_value), list(self.cached_output)

    def clear(self):
        """Clear all cached features."""
        self.cached_key.clear()
        self.cached_value.clear()
        self.cached_output.clear()


class FeatureBankAttentionProcessor:
    """
    Attention processor with feature injection.
    """

    DEBUG_PRINTS = (
        False  # Class-level debug flag, turn on/off all debug prints from this class
    )

    def __init__(
        self,
        use_feature_injection=True,
        feature_injection_strength=0.8,
        feature_similarity_threshold=0.98,
        interval=4,
        max_frames=4,
    ):
        self.use_feature_injection = use_feature_injection
        self.fi_strength = feature_injection_strength
        self.threshold = feature_similarity_threshold
        self.frame_id = 0
        self.interval = interval
        self.feature_bank = FeatureBank(max_frames=max_frames)
        self.key_projection: Optional[nn.Linear] = (
            None  # Linear layer for key projection
        )
        self.value_projection: Optional[nn.Linear] = (
            None  # Linear layer for value projection
        )
        self.output_projection: Optional[nn.Linear] = (
            None  # Linear layer for output projection
        )

    def _debug_print(self, *args, **kwargs):
        """Conditional debug print based on class-level DEBUG_PRINTS flag."""
        if self.DEBUG_PRINTS:
            print("[FeatureBankAttentionProcessor DEBUG]", *args, **kwargs)

    def _project_tensor(
        self,
        tensor: torch.Tensor,
        projection_layer: Optional[nn.Linear],
        expected_dim: int,
        layer_type: str,
    ) -> torch.Tensor:
        """
        Handles tensor projection if dimension mismatch is detected.
        Creates projection layer if needed and applies it.
        """
        if (
            projection_layer is None
            or projection_layer.in_features != tensor.shape[-1]
            or projection_layer.out_features != expected_dim
        ):
            self._debug_print(
                f"\n--- DEBUG: {layer_type} dimension mismatch! Projecting cached {layer_type.lower()}s. ---"
            )
            self._debug_print(
                f"Original {layer_type.lower()} dimension: {tensor.shape[-1]}, Expected: {expected_dim}"
            )
            new_projection_layer = (
                nn.Linear(tensor.shape[-1], expected_dim)
                .to(tensor.device)
                .to(tensor.dtype)
            )
            self._debug_print(
                f"Creating new {layer_type.lower()} projection layer: In Features={tensor.shape[-1]}, Out Features={expected_dim}"
            )
            if layer_type == "Key":
                self.key_projection = new_projection_layer
            elif layer_type == "Value":
                self.value_projection = new_projection_layer
            elif layer_type == "Output":
                self.output_projection = new_projection_layer
            projection_layer = (
                new_projection_layer  # Use the newly created or updated layer
            )

        projected_tensor = projection_layer(tensor)
        self._debug_print(
            f"Shape of cached {layer_type.lower()}s after projection: {projected_tensor.shape}"
        )
        return projected_tensor

    def __call__(
        self,
        attn: nn.Module,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        """
        Forward pass of feature bank attention with feature injection.
        """
        args = () if USE_PEFT_BACKEND else (scale,)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)

        # Self-attention detection (heuristic based on input feature dimensions)
        is_self_attn = attn.to_k.in_features == attn.to_q.in_features
        current_encoder_hidden_states = (
            hidden_states if is_self_attn else encoder_hidden_states
        )

        self._debug_print("\n--- Attention Module Info ---")
        self._debug_print("Module type:", type(attn))
        self._debug_print("Module:", attn)
        self._debug_print("Is self-attention (heuristic):", is_self_attn)
        self._debug_print("Shape of hidden_states:", hidden_states.shape)
        self._debug_print(
            "Shape of current_encoder_hidden_states before attn.to_k:",
            (
                current_encoder_hidden_states.shape
                if current_encoder_hidden_states is not None
                else "None"
            ),
        )
        self._debug_print("Shape of attn.to_k.weight:", attn.to_k.weight.shape)

        key = attn.to_k(current_encoder_hidden_states)
        value = attn.to_v(current_encoder_hidden_states)

        cached_key_tensor = key.clone()
        cached_value_tensor = value.clone()

        num_heads = attn.heads  # Assuming attn.heads exists in CrossAttention
        head_dim = query.shape[-1] // num_heads
        query = (
            query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2).contiguous()
        )  # [batch_size, num_heads, seq_len_q, head_dim]

        if is_self_attn:
            (
                cached_keys_list,
                cached_values_list,
                cached_outputs,
            ) = self.feature_bank.get()

            self._debug_print(
                "\n--- Feature Bank Get Output ---"
            )  # Separator for clarity
            self._debug_print(
                f"Type of cached_keys_list from feature_bank.get(): {type(cached_keys_list)}"
            )
            self._debug_print(
                f"Content of cached_keys_list from feature_bank.get(): {cached_keys_list}"
            )

            if (
                cached_keys_list
                and cached_keys_list[0] is not None
                and torch.is_tensor(cached_keys_list[0])
            ):
                cached_key_tensor_from_list = cached_keys_list[0]
                cached_key_tensor_from_list = self._project_tensor(
                    cached_key_tensor_from_list,
                    self.key_projection,
                    key.shape[-1],
                    "Key",
                )

                self._debug_print("\n--- DEBUG: Before torch.cat (Keys) ---")
                self._debug_print(
                    "Module:", attn
                )  # Print the module itself for identification
                self._debug_print("Type of key:", type(key))
                self._debug_print("Shape of key:", key.shape)
                self._debug_print(
                    "Type of cached_key_tensor_from_list:",
                    type(cached_key_tensor_from_list),
                )
                self._debug_print(
                    "Shape of cached_key_tensor_from_list:",
                    cached_key_tensor_from_list.shape,
                )

                key = torch.cat([key, cached_key_tensor_from_list], dim=1)

            if (
                cached_values_list
                and cached_values_list[0] is not None
                and torch.is_tensor(cached_values_list[0])
            ):
                cached_value_tensor_from_list = cached_values_list[0]
                cached_value_tensor_from_list = self._project_tensor(
                    cached_value_tensor_from_list,
                    self.value_projection,
                    value.shape[-1],
                    "Value",
                )

                self._debug_print("\n--- DEBUG: Before torch.cat (Values) ---")
                self._debug_print(
                    "Module:", attn
                )  # Print the module itself for identification
                self._debug_print("Type of value:", type(value))
                self._debug_print("Shape of value:", value.shape)
                self._debug_print(
                    "Type of cached_value_tensor_from_list:",
                    type(cached_value_tensor_from_list),
                )
                self._debug_print(
                    "Shape of cached_value_tensor_from_list:",
                    cached_value_tensor_from_list.shape,
                )

                value = torch.cat([value, cached_value_tensor_from_list], dim=1)

            key = (
                key.view(-1, key.shape[1], num_heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )  # [batch_size, num_heads, seq_len_kv, head_dim] (seq_len_kv == key_tokens)
            value = (
                value.view(-1, value.shape[1], num_heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )  # [batch_size, num_heads, seq_len_kv, head_dim] (seq_len_kv == key_tokens)
        else:
            key = (
                key.view(batch_size, -1, num_heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )  # [batch_size, num_heads, seq_len_kv, head_dim] (seq_len_kv == key_tokens)
            value = (
                value.view(batch_size, -1, num_heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )  # [batch_size, num_heads, seq_len_kv, head_dim] (seq_len_kv == key_tokens)

        # --- Manual attention implementation ---
        scale_factor = 1.0 / query.shape[-1] ** 0.5
        query = query * scale_factor
        attn_weights = torch.bmm(
            query.view(batch_size * num_heads, -1, head_dim),
            key.view(batch_size * num_heads, -1, head_dim).transpose(-2, -1),
        )  # [batch_size * num_heads, seq_len_q, seq_len_kv]
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.bmm(
            attn_weights, value.view(batch_size * num_heads, -1, head_dim)
        )  # [batch_size * num_heads, seq_len_q, head_dim]
        hidden_states = (
            attn_output.view(batch_size, num_heads, -1, head_dim)
            .transpose(1, 2)
            .contiguous()
        )  # [batch_size, seq_len_q, num_heads * head_dim]
        # --- End of manual attention implementation ---

        self._debug_print(f"Type of attn before to_out: {type(attn)}")  # Debug print

        hidden_states = hidden_states.to(
            query.dtype
        )  # Keep this line to ensure correct dtype
        hidden_states = hidden_states.reshape(
            batch_size, -1, num_heads * head_dim
        ).contiguous()  # Modified reshape - removed transpose after reshape

        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        rescale_output_factor = 1.0
        cached_output_tensor = hidden_states.clone()

        if self.use_feature_injection:
            _, _, cached_outputs = self.feature_bank.get()
            self._debug_print(
                f"Type of cached_outputs: {type(cached_outputs)}"
            )  # Debug print

            if (
                cached_outputs
                and cached_outputs[0] is not None
                and cached_outputs[0].numel() > 0
            ):  # Corrected condition
                cached_output_tensor_from_list = cached_outputs[
                    0
                ]  # Get the cached output tensor
                cached_output_tensor_from_list = self._project_tensor(
                    cached_output_tensor_from_list,
                    self.output_projection,
                    hidden_states.shape[-1],
                    "Output",
                )

                self._debug_print("\n--- DEBUG: Before extract_nn_features ---")
                self._debug_print("Shape of hidden_states:", hidden_states.shape)
                self._debug_print(
                    "Shape of cached_output_tensor_from_list:",
                    cached_output_tensor_from_list.shape,
                )

                nn_hidden_states = extract_nn_features(
                    hidden_states,
                    cached_output_tensor_from_list,
                    threshold=self.threshold,
                )  # Pass PROJECTED tensor
                hidden_states = (
                    hidden_states * (1 - self.fi_strength)
                    + self.fi_strength * nn_hidden_states
                )

        hidden_states = hidden_states / rescale_output_factor

        if self.frame_id % self.interval == 0 and is_self_attn:
            self.feature_bank.add(
                cached_key_tensor, cached_value_tensor, cached_output_tensor
            )

        self.frame_id += 1
        return hidden_states


class FeatureBankAttentionProcessorNode:
    """
    ComfyUI Node for Feature Bank Attention Processor.
    """

    def __init__(self):
        super().__init__()
        self.cross_attention_hook = None
        self.feature_bank_processor = FeatureBankAttentionProcessor()

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Tuple[str, Dict[str, Any]]]]:
        """Input types for FeatureBankAttentionProcessorNode node."""
        return {
            "required": {
                "model": ("MODEL",),
                "use_feature_injection": (
                    "BOOLEAN",
                    {"default": True},
                ),  # Configurable FI
                "feature_injection_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
                ),  # Configurable FI strength
                "feature_similarity_threshold": (
                    "FLOAT",
                    {"default": 0.98, "min": 0.0, "max": 1.0, "step": 0.01},
                ),  # Configurable threshold
                "feature_cache_interval": (
                    "INT",
                    {"default": 4, "min": 1, "max": 64},
                ),  # Configurable interval
                "feature_bank_max_frames": (
                    "INT",
                    {"default": 4, "min": 1, "max": 16},
                ),  # Configurable max frames in bank
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "update_node"
    CATEGORY = "StreamPack/model_patches/unet"

    def update_node(
        self,
        model: Any,
        use_feature_injection: bool,
        feature_injection_strength: float,
        feature_similarity_threshold: float,
        feature_cache_interval: int,
        feature_bank_max_frames: int,
    ) -> Tuple[Any]:
        """
        Update method for FeatureBankAttentionProcessorNode. Sets up feature injection and attention hooks.
        """
        print(f"[FeatureBankAttentionProcessorNode] Initializing ")

        # Update FeatureBankProcessor configuration from node inputs on each update call
        self.feature_bank_processor.use_feature_injection = use_feature_injection
        self.feature_bank_processor.fi_strength = feature_injection_strength
        self.feature_bank_processor.threshold = feature_similarity_threshold
        self.feature_bank_processor.interval = feature_cache_interval
        self.feature_bank_processor.feature_bank = FeatureBank(
            max_frames=feature_bank_max_frames
        )  # Re-instantiate feature bank with new max_frames if it changes

        def cross_attention_forward(
            module: nn.Module,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            value: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> (
            torch.Tensor
        ):  # Added **kwargs to accept potential extra args from original forward
            """Optimized cross attention using FeatureBankAttentionProcessor."""
            context = x if context is None else context
            encoder_hidden_states = context  # Assuming 'context' in cross_attention_forward corresponds to encoder_hidden_states

            return self.feature_bank_processor(
                attn=module,  # Pass the current module as 'attn'
                hidden_states=x,
                encoder_hidden_states=encoder_hidden_states,
                scale=1.0,  # Or pass scale if it's relevant, currently hardcoded to 1.0 in processor
            )

        def hook_cross_attention(
            module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor
        ) -> torch.Tensor:
            """Forward hook to replace cross-attention forward method."""
            if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                if not hasattr(module, "_original_forward"):
                    module._original_forward = module.forward
                module.forward = lambda *args, **kwargs: cross_attention_forward(
                    module, *args, **kwargs
                )
            return output

        # Remove old hooks if they exist
        if self.cross_attention_hook is not None:
            self.cross_attention_hook.remove()

        # Clone model and apply hooks
        m = model.clone()

        def register_hooks(module: nn.Module):
            """Recursively register forward hooks for cross-attention modules."""
            if isinstance(module, torch.nn.Module) and hasattr(module, "to_q"):
                self.cross_attention_hook = module.register_forward_hook(
                    hook_cross_attention
                )

        m.model.apply(register_hooks)
        return (m,)
