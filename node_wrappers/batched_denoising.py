from ..src.batched_denoising.batched_denoising import (
    StreamBatchSampler,
    StreamScheduler,
)

NODE_CLASS_MAPPINGS = {
    "StreamBatchSampler": StreamBatchSampler,
    "StreamScheduler": StreamScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamBatchSampler": "Stream Batch Sampler",
    "StreamScheduler": "Stream Scheduler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
