from ..src.batched_denoising.batched_denoising import StreamBatchSampler, StreamScheduler, StreamFrameBuffer

NODE_CLASS_MAPPINGS = {
    "StreamBatchSampler": StreamBatchSampler,
    "StreamScheduler": StreamScheduler,
    "StreamFrameBuffer": StreamFrameBuffer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamBatchSampler": "Stream Batch Sampler",
    "StreamScheduler": "Stream Scheduler",
    "StreamFrameBuffer": "Stream Frame Buffer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]