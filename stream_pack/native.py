"""Service boundaries around native ComfyUI workflow nodes."""

from functools import lru_cache


@lru_cache(maxsize=1)
def _checkpoint(name):
    from comfy.nodes_context import get_nodes

    loader = get_nodes().NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
    return getattr(loader, loader.FUNCTION)(ckpt_name=name)


class ResidentCheckpoint:
    """Retain immutable model weights while the core clears request caches."""

    CATEGORY = "StreamPack/Service"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": ("STRING",)}}

    def load(self, ckpt_name):
        return _checkpoint(ckpt_name)


class AudioSeconds:
    CATEGORY = "StreamPack/Service"
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"seconds": ("FLOAT", {"default": 30, "min": 1, "max": 120})}
        }

    def execute(self, seconds):
        return (float(seconds),)


class AudioArtifact:
    CATEGORY = "StreamPack/Service"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    def execute(self, audio):
        import json

        import soundfile

        from .models import artifact, output_path

        waveform = audio["waveform"].detach().cpu().numpy()[0].T
        path = output_path(".wav")
        soundfile.write(path, waveform, audio["sample_rate"])
        result = artifact(path, "audio/wav")
        result["usage"] = {"audio_seconds": len(waveform) / audio["sample_rate"]}
        return (json.dumps(result),)


def ace_workflow():
    return {
        "1": {
            "class_type": "StreamPackResidentCheckpoint",
            "inputs": {"ckpt_name": "ace_step_1.5_turbo_aio.safetensors"},
        },
        "2": {"class_type": "StreamPackAudioSeconds", "inputs": {"seconds": 30}},
        "3": {
            "class_type": "TextEncodeAceStepAudio1.5",
            "inputs": {
                "clip": ["1", 1],
                "tags": "",
                "lyrics": "",
                "seed": 0,
                "bpm": 120,
                "duration": ["2", 0],
                "timesignature": "4",
                "language": "en",
                "keyscale": "C major",
                "generate_audio_codes": False,
                "cfg_scale": 2.0,
                "temperature": 0.85,
                "top_p": 0.9,
                "top_k": 0,
                "min_p": 0.0,
            },
        },
        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 3.0},
        },
        "5": {
            "class_type": "EmptyAceStep1.5LatentAudio",
            "inputs": {"seconds": ["2", 0], "batch_size": 1},
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["3", 0]},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["3", 0],
                "negative": ["6", 0],
                "latent_image": ["5", 0],
                "seed": 0,
                "steps": 8,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["7", 0], "vae": ["1", 2]},
        },
        "9": {"class_type": "StreamPackAudioArtifact", "inputs": {"audio": ["8", 0]}},
        "10": {"class_type": "StreamPackOutput", "inputs": {"value": ["9", 0]}},
    }
