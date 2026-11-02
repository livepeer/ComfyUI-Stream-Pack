"""Lazy local model adapters. Artifacts stay in operator-controlled directories.

All bundled capabilities remain candidates until their exact GPU image passes acceptance.
No adapter calls a hosted inference API.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from .catalog import load_catalog


def input_path(value: str) -> Path:
    root = Path(os.environ["STREAM_PACK_INPUT_DIR"]).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Input must be an existing staged media file")
    return path


def output_path(suffix: str) -> Path:
    root = Path(os.environ["STREAM_PACK_OUTPUT_DIR"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root / (uuid4().hex + suffix)


def artifact(path: Path, media_type: str) -> dict:
    return {"artifacts": [{"path": str(path), "media_type": media_type}]}


@lru_cache(maxsize=1)
def load_model(capability: str):
    """One resident model per execution worker; worker exit releases CUDA memory."""
    import torch

    descriptor = load_catalog()[capability].document
    model = next(
        (
            part
            for part in descriptor.get("components", [])
            if part.get("role") == "execution"
        ),
        descriptor["model"],
    )
    source, revision = model["source"], model["revision"]
    from huggingface_hub import snapshot_download

    # Reference implementations may forget to forward revision to subfolder
    # loaders. A pinned local snapshot keeps every nested lookup offline and
    # bound to the same checkpoint, without relying on a mutable refs/main.
    source = snapshot_download(source, revision=revision, local_files_only=True)
    options = {"local_files_only": True}
    if capability in {"flux-image", "qwen-edit", "wan-video"}:
        from diffusers import (
            Flux2KleinPipeline,
            QwenImageEditPlusPipeline,
            WanImageToVideoPipeline,
        )

        cls = {
            "flux-image": Flux2KleinPipeline,
            "qwen-edit": QwenImageEditPlusPipeline,
            "wan-video": WanImageToVideoPipeline,
        }[capability]
        pipeline = cls.from_pretrained(source, torch_dtype=torch.bfloat16, **options)
        if capability in {"qwen-edit", "wan-video"}:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")
        return pipeline
    if capability == "qwen-chat":
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        return (
            AutoModelForImageTextToText.from_pretrained(
                source,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                ),
                **options,
            ),
            AutoProcessor.from_pretrained(source, **options),
        )
    if capability == "whisper":
        from transformers import pipeline

        return pipeline(
            "automatic-speech-recognition",
            model=source,
            device="cuda",
            torch_dtype=torch.float16,
            model_kwargs={"local_files_only": True},
        )
    if capability == "qwen-speech":
        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel.from_pretrained(
            source, device_map="cuda:0", dtype=torch.bfloat16, **options
        )
    if capability == "birefnet":
        from transformers import AutoModelForImageSegmentation

        # The architecture is supplied by this immutable model revision, never by user input.
        return (
            AutoModelForImageSegmentation.from_pretrained(
                source, trust_remote_code=True, **options
            )
            .to("cuda")
            .eval()
        )
    if capability == "triposr":
        from tsr.system import TSR

        tokenizer = next(
            part
            for part in descriptor["components"]
            if part["role"] == "image_tokenizer"
        )
        directory = snapshot_download(
            tokenizer["source"], revision=tokenizer["revision"], local_files_only=True
        )
        model = TSR.from_pretrained(
            source,
            config_name="config.yaml",
            weight_name="model.ckpt",
            image_tokenizer_path=directory,
        ).to("cuda")
        model.renderer.set_chunk_size(8192)
        return model
    raise ValueError("Unknown local capability")


def execute(capability: str, inputs: dict) -> dict:
    import torch
    from PIL import Image

    spec = load_catalog()[capability]
    spec.validate(inputs)
    model = load_model(capability)
    with torch.inference_mode():
        if capability == "qwen-chat":
            lm, processor = model
            messages = inputs["messages"]
            # URL fetching belongs to the gateway staging boundary, never the model loader.
            if any(not isinstance(m.get("content"), str) for m in messages):
                raise ValueError("This candidate supports text messages only")
            encoded = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(lm.device)
            generated = lm.generate(
                **encoded,
                max_new_tokens=inputs.get("max_tokens", 1024),
                do_sample=False,
            )
            count = encoded["input_ids"].shape[1]
            text = processor.batch_decode(
                generated[:, count:], skip_special_tokens=True
            )[0]
            return {
                "text": text,
                "artifacts": [],
                "usage": {
                    "input_tokens": count,
                    "output_tokens": generated.shape[1] - count,
                },
            }
        if capability in {"flux-image", "qwen-edit", "wan-video"}:
            kwargs = {
                "prompt": inputs["prompt"],
                "generator": torch.Generator(device="cuda").manual_seed(
                    inputs.get("seed", 0)
                ),
            }
            if "image" in inputs:
                kwargs["image"] = Image.open(input_path(inputs["image"])).convert("RGB")
            if capability == "wan-video":
                from diffusers.utils import export_to_video

                kwargs["num_frames"] = inputs.get("frames", 81)
                path = output_path(".mp4")
                export_to_video(model(**kwargs).frames[0], str(path), fps=16)
                return artifact(path, "video/mp4")
            kwargs.update(
                width=inputs.get("width", 1024), height=inputs.get("height", 1024)
            )
            if capability == "flux-image":
                kwargs.update(num_inference_steps=4, guidance_scale=1.0)
            path = output_path(".png")
            model(**kwargs).images[0].save(path)
            return artifact(path, "image/png")
        if capability == "whisper":
            kwargs = {"language": inputs["language"]} if inputs.get("language") else {}
            result = model(str(input_path(inputs["audio"])), generate_kwargs=kwargs)
            return {"text": result["text"], "artifacts": []}
        if capability == "qwen-speech":
            import soundfile

            waveforms, rate = model.generate_voice_clone(
                text=inputs["input"],
                language=inputs.get("language", "Auto"),
                ref_audio=str(input_path(inputs["reference_audio"])),
                ref_text=inputs["reference_text"],
            )
            path = output_path(".wav")
            soundfile.write(path, waveforms[0], rate)
            return artifact(path, "audio/wav")
        if capability == "birefnet":
            from torchvision import transforms

            image = Image.open(input_path(inputs["image"])).convert("RGB")
            tensor = (
                transforms.Compose(
                    [
                        transforms.Resize((1024, 1024)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )(image)
                .unsqueeze(0)
                .to("cuda")
            )
            mask = model(tensor)[-1].sigmoid().cpu()[0].squeeze()
            image.putalpha(transforms.ToPILImage()(mask).resize(image.size))
            path = output_path(".png")
            image.save(path)
            return artifact(path, "image/png")
        if capability == "triposr":
            image = Image.open(input_path(inputs["image"])).convert("RGB")
            codes = model([image], device="cuda")
            mesh = model.extract_mesh(codes, has_vertex_color=True)[0]
            path = output_path(".obj")
            mesh.export(str(path))
            return artifact(path, "model/obj")
    raise ValueError("Unknown local capability")
