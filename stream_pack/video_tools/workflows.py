"""ComfyUI API workflows for the ffmpeg capabilities.

The graphs use core nodes (LoadVideo, ImageScale, Video Slice, CreateVideo,
SaveVideo, LoadAudio, AudioAdjustVolume, SaveAudio, ImageFromBatch, SaveImage,
ConcatenateVideo, LoadImage, ImageBatch). PyAV inside those nodes is Comfy's
own encoder. Nothing here starts an ffmpeg process.
"""

from __future__ import annotations

import copy
from typing import Any

ROTATION_LABELS = {
    0: "none",
    90: "90 degrees",
    180: "180 degrees",
    270: "270 degrees",
}
FLIP_LABELS = {
    "horizontal": "y-axis: horizontally",
    "vertical": "x-axis: vertically",
}


def live_filter(height: int, rotation: int, flip: str) -> dict[str, Any]:
    """Frame workflow for the streaming pipeline: LoadTensor through SaveTensor."""
    prompt: dict[str, Any] = {
        "1": {"class_type": "LoadTensor", "inputs": {}},
        "2": {
            "class_type": "ImageRotate",
            "inputs": {
                "image": ["1", 0],
                "rotation": ROTATION_LABELS[int(rotation)],
            },
        },
        "3": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["2", 0],
                "upscale_method": "lanczos",
                "width": 0,
                "height": int(height),
                "crop": "disabled",
            },
        },
    }
    source = "3"
    if flip in FLIP_LABELS:
        prompt["4"] = {
            "class_type": "ImageFlip",
            "inputs": {"image": ["3", 0], "flip_method": FLIP_LABELS[flip]},
        }
        source = "4"
    prompt["5"] = {
        "class_type": "SaveTensor",
        "inputs": {"images": [source, 0]},
    }
    return prompt


def scale_video(
    filename: str, prefix: str, height: int, width: int, mode: str
) -> dict[str, Any]:
    """ImageScale keeps aspect when the other side is 0. Stretch uses both."""
    crop = "center" if mode == "crop" else "disabled"
    if mode == "stretch":
        width = int(width or 0)
        height = int(height or 0)
    elif width and not height:
        height = 0
    else:
        width = 0
        height = int(height or 720)
    return {
        "1": _load_video(filename),
        "2": _components("1"),
        "3": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["2", 0],
                "upscale_method": "lanczos",
                "width": int(width or 0),
                "height": int(height or 0),
                "crop": crop,
            },
        },
        "4": _create_video("3", "2"),
        "5": _save_video("4", prefix),
    }


def trim_video(
    filename: str, prefix: str, start_time: float, duration: float
) -> dict[str, Any]:
    return {
        "1": _load_video(filename),
        "2": {
            "class_type": "Video Slice",
            "inputs": {
                "video": ["1", 0],
                "start_time": float(start_time),
                "duration": float(duration),
                "strict_duration": False,
            },
        },
        "3": _save_video("2", prefix),
    }


def loudnorm(filename: str, prefix: str, volume_db: int) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": filename}},
        "2": {
            "class_type": "AudioAdjustVolume",
            "inputs": {"audio": ["1", 0], "volume": int(volume_db)},
        },
        "3": {
            "class_type": "SaveAudio",
            "inputs": {"audio": ["2", 0], "filename_prefix": prefix},
        },
    }


def extract_frame(filename: str, prefix: str, batch_index: int) -> dict[str, Any]:
    return {
        "1": _load_video(filename),
        "2": _components("1"),
        "3": {
            "class_type": "ImageFromBatch",
            "inputs": {
                "image": ["2", 0],
                "batch_index": int(batch_index),
                "length": 1,
            },
        },
        "4": {
            "class_type": "SaveImage",
            "inputs": {"images": ["3", 0], "filename_prefix": prefix},
        },
    }


def metadata(filename: str) -> dict[str, Any]:
    return {
        "1": _load_video(filename),
        "2": _components("1"),
    }


def merge_audio_video(video: str, audio: str, prefix: str) -> dict[str, Any]:
    return {
        "1": _load_video(video),
        "2": _components("1"),
        "3": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "4": _create_video("2", "2", audio_node="3"),
        "5": _save_video("4", prefix),
    }


def merge_videos(filenames: list[str], prefix: str) -> dict[str, Any]:
    prompt: dict[str, Any] = {}
    links = []
    for index, name in enumerate(filenames):
        node_id = str(index + 1)
        prompt[node_id] = _load_video(name)
        links.append([node_id, 0])
    concat_id = str(len(filenames) + 1)
    inputs: dict[str, Any] = {"codec": "auto"}
    for index, link in enumerate(links):
        inputs[f"videos.video{index}"] = link
    prompt[concat_id] = {"class_type": "ConcatenateVideo", "inputs": inputs}
    prompt[str(len(filenames) + 2)] = _save_video(concat_id, prefix)
    return prompt


def images_to_video(
    filenames: list[tuple[str, int]], prefix: str, fps: float
) -> dict[str, Any]:
    """Repeat each still, batch them, then CreateVideo."""
    prompt: dict[str, Any] = {}
    next_id = 1
    batch_source = ""
    for name, frames in filenames:
        load_id = str(next_id)
        next_id += 1
        prompt[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        current = load_id
        if int(frames) > 1:
            repeat_id = str(next_id)
            next_id += 1
            prompt[repeat_id] = {
                "class_type": "RepeatImageBatch",
                "inputs": {"image": [load_id, 0], "amount": int(frames)},
            }
            current = repeat_id
        if not batch_source:
            batch_source = current
            continue
        merge_id = str(next_id)
        next_id += 1
        prompt[merge_id] = {
            "class_type": "ImageBatch",
            "inputs": {"image1": [batch_source, 0], "image2": [current, 0]},
        }
        batch_source = merge_id
    create_id = str(next_id)
    next_id += 1
    prompt[create_id] = {
        "class_type": "CreateVideo",
        "inputs": {"images": [batch_source, 0], "fps": float(fps)},
    }
    prompt[str(next_id)] = _save_video(create_id, prefix)
    return prompt


def compose(items: list[tuple[Any, ...]], prefix: str, fps: float) -> dict[str, Any]:
    """Turn image and video keyframes into one clip.

    Each item is ``("image", filename, frames)`` or ``("video", filename)``.
    """
    prompt: dict[str, Any] = {}
    next_id = 1
    links: list[list[Any]] = []
    for item in items:
        if item[0] == "image":
            _, name, frames = item
            load_id = str(next_id)
            next_id += 1
            prompt[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
            current = load_id
            if int(frames) > 1:
                repeat_id = str(next_id)
                next_id += 1
                prompt[repeat_id] = {
                    "class_type": "RepeatImageBatch",
                    "inputs": {"image": [load_id, 0], "amount": int(frames)},
                }
                current = repeat_id
            create_id = str(next_id)
            next_id += 1
            prompt[create_id] = {
                "class_type": "CreateVideo",
                "inputs": {"images": [current, 0], "fps": float(fps)},
            }
            links.append([create_id, 0])
            continue
        _, name = item
        load_id = str(next_id)
        next_id += 1
        prompt[load_id] = _load_video(name)
        links.append([load_id, 0])
    if len(links) == 1:
        prompt[str(next_id)] = _save_video(links[0][0], prefix)
        return prompt
    concat_id = str(next_id)
    next_id += 1
    inputs: dict[str, Any] = {"codec": "auto"}
    for index, link in enumerate(links):
        inputs[f"videos.video{index}"] = link
    prompt[concat_id] = {"class_type": "ConcatenateVideo", "inputs": inputs}
    prompt[str(next_id)] = _save_video(concat_id, prefix)
    return prompt


def clone(prompt: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(prompt)


def _load_video(filename: str) -> dict[str, Any]:
    return {"class_type": "LoadVideo", "inputs": {"file": filename}}


def _components(node_id: str) -> dict[str, Any]:
    return {"class_type": "GetVideoComponents", "inputs": {"video": [node_id, 0]}}


def _create_video(
    images_node: str, components_node: str, audio_node: str = ""
) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "images": [images_node, 0],
        "fps": [components_node, 2],
    }
    if audio_node:
        inputs["audio"] = [audio_node, 0]
    else:
        inputs["audio"] = [components_node, 1]
    return {"class_type": "CreateVideo", "inputs": inputs}


def _save_video(node_id: str, prefix: str) -> dict[str, Any]:
    return {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [node_id, 0],
            "filename_prefix": prefix,
            "format": "mp4",
        },
    }
