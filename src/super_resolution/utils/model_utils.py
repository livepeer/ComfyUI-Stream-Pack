import os
import sys
import requests
import json
from pathlib import Path
import itertools
import shutil
import time

# Fix the import path to use relative import
from ..models.sr_model import SuperResolutionModel

def get_models_root_dir():
    """Get the root directory for downloaded SR models"""
    # Find the ComfyUI_SuperResolution directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Create downloaded_models directory separate from code
    return os.path.join(base_dir, "downloaded_models")

def get_model_path(model_family, scale):
    """Get the path to a model file, ensuring directories exist"""
    # Normalize model family name to lowercase for consistency
    model_family = model_family.lower()
    
    # Check if model family exists in our configurations
    if model_family not in SuperResolutionModel.MODEL_CONFIGS:
        print(f"Error: Unknown model family '{model_family}'")
        return None
    
    # Get configuration for this model
    config = SuperResolutionModel.MODEL_CONFIGS[model_family]
    
    # Convert scale to integer for comparison
    scale_int = int(scale.replace('x', ''))
    
    # Check if scale is supported for this model
    if scale_int not in config["supported_scales"]:
        print(f"Error: Scale {scale} not supported for {model_family}")
        return None
    
    # Get model filename from pattern
    model_filename = config["filename_pattern"].format(scale=scale_int)
    
    # Create family-specific subdirectory
    models_dir = os.path.join(get_models_root_dir(), model_family.upper())
    os.makedirs(models_dir, exist_ok=True)
    
    # Return the full path to the model file
    return os.path.join(models_dir, model_filename)

def download_model_file(url, destination):
    """Download a model file with progress indication"""
    # Skip if model already exists
    if os.path.exists(destination):
        print(f"Model already exists: {os.path.basename(destination)}")
        return True
    
    print(f"Downloading {os.path.basename(destination)}...")
    try:
        # Download with progress indication for large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(destination, 'wb') as f:
                if total_size > 1024*1024:  # Show progress for files > 1MB
                    downloaded = 0
                    start_time = time.time()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Show progress every 1MB
                            if downloaded % (1024*1024) < 8192:
                                percent = 100 * downloaded / total_size
                                elapsed = time.time() - start_time
                                speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                                print(f"  {percent:.1f}% ({downloaded/(1024*1024):.1f}MB) - {speed:.1f}MB/s")
                else:
                    # For small files, just download without progress
                    shutil.copyfileobj(r.raw, f)
        print(f"Downloaded {os.path.basename(destination)} successfully!")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Remove partial file if download failed
        if os.path.exists(destination):
            os.remove(destination)
        return False

def download_model(model_family, scale):
    """Download a specific SR model on-demand if it doesn't exist"""
    # Normalize model family name to lowercase for consistency
    model_family_norm = model_family.lower()
    
    # Check if valid model family
    if model_family_norm not in SuperResolutionModel.MODEL_CONFIGS:
        print(f"Error: Unknown model family '{model_family}'")
        return None
    
    config = SuperResolutionModel.MODEL_CONFIGS[model_family_norm]
    
    # Build model download URL map if we don't have a static one
    model_urls = {
        "fsrcnn": {
            "base_url": "https://raw.githubusercontent.com/ryanontheinside/FSRCNN_Tensorflow/master/models/",
        },
        "fsrcnn-small": {
            "base_url": "https://raw.githubusercontent.com/ryanontheinside/FSRCNN_Tensorflow/master/models/",
        },
        "edsr": {
            "base_url": "https://raw.githubusercontent.com/ryanontheinside/EDSR_Tensorflow/master/models/",
        },
        "espcn": {
            "base_url": "https://raw.githubusercontent.com/ryanontheinside/TF-ESPCN/master/export/",
        },
        "lapsrn": {
            "base_url": "https://raw.githubusercontent.com/ryanontheinside/TF-LapSRN/master/export/",
        },
        "vdsr": {
            "base_url": "https://raw.githubusercontent.com/ryanontheinside/TF-VDSR/master/export/",
        }
    }
    
    # Check if we have a download URL for this model
    if model_family_norm not in model_urls:
        print(f"Error: No download URL for model family '{model_family}'")
        return None
    
    # Get the model path
    model_path = get_model_path(model_family, scale)
    if not model_path:
        return None
    
    # If model already exists, return its path
    if os.path.exists(model_path):
        return model_path
    
    # Extract the scale number from the format
    scale_int = int(scale.replace('x', ''))
    
    # Get the base URL and create the download URL
    base_url = model_urls[model_family_norm]["base_url"]
    filename = config["filename_pattern"].format(scale=scale_int)
    url = base_url + filename
    
    if download_model_file(url, model_path):
        return model_path
    else:
        return None 