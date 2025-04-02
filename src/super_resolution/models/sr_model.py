import os
import cv2
import torch
import numpy as np

class SuperResolutionModel:
    """Unified Super Resolution Model for all algorithm types"""
    
    # Configuration registry for all model types
    MODEL_CONFIGS = {
        "fsrcnn": {
            "algorithm": "fsrcnn",
            "repo_subdir": "FSRCNN_Tensorflow/models",
            "filename_pattern": "FSRCNN_x{scale}.pb",
            "supported_scales": [2, 3, 4],
        },
        "fsrcnn-small": {
            "algorithm": "fsrcnn",
            "repo_subdir": "FSRCNN_Tensorflow/models",
            "filename_pattern": "FSRCNN-small_x{scale}.pb",
            "supported_scales": [2, 3, 4],
        },
        "edsr": {
            "algorithm": "edsr",
            "repo_subdir": "EDSR_Tensorflow/models",
            "filename_pattern": "EDSR_x{scale}.pb",
            "supported_scales": [2, 3, 4],
        },
        "espcn": {
            "algorithm": "espcn",
            "repo_subdir": "TF-ESPCN/export",
            "filename_pattern": "ESPCN_x{scale}.pb",
            "supported_scales": [2, 3, 4],
        },
        "lapsrn": {
            "algorithm": "lapsrn",
            "repo_subdir": "TF-LapSRN/export",
            "filename_pattern": "LapSRN_x{scale}.pb",
            "supported_scales": [2, 3, 4],
        },
        "vdsr": {
            "algorithm": "vdsr",
            "repo_subdir": "TF-VDSR/export",
            "filename_pattern": "VDSR_x{scale}.pb",
            "supported_scales": [2, 3, 4],
        }
    }
    
    @classmethod
    def is_cuda_available(cls):
        """Check if CUDA is available in OpenCV"""
        # Check if CUDA module exists in OpenCV
        if not hasattr(cv2, "cuda") or cv2.cuda.getCudaEnabledDeviceCount() == 0:
            return False
        
        # Check if DNN module has CUDA support
        try:
            # Try to get available backends if the method exists
            if hasattr(cv2.dnn, "getAvailableBackends"):
                backends = cv2.dnn.getAvailableBackends()
                if cv2.dnn.DNN_BACKEND_CUDA not in backends:
                    return False
                    
            # Check for CUDA targets if method exists
            if hasattr(cv2.dnn, "getAvailableTargets"):
                targets = cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)
                if len(targets) == 0:
                    return False
        except Exception:
            return False
            
        return True
    
    @classmethod
    def create(cls, model_type, scale_factor, **kwargs):
        """Factory method to create a super resolution model by type"""
        model_type = model_type.lower()
        
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return cls(model_type, scale_factor, **kwargs)
    
    def __init__(self, model_name, scale_factor, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name.lower()
        self.scale_factor = int(scale_factor)
        self.device = device
        self.model = None
        self.loaded = False
        self.using_cuda = False
        
        # Get configuration for this model
        self.config = self.MODEL_CONFIGS.get(self.model_name)
        if not self.config:
            raise ValueError(f"No configuration found for model: {model_name}")
            
        # Validate scale factor
        if self.scale_factor not in self.config["supported_scales"]:
            raise ValueError(f"Scale factor {scale_factor} not supported for {model_name}. "
                           f"Supported scales: {self.config['supported_scales']}")
    
    def load_model(self, model_path=None):
        """Load the super resolution model"""
        if model_path is None:
            # Try to find model in standard locations
            model_dir = self.model_name.upper()
            
            # Search paths from highest to lowest priority
            search_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../downloaded_models", model_dir),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../models"),
                os.path.expanduser("~/.cache/comfyui/custom_nodes/ComfyUI_SuperResolution/models"),
            ]
            
            # Add repo-specific path if available
            if self.config["repo_subdir"]:
                search_paths.append(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../", self.config["repo_subdir"])
                )
            
            for path in search_paths:
                filename = self.config["filename_pattern"].format(scale=self.scale_factor)
                candidate = os.path.join(path, filename)
                if os.path.exists(candidate):
                    model_path = candidate
                    break
                    
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find {self.model_name} x{self.scale_factor} model file")
            
        # Load model using OpenCV DNN Super Resolution with the correct method
        try:
            # Try different methods to create the DNN SuperRes object for cross-platform compatibility
            # This handles differences between Windows and Linux OpenCV builds
            try:
                # Method 1: Windows-style API (most common)
                self.model = cv2.dnn_superres.DnnSuperResImpl_create()
            except AttributeError:
                try:
                    # Method 2: More generic constructor if available
                    self.model = cv2.dnn_superres.SuperResolution_create()
                except AttributeError:
                    # Method 3: Direct class instantiation if available
                    self.model = cv2.dnn_superres.DnnSuperResImpl()
            
            self.model.readModel(model_path)
            self.model.setModel(self.config["algorithm"], self.scale_factor)
            
            # Check CUDA availability during model loading
            # Note: We don't raise errors here, as that will be handled during upscale if use_cuda=True
            system_has_cuda = torch.cuda.is_available()
            opencv_has_cuda = self.is_cuda_available()
            
            if system_has_cuda and opencv_has_cuda:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print(f"Model loaded with CUDA backend for {self.model_name}")
                self.using_cuda = True
            else:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"Model loaded with CPU backend for {self.model_name}")
                self.using_cuda = False
                
            self.loaded = True
        except Exception as e:
            raise RuntimeError(f"Error with OpenCV DNN Super Resolution: {e}. Check that opencv-contrib-python is installed.")
            
        return self
    
    def upscale(self, image, use_cuda=True):
        """Upscale an image using the model"""
        if not self.loaded:
            self.load_model()
            
        # Check CUDA requirements if use_cuda is enabled
        if use_cuda:
            system_has_cuda = torch.cuda.is_available()
            opencv_has_cuda = self.is_cuda_available()
            
            if not system_has_cuda:
                raise RuntimeError("CUDA acceleration requested but CUDA is not available on your system. "
                                 "Either install CUDA or disable CUDA acceleration in the node settings.")
                                 
            if not opencv_has_cuda:
                raise RuntimeError("CUDA acceleration requested but your OpenCV build doesn't support CUDA. "
                                 "Install a CUDA-enabled OpenCV build (via conda: 'conda install -c conda-forge opencv cudatoolkit') "
                                 "or disable CUDA acceleration in the node settings.")
        
        # Convert to cv2 format if needed
        cv2_image = self.convert_comfy_to_cv2(image)
        
        # Process the image
        upscaled = self.model.upsample(cv2_image)
        
        # Convert back to original format
        return self.convert_cv2_to_comfy(upscaled, image)
    
    def convert_comfy_to_cv2(self, tensor):
        """Convert ComfyUI tensor (BHWC) to cv2 image (HWC)"""
        # Handle both tensor and numpy array inputs
        if isinstance(tensor, torch.Tensor):
            # Ensure tensor is on CPU and convert to numpy
            image = tensor.detach().cpu().numpy()
        else:
            image = tensor
            
        # Take first image if batch dimension exists
        if len(image.shape) == 4:
            image = image[0]
            
        # Convert float tensor [0,1] to uint8 [0,255]
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
            
        # Convert RGB to BGR for OpenCV
        if image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        return image
    
    def convert_cv2_to_comfy(self, image, original_tensor=None):
        """Convert cv2 image (HWC) back to ComfyUI tensor format (BHWC)"""
        # Convert BGR to RGB
        if image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Convert to float [0,1] if original was float
        if original_tensor is not None and (
            isinstance(original_tensor, torch.Tensor) and original_tensor.dtype == torch.float32 or
            isinstance(original_tensor, np.ndarray) and original_tensor.dtype == np.float32
        ):
            image = image.astype(np.float32) / 255.0
            
        # Add batch dimension if needed
        if original_tensor is not None and len(original_tensor.shape) == 4:
            image = np.expand_dims(image, axis=0)
            
        # Convert to torch tensor if original was tensor
        if isinstance(original_tensor, torch.Tensor):
            image = torch.from_numpy(image).to(original_tensor.device)
            
        return image
    
    def release(self):
        """Release resources"""
        self.model = None
        self.loaded = False
        
    def __del__(self):
        self.release() 