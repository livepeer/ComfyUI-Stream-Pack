class SuperResolutionModelLoader:
    """Node that loads a super resolution model and outputs a reference to it"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["FSRCNN", "FSRCNN-small", "EDSR", "ESPCN", "LapSRN", "VDSR"], {
                    "default": "FSRCNN",
                    "tooltip": "Select super resolution model type:\n"
                              "• FSRCNN: Good balance of quality and speed\n"
                              "• FSRCNN-small: Fastest, lower quality option\n"
                              "• ESPCN: Efficient for text and line art\n"
                              "• LapSRN: Better edge preservation\n"
                              "• EDSR: Highest quality, slower processing\n"
                              "• VDSR: Very Deep SR with sharp edge reconstruction\n"
                }),
                "scale_factor": (["2", "3", "4"], {
                    "default": "2",
                    "tooltip": "Upscaling factor:\n"
                              "• 2: Double resolution (e.g., 512x512 → 1024x1024)\n"
                              "• 3: Triple resolution (e.g., 512x512 → 1536x1536)\n"
                              "• 4: Quadruple resolution (e.g., 512x512 → 2048x2048)"
                }),
            }
        }

    RETURN_TYPES = ("SR_MODEL",)
    RETURN_NAMES = ("sr_model",)
    FUNCTION = "load_model"
    CATEGORY = "StreamPack/SuperResolution"

    def load_model(self, model_type, scale_factor):
        """Load the selected super resolution model and return a reference"""
        
        # Import unified model class
        from ..models import SuperResolutionModel
        
        # Import model download utilities
        from ..utils.model_utils import download_model
        
        # On-demand download the model file if it doesn't exist
        scale_key = f"x{scale_factor}"
        model_path = None
        
        # Download the model if needed
        model_path = download_model(model_type, scale_key)
        if not model_path:
            print(f"Warning: Could not download model {model_type} {scale_key}")
        
        # Create the model using our factory method
        model = SuperResolutionModel.create(model_type, int(scale_factor))
            
        # Load the model with the downloaded path if available
        if model_path:
            model.load_model(model_path)
        else:
            # Fall back to default paths
            model.load_model()
            
        return (model,)


class SuperResolutionUpscale:
    """Node that applies a super resolution model to an image"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sr_model": ("SR_MODEL",),
                "use_cuda": (["True", "False"], {
                    "default": "True",
                    "tooltip": "Whether to use CUDA for acceleration (requires CUDA-enabled OpenCV build)"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale_image"
    OUTPUT_NODE = True
    CATEGORY = "StreamPack/SuperResolution"

    def upscale_image(self, image, sr_model, use_cuda):
        """Upscale image using the provided super resolution model"""
        
        # Convert use_cuda string to boolean
        use_cuda = use_cuda == "True"
        
        # Upscale the image with the use_cuda parameter
        result = sr_model.upscale(image, use_cuda=use_cuda)
        
        return (result,) 