import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_opencv_cuda():
    try:
        import cv2
        # Check if CUDA is already available
        logger.info("ComfyUI-Stream-Pack: Checking OpenCV CUDA availability...")
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            logger.info("OpenCV CUDA is installed and working")
            return True
        else:
            logger.warning("ComfyUI-Stream-Pack: OpenCV CUDA is not available or not working. Super Resolution node may not work")
            return False
    except ImportError:
        logger.warning("ComfyUI-Stream-Pack: OpenCV is not installed")
        return False

check_opencv_cuda()