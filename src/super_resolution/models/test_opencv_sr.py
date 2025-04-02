import cv2
import sys
import torch

def test_opencv_version():
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Python Version: {sys.version}")
    print(f"CUDA available (torch): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (torch): {torch.version.cuda}")

def test_cuda_availability():
    print("\nTesting OpenCV CUDA Support:")
    
    # Check if CUDA module exists
    print(f"cv2.cuda exists: {hasattr(cv2, 'cuda')}")
    if hasattr(cv2, 'cuda'):
        print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
    
    # Check DNN CUDA backend availability
    print("\nTesting DNN CUDA Backend:")
    if hasattr(cv2.dnn, "getAvailableBackends"):
        backends = cv2.dnn.getAvailableBackends()
        print(f"Available backends: {backends}")
        print(f"Has CUDA backend: {cv2.dnn.DNN_BACKEND_CUDA in backends}")
    else:
        print("cv2.dnn.getAvailableBackends not available")
        
    if hasattr(cv2.dnn, "getAvailableTargets"):
        try:
            targets = cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_CUDA)
            print(f"Available CUDA targets: {targets}")
        except Exception as e:
            print(f"Error getting CUDA targets: {e}")
    else:
        print("cv2.dnn.getAvailableTargets not available")

def test_dnn_superres_availability():
    print("\nTesting DNN SuperRes availability:")
    print(f"cv2.dnn_superres exists: {hasattr(cv2, 'dnn_superres')}")
    
    if hasattr(cv2, 'dnn_superres'):
        print("\nAvailable attributes in cv2.dnn_superres:")
        for attr in dir(cv2.dnn_superres):
            if not attr.startswith('__'):
                print(f"- {attr}")

def test_create_methods():
    print("\nTesting creation methods:")
    
    methods = [
        ("DnnSuperResImpl()", lambda: cv2.dnn_superres.DnnSuperResImpl()),
        ("DnnSuperResImpl_create()", lambda: cv2.dnn_superres.DnnSuperResImpl_create())
    ]
    
    working_methods = []
    
    for method_name, method_func in methods:
        try:
            model = method_func()
            print(f"✓ {method_name} - SUCCESS")
            working_methods.append((method_name, model))
        except Exception as e:
            print(f"✗ {method_name} - FAILED: {str(e)}")
    
    return working_methods

def test_model_methods(working_methods):
    if not working_methods:
        print("\nNo working creation methods found!")
        return
        
    print("\nTesting model methods and CUDA setup on working instances:")
    for method_name, model in working_methods:
        print(f"\nTesting {method_name}:")
        try:
            print("Basic methods:")
            print("- hasattr readModel:", hasattr(model, 'readModel'))
            print("- hasattr setModel:", hasattr(model, 'setModel'))
            print("- hasattr upsample:", hasattr(model, 'upsample'))
            
            print("\nCUDA-related methods:")
            print("- hasattr setPreferableBackend:", hasattr(model, 'setPreferableBackend'))
            if hasattr(model, 'setPreferableBackend'):
                try:
                    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    print("  ✓ Successfully set CUDA backend")
                except Exception as e:
                    print(f"  ✗ Failed to set CUDA backend: {e}")
                    
            print("- hasattr setPreferableTarget:", hasattr(model, 'setPreferableTarget'))
            if hasattr(model, 'setPreferableTarget'):
                try:
                    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("  ✓ Successfully set CUDA target")
                except Exception as e:
                    print(f"  ✗ Failed to set CUDA target: {e}")
                    
        except Exception as e:
            print(f"Error testing methods: {str(e)}")

if __name__ == "__main__":
    print("OpenCV Super Resolution CUDA Test Script")
    print("=" * 40)
    
    test_opencv_version()
    test_cuda_availability()
    test_dnn_superres_availability()
    working_methods = test_create_methods()
    test_model_methods(working_methods) 