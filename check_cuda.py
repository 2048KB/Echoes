import torch
import sys
import platform
import subprocess
import os

def get_gpu_info():
    try:
        if platform.system() == 'Windows':
            import wmi
            w = wmi.WMI()
            gpu_info = w.Win32_VideoController()
            return [gpu.Name for gpu in gpu_info]
    except:
        return []

def check_system():
    print("=== System Information ===")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    print("\n=== CUDA Information ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print("\nYour system is ready to use GPU acceleration!")
    else:
        print("\nCUDA is not available. Here's what you need to do:")
        
        gpus = get_gpu_info()
        if gpus:
            print("\nDetected GPUs:")
            for gpu in gpus:
                print(f"- {gpu}")
        else:
            print("\nNo NVIDIA GPUs detected")
        
        print("\nTo enable GPU acceleration, follow these steps:")
        print("1. Verify you have an NVIDIA GPU")
        print("2. Install NVIDIA GPU drivers from: https://www.nvidia.com/Download/index.aspx")
        print("3. Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads")
        print("4. Reinstall PyTorch with CUDA support using this command:")
        print("   pip uninstall torch")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    check_system() 