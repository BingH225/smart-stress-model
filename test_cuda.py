import torch
import sys

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    try:
        device_count = torch.cuda.device_count()
        print(f"CUDA Device Count: {device_count}")
        for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            capability = torch.cuda.get_device_capability(i)
            print(f"  Compute Capability: {capability}")
        
        print("\nAttempting to move a tensor to CUDA and perform a basic operation...")
        device = torch.device("cuda")
        x = torch.tensor([1.0, 2.0]).to(device)
        y = torch.tensor([3.0, 4.0]).to(device)
        z = x + y
        print(f"Success! Result: {z}")
        
    except Exception as e:
        print(f"\nERROR calling CUDA: {e}")
        sys.exit(1)
else:
    print("\nCUDA is not available.")
