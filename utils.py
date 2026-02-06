# utils.py
import torch

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")

