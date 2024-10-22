import torch


def get_device():
    return torch.device(
        "cuda"
        if torch.cuda.is_available()  # If GPU is to be used
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
