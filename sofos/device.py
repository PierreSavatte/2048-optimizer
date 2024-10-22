import torch


def get_device():
    device_name = (
        "cuda"
        if torch.cuda.is_available()  # If GPU is to be used
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using {device_name=}")
    return torch.device(device_name)
