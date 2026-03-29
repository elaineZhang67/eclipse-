import torch


def resolve_device(preferred="auto"):
    preferred = str(preferred or "auto").strip().lower()

    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device 'cuda' but CUDA is not available.")
        return torch.device("cuda")

    if preferred == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("Requested device 'mps' but MPS is not available.")
        return torch.device("mps")

    if preferred == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {preferred}")


def default_torch_dtype(device):
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def model_load_kwargs(device):
    kwargs = {"torch_dtype": default_torch_dtype(device)}
    if device.type == "cuda":
        kwargs["device_map"] = "auto"
    return kwargs
