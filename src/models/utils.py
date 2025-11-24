import torch

try:
    import intel_extension_for_pytorch as ipex

    IPEX_AVAILABLE = True
except (ImportError, OSError):
    IPEX_AVAILABLE = False


def get_device():
    if IPEX_AVAILABLE and hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"Intel Arc GPU detected: {torch.xpu.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU detected)")
        return device


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
