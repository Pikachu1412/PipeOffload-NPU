_NPU_AVAILABLE = None


def is_npu_available() -> bool:
    """Check if NPU (Ascend) is available in the current environment.

    This function checks whether the torch_npu package is installed and
    if there are available NPU devices.

    Returns:
        bool: True if NPU is available, False otherwise.
    """
    global _NPU_AVAILABLE

    if _NPU_AVAILABLE is not None:
        return _NPU_AVAILABLE

    try:
        import torch_npu
        _NPU_AVAILABLE = torch_npu.npu.is_available() if hasattr(
            torch_npu.npu, 'is_available') else torch_npu.npu.device_count() > 0
    except (ImportError, ModuleNotFoundError):
        _NPU_AVAILABLE = False

    return _NPU_AVAILABLE

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
