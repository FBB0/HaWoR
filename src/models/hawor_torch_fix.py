"""
Fix for PyTorch 2.8 weights_only security changes
This module patches torch.load to work with HaWoR checkpoints
"""

import torch
import functools

# Store the original torch.load function
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """
    Patched version of torch.load that sets weights_only=False for HaWoR checkpoints
    This is safe since we trust the HaWoR model checkpoints from the official repository
    """
    # Set weights_only=False for model loading (trusted checkpoints)
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

def apply_torch_fix():
    """Apply the torch.load fix for HaWoR compatibility"""
    torch.load = patched_torch_load
    print("✅ Applied PyTorch 2.8 compatibility fix for HaWoR model loading")

def restore_torch_load():
    """Restore the original torch.load function"""
    torch.load = _original_torch_load
    print("✅ Restored original torch.load function")

# Context manager for safe patching
class TorchLoadFix:
    """Context manager to temporarily patch torch.load for HaWoR model loading"""

    def __enter__(self):
        apply_torch_fix()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        restore_torch_load()

# Auto-apply fix when imported (for convenience)
if __name__ != "__main__":
    apply_torch_fix()