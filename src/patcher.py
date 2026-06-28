import torch
import torch.nn as nn
from src.layers import QuantizedLinear

def apply_runtime_system_patch(module: nn.Module) -> int:
    """
    Recursively traverses the live model execution graph and hot-swaps
    standard nn.Linear layers for custom optimized QuantizedLinear layers.
    """
    patched_count = 0

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # STRATEGIC ALIGNMENT: Pass the entire child module directly into the constructor
            quantized_layer = QuantizedLinear(child)

            # Reassign the layer inside the parent module namespace structure
            setattr(module, name, quantized_layer)
            patched_count += 1
        else:
            # Continue scanning deeper down the submodule tree structures
            patched_count += apply_runtime_system_patch(child)

    return patched_count
