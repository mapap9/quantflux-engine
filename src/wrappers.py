import torch
import torch.nn as nn
from src.patcher import apply_runtime_system_patch
from torch.utils.checkpoint import checkpoint

class FluxParallelTransformerBlockWrapper(nn.Module):
    """
    Production-grade execution shell that dynamically captures any variadic keyword
    arguments passed by the host framework to enforce non-reentrant activation dropping.
    """
    def __init__(self, original_flux_block: nn.Module):
        super().__init__()
        self.block = original_flux_block

    def forward(self, *args, **kwargs):
        def _custom_forward(*inner_args, **inner_kwargs):
            return self.block(*inner_args, **inner_kwargs)
        return checkpoint(_custom_forward, *args, use_reentrant=False, **kwargs)


def optimize_production_flux_graph(flux_model: nn.Module) -> nn.Module:
    print("⏳ Stage 1: Running recursive out-of-core int8 weight compression...")
    swapped_layers = apply_runtime_system_patch(flux_model)
    print(f"✅ Swapped {swapped_layers} production linear projections to QuantizedLinear.")

    print("⏳ Stage 2: Deploying dual-stream and single-stream activation checkpointing wrappers...")

    # 1. Wrap the dual-stream transformer blocks
    if hasattr(flux_model, 'transformer_blocks'):
        for i, block in enumerate(flux_model.transformer_blocks):
            flux_model.transformer_blocks[i] = FluxParallelTransformerBlockWrapper(block)
        print(f"✅ Wrapped {len(flux_model.transformer_blocks)} dual-stream blocks.")

    # 2. Wrap the single-stream transformer blocks (The missing OOM culprit)
    if hasattr(flux_model, 'single_transformer_blocks'):
        for i, block in enumerate(flux_model.single_transformer_blocks):
            flux_model.single_transformer_blocks[i] = FluxParallelTransformerBlockWrapper(block)
        print(f"✅ Wrapped {len(flux_model.single_transformer_blocks)} single-stream blocks.")

    return flux_model
