import torch
import torch.nn as nn

class QuantizedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # 1. Compress weights immediately to int8 scale
        weight = original_linear.weight.data
        scale = weight.abs().max() / 127.0
        scale = torch.clamp(scale, min=1e-5)

        q_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)

        # 2. Register buffers
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)

        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # STRATEGIC FIX: Force incoming runtime tensors onto the weight buffer's active device allocation
        if x.device != self.q_weight.device:
            x = x.to(self.q_weight.device)

        # Dequantize parameters back into host inference precision dynamically
        dequantized_weight = self.q_weight.to(x.dtype) * self.scale.to(x.dtype)

        if self.bias is not None:
            return torch.nn.functional.linear(x, dequantized_weight, self.bias.to(x.dtype))
        return torch.nn.functional.linear(x, dequantized_weight)
