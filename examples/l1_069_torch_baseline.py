"""Independent Torch candidate for the exact L1/069 return-value ABI."""
import torch


@torch.no_grad()
def run(hidden_states, residual, weight, eps):
    summed = hidden_states + residual
    normalized = summed.float()
    inverse_rms = torch.rsqrt(torch.mean(normalized * normalized, dim=-1, keepdim=True) + eps)
    return (normalized * inverse_rms).to(torch.bfloat16) * weight
