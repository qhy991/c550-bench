"""Torch baseline for L1/048 dual projection + GELU-tanh gate."""
import math
import torch


@torch.no_grad()
def run(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
) -> torch.Tensor:
    gate_output = torch.matmul(x, gate_proj.t())
    up_output = torch.matmul(x, up_proj.t())
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    gate_f = gate_output.to(torch.float32)
    inner = sqrt_2_over_pi * (gate_f + 0.044715 * gate_f.pow(3))
    activated = 0.5 * gate_f * (1.0 + torch.tanh(inner))
    return activated.to(gate_output.dtype) * up_output
