"""MetaX PyTorch GEMM candidate for L1/003; C550 device qualification pending."""
import torch


@torch.no_grad()
def run(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # All original workloads retain the entire sequence, without a slicing input.
    return torch.matmul(hidden_states, weight.t())
