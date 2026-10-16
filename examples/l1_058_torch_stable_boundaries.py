import torch


def run(topk_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, positions = torch.sort(topk_idx.reshape(-1), stable=True)
    thresholds = torch.arange(257, dtype=topk_idx.dtype, device=topk_idx.device)
    offsets = torch.searchsorted(values, thresholds, right=False)
    return positions.to(torch.int32), offsets.to(torch.int32)
