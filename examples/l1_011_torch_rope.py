"""Torch baseline for L1/011 Llama3 RoPE frequency scaling."""
import torch


@torch.no_grad()
def run(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scaling: float,
) -> torch.Tensor:
    batch_size = position_ids.shape[0]
    inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * attention_scaling
    sin = emb.sin() * attention_scaling
    return torch.stack([cos, sin], dim=-1).to(dtype=torch.bfloat16)
