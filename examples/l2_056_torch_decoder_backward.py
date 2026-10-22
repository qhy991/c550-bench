"""Torch baseline for L2/056 complete decoder-layer backward."""
import math
import torch
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.no_grad()
def run(
    grad_output: torch.Tensor,
    residual: torch.Tensor,
    attn_input: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    query_states_rotated: torch.Tensor,
    key_states_rotated: torch.Tensor,
    key_states_repeated: torch.Tensor,
    value_states_repeated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    residual2: torch.Tensor,
    ffn_input: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    silu_up: torch.Tensor,
    swiglu_output: torch.Tensor,
    input_ln_weight: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    o_weight: torch.Tensor,
    post_attn_ln_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    variance1: torch.Tensor,
    variance2: torch.Tensor,
    hidden_states_normalized1: torch.Tensor,
    hidden_states_normalized2: torch.Tensor,
    eps: float,
):
    batch_size, seq_len, hidden_size = grad_output.shape
    num_heads = 32
    num_kv_heads = 8
    head_dim = 160
    intermediate_size = 14336
    num_key_value_groups = num_heads // num_kv_heads

    grad_residual2 = grad_output
    grad_ffn_output = grad_output
    grad_swiglu_output = F.linear(grad_ffn_output, down_weight.t())
    grad_down_weight = grad_ffn_output.reshape(-1, hidden_size).t() @ swiglu_output.reshape(-1, intermediate_size)
    grad_gate = grad_swiglu_output * silu_up
    sigmoid_up = torch.sigmoid(up.to(torch.float32))
    grad_silu = sigmoid_up * (1.0 + up.to(torch.float32) * (1.0 - sigmoid_up))
    grad_up = grad_swiglu_output * gate * grad_silu.to(up.dtype)
    grad_ffn_input_gate = F.linear(grad_gate, gate_weight.t())
    grad_ffn_input_up = F.linear(grad_up, up_weight.t())
    grad_ffn_input = grad_ffn_input_gate + grad_ffn_input_up
    grad_gate_weight = grad_gate.reshape(-1, intermediate_size).t() @ ffn_input.reshape(-1, hidden_size)
    grad_up_weight = grad_up.reshape(-1, intermediate_size).t() @ ffn_input.reshape(-1, hidden_size)
    grad_ffn_input_fp32 = grad_ffn_input.to(torch.float32)
    grad_post_attn_ln_weight = (grad_ffn_input_fp32 * hidden_states_normalized2).sum(dim=[0, 1])
    N = hidden_size
    rsqrt_var2 = torch.rsqrt(variance2 + eps)
    grad_normalized2 = grad_ffn_input_fp32 * post_attn_ln_weight.to(torch.float32)
    grad_hidden_states2 = grad_normalized2 * rsqrt_var2
    grad_var2 = -0.5 * (grad_normalized2 * residual2.to(torch.float32)).sum(dim=-1, keepdim=True) * rsqrt_var2.pow(3)
    grad_hidden_states2 = grad_hidden_states2 + (2.0 / N) * residual2.to(torch.float32) * grad_var2
    grad_hidden_states2 = grad_hidden_states2.to(residual2.dtype)
    grad_hidden_states_attn = grad_residual2 + grad_hidden_states2
    grad_residual1 = grad_hidden_states_attn
    grad_attn_output_proj = grad_hidden_states_attn
    grad_attn_output = F.linear(grad_attn_output_proj, o_weight.t())
    grad_o_weight = grad_attn_output_proj.reshape(-1, hidden_size).t() @ attn_output.reshape(-1, num_heads * head_dim)
    grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    grad_attn_weights = torch.matmul(grad_attn_output, value_states_repeated.transpose(2, 3))
    grad_value_states_repeated = torch.matmul(attn_weights.transpose(2, 3), grad_attn_output)
    grad_attn_weights_fp32 = grad_attn_weights.to(torch.float32)
    attn_weights_fp32 = attn_weights.to(torch.float32)
    grad_attn_logits = attn_weights_fp32 * (grad_attn_weights_fp32 - (grad_attn_weights_fp32 * attn_weights_fp32).sum(dim=-1, keepdim=True))
    grad_attn_logits = (grad_attn_logits / math.sqrt(head_dim)).to(query_states_rotated.dtype)
    grad_query_states_rotated = torch.matmul(grad_attn_logits, key_states_repeated)
    grad_key_states_repeated = torch.matmul(grad_attn_logits.transpose(2, 3), query_states_rotated)
    grad_key_states_repeated = grad_key_states_repeated.reshape(batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim)
    grad_key_states_rotated = grad_key_states_repeated.sum(dim=2)
    grad_value_states_repeated = grad_value_states_repeated.reshape(batch_size, num_kv_heads, num_key_value_groups, seq_len, head_dim)
    grad_value_states = grad_value_states_repeated.sum(dim=2)
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    grad_query_states = (grad_query_states_rotated * cos_expanded) + (rotate_half(grad_query_states_rotated) * (-sin_expanded))
    grad_key_states = (grad_key_states_rotated * cos_expanded) + (rotate_half(grad_key_states_rotated) * (-sin_expanded))
    grad_query_states = grad_query_states.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)
    grad_key_states = grad_key_states.transpose(1, 2).reshape(batch_size, seq_len, num_kv_heads * head_dim)
    grad_value_states = grad_value_states.transpose(1, 2).reshape(batch_size, seq_len, num_kv_heads * head_dim)
    grad_attn_input_q = F.linear(grad_query_states, q_weight.t())
    grad_attn_input_k = F.linear(grad_key_states, k_weight.t())
    grad_attn_input_v = F.linear(grad_value_states, v_weight.t())
    grad_attn_input = grad_attn_input_q + grad_attn_input_k + grad_attn_input_v
    grad_q_weight = grad_query_states.reshape(-1, num_heads * head_dim).t() @ attn_input.reshape(-1, hidden_size)
    grad_k_weight = grad_key_states.reshape(-1, num_kv_heads * head_dim).t() @ attn_input.reshape(-1, hidden_size)
    grad_v_weight = grad_value_states.reshape(-1, num_kv_heads * head_dim).t() @ attn_input.reshape(-1, hidden_size)
    grad_attn_input_fp32 = grad_attn_input.to(torch.float32)
    grad_input_ln_weight = (grad_attn_input_fp32 * hidden_states_normalized1).sum(dim=[0, 1])
    rsqrt_var1 = torch.rsqrt(variance1 + eps)
    grad_normalized1 = grad_attn_input_fp32 * input_ln_weight.to(torch.float32)
    grad_hidden_states1 = grad_normalized1 * rsqrt_var1
    grad_var1 = -0.5 * (grad_normalized1 * residual.to(torch.float32)).sum(dim=-1, keepdim=True) * rsqrt_var1.pow(3)
    grad_hidden_states1 = grad_hidden_states1 + (2.0 / N) * residual.to(torch.float32) * grad_var1
    grad_hidden_states1 = grad_hidden_states1.to(residual.dtype)
    grad_input = grad_residual1 + grad_hidden_states1
    return (
        grad_input, grad_input_ln_weight,
        grad_q_weight, grad_k_weight, grad_v_weight, grad_o_weight,
        grad_post_attn_ln_weight, grad_gate_weight, grad_up_weight, grad_down_weight,
    )
