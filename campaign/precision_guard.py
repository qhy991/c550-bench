"""Task-owned checks for previously demonstrated precision violations."""
import json


def validate(root, task, receipts):
    for name in receipts:
        r = json.loads((root / name).read_text())
        d = json.loads((root / r['schedule']).read_text())
        buffers = {b['name']: b for b in d['buffers']}
        mma = [o for o in d['operations'] if o['kind'] == 'mma']

        if task == 'L1/069_rms_norm':
            for op in d['operations']:
                if op['kind'] == 'reduce' and any(
                    buffers[n]['dtype'] != 'fp32'
                    for n in op['reads'] + op['writes']
                ):
                    raise ValueError('RMSNorm floating reduction must remain FP32')

        if task == 'L1/048_fused_gate_up_projection_with_swiglu':
            for op in mma:
                if (any(buffers[n]['dtype'] != 'bf16' for n in op['reads'][:2])
                        or op['parameters']['accumulator'] != 'fp32'):
                    raise ValueError(
                        'GateUp keeps BF16 projections with FP32 accumulation'
                    )

        if task == 'L1/077_whisper_decoder_output_projection':
            for op in mma:
                if (any(buffers[n]['dtype'] != 'fp16' for n in op['reads'][:2])
                        or op['parameters']['accumulator'] != 'fp32'):
                    raise ValueError(
                        'Whisper output projection requires FP16 operands and FP32 accumulation'
                    )
            for op in d['operations']:
                if op['kind'] == 'reduce' and any(
                    buffers[n]['dtype'] != 'fp32'
                    for n in op['reads'] + op['writes']
                ):
                    raise ValueError(
                        'Whisper output projection contraction reduction must retain FP32'
                    )

    return {
        'status': 'passed',
        'scope': (
            'demonstrated Task-owned precision rules; '
            'final wrapper and rounding-chain review remains required'
        ),
        'receipts': receipts,
    }
