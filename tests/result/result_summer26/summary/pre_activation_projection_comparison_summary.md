# Actual pre-activation projection comparison

## Conditions

- Existing projection functional aggregate: real HF safetensors weight, random_bf16 activation, seed 42, output_stage=post_activation, result_command=RDAF16, CPU NumPy BF16 reference with AF/ReLU.
- Actual pre-activation projection: real HF safetensors weight, activation captured from the selected decoder layer projection input during a BF16 Hugging Face forward pass, output_stage=pre_activation, result_command=RDMAC16, no AF/ReLU.
- Actual pre-activation references: CPU NumPy BF16 GEMV and GPU torch BF16 projection output captured from the same forward pass.
- Channel layout: row_sharded, output_row_to_channel_and_bank.

## Comparison

| Projection | random_bf16 post within_1 | random_bf16 post max ULP | actual pre PIM-vs-CPU within_1 | actual pre PIM-vs-CPU max ULP | actual pre PIM-vs-GPU within_1 | actual pre PIM-vs-GPU max ULP |
|---|---:|---:|---:|---:|---:|---:|
| self_attn.q_proj | 2047/2048 | 3 | 2048/2048 | 1 | 2048/2048 | 0 |
| self_attn.k_proj | 512/512 | 1 | 512/512 | 1 | 512/512 | 0 |
| self_attn.v_proj | 512/512 | 1 | 512/512 | 1 | 512/512 | 0 |
| self_attn.o_proj | 2048/2048 | 1 | 2048/2048 | 1 | 2048/2048 | 1 |
| mlp.gate_proj | 8190/8192 | 2 | 8192/8192 | 1 | 8191/8192 | 8 |
| mlp.up_proj | 8191/8192 | 3 | 8192/8192 | 1 | 8192/8192 | 1 |
| mlp.down_proj | 2048/2048 | 1 | 2048/2048 | 1 | 2048/2048 | 1 |
