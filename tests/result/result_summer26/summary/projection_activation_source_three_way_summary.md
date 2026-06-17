# Projection activation source three-way comparison

## Conditions

- Common: Llama3.2-1B-Instruct, layer 0, real HF safetensors weight, row_sharded layout, output_stage=pre_activation, result_command=RDMAC16, no AF/ReLU.
- Random BF16: seed 42 standard-normal BF16 vector.
- Prompt-hash BF16: deterministic BF16 vector generated from `/home/jhlee/aimvsim_summer26_v2/result_summer26/artifacts/prompt_inputs_maxnew16.json` and seed 42.
- Actual hook: BF16 projection input captured from a Hugging Face forward pass on WikiText seq_len=16, token 15.
- Each cell is `within_1_ulp, max ULP`.

## Results

| Projection | Random PIM-vs-CPU | Random PIM-vs-GPU | Prompt-hash PIM-vs-CPU | Prompt-hash PIM-vs-GPU | Actual hook PIM-vs-CPU | Actual hook PIM-vs-GPU |
|---|---:|---:|---:|---:|---:|---:|
| self_attn.q_proj | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 0 |
| self_attn.k_proj | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 |
| self_attn.v_proj | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 | 512/512, max 1 | 512/512, max 0 |
| self_attn.o_proj | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 0 | 2048/2048, max 1 | 2048/2048, max 1 |
| mlp.gate_proj | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8191/8192, max 8 |
| mlp.up_proj | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 1 | 8192/8192, max 0 | 8192/8192, max 1 | 8192/8192, max 1 |
| mlp.down_proj | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 | 2048/2048, max 1 |
