#!/usr/bin/env python3
"""Generate AiM matmul trace files with parameterized P, Q, R shapes.

Matmul: C[P×R] = A[P×Q] × B[Q×R]

AiM mapping:
  - P (batch): number of output vectors (rows of C), P=1 for single-batch
  - Q (input dim): input vector length = Q, mapped to opsize = Q/16 tiles
  - R (output dim): number of output rows = R, each row needs WR_BIAS+WR_GB+MAC_ABK+AF+RD_AF

ISR sequence per output row:
  WR_BIAS <opsize> <channel_mask>
  WR_GB <opsize> <row> <channel_mask>
  MAC_ABK <opsize> <channel_mask> <row>
  AF <channel_mask>
  RD_AF <opsize> <channel_mask>

Usage:
  python generate_matmul_trace.py --p 1 --q 1024 --r 1024 -o trace.trace
"""

import argparse
import os


def generate_trace(p, q, r, output_path, channel_mask="0xffffffff"):
    opsize = q // 16  # tiles per MAC
    assert q % 16 == 0, f"Q must be multiple of 16, got {q}"

    with open(output_path, 'w') as f:
        f.write(f"# AiM matmul trace: P={p}, Q={q}, R={r}\n")
        f.write(f"# C[{p}x{r}] = A[{p}x{q}] x B[{q}x{r}]\n")
        f.write(f"# opsize={opsize} (Q/16), channel_mask={channel_mask}\n")
        f.write(f"# ISR per row: WR_BIAS + WR_GB + MAC_ABK + AF + RD_AF\n")
        f.write(f"# Total ISR: {p * r * 5 + 1} ({p}×{r}×5 + EOC)\n")
        f.write(f"#\n")

        for batch in range(p):
            for row in range(r):
                f.write(f"AiM WR_BIAS {opsize} {channel_mask}\n")
                f.write(f"AiM WR_GB {opsize} 0 {channel_mask}\n")
                f.write(f"AiM MAC_ABK {opsize} {channel_mask} {row}\n")
                f.write(f"AiM AF {channel_mask}\n")
                f.write(f"AiM RD_AF {opsize} {channel_mask}\n")

        f.write("AiM ISR_EOC\n")

    total_isr = p * r * 5 + 1
    print(f"Generated: {output_path}")
    print(f"  Shape: P={p} Q={q} R={r}")
    print(f"  opsize={opsize}, ISR={total_isr}")


def main():
    parser = argparse.ArgumentParser(description='Generate AiM matmul trace')
    parser.add_argument('--p', type=int, default=1, help='Batch size (rows of A)')
    parser.add_argument('--q', type=int, default=1024, help='Input dimension (cols of A = rows of B)')
    parser.add_argument('--r', type=int, default=1024, help='Output dimension (cols of B)')
    parser.add_argument('--channel-mask', default='0xffffffff', help='Channel mask')
    parser.add_argument('-o', '--output', required=True, help='Output trace file')
    args = parser.parse_args()

    generate_trace(args.p, args.q, args.r, args.output, args.channel_mask)


if __name__ == '__main__':
    main()
