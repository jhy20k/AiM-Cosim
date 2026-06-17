# AiM-Cosim Docker Environment

AiM Simulator + Verilator timing & functional co-verification framework.

## Quick Start

```bash
# From project root
make docker-build    # Build Docker image
make docker-run      # Start container
make docker-shell    # Open shell in container
```

## Inside Container

```bash
cd /workspace
make build

# Run timing-only (Mode 1)
./build_mag32/extern/aim_simulator/ramulator2 -f src/configs/aim_timing_only.yaml -t <trace>

# Run timing+functional (Mode 2)
./build_mag32/extern/aim_simulator/ramulator2 -f src/configs/aim_rtl.yaml -t <trace>
```

The default Docker flow targets the validated GDDR6-AiM MAG32 RTL path. HBM-PIM
source files and configs under `src/` are future-work scaffolding and are not
part of the default smoke path.

## Volume Mounts

| Mount | Purpose |
|-------|---------|
| `..:/workspace` | Source code |
| `../extern/aim_simulator` -> `/workspace/extern/aim_simulator` | aim_simulator source |
| `../../CENT` -> `/workspace/extern/CENT` | CENT trace generation (volume mount, not submodule) |
| `aim-cosim-build` | Isolated build directory (WSL2 performance) |
| `aim-cosim-ccache` | Compiler cache |

## Environment

- **Verilator**: v5.024 (multi-stage copy from official image)
- **C++ Compiler**: GCC 12 (C++20 support)
- **Python**: 3.10 with numpy, matplotlib, pytest, torch, transformers, datasets
- **Build System**: CMake + Ninja

## CENT Integration

CENT is mounted as a Docker volume (not a submodule) because:
1. CENT already has aim_simulator as its own submodule (nested submodule conflict)
2. CENT is a Python project that generates ISR traces — AiM-Cosim only consumes trace files
3. Volume mount allows direct trace generation and consumption

```bash
# Generate traces with CENT
cd /workspace/extern/CENT/cent_simulation
python3 run_sim.py --model Llama2-7B --generate_trace

# Use traces in AiM-Cosim
cd /workspace
./build_mag32/extern/aim_simulator/ramulator2 -f src/configs/aim_rtl.yaml -t /workspace/extern/CENT/cent_simulation/<trace>
```
