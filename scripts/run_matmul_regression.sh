#!/bin/bash
# =============================================================================
# Matmul Regression Test: P,Q,R Shape Variations
# Timing Identity (Mode 1 == Mode 2) + Execution Speed Comparison
#
# P (batch) = 1 (fixed, single-batch; multi-batch is TODO)
# Q (input dim) = 256, 512, 1024, 2048, 4096, 8192
# R (output dim) = 256, 512, 1024, 2048, 4096, 8192
#
# Usage: bash /workspace/scripts/run_matmul_regression.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-build_mag32}"
RAMULATOR="${RAMULATOR:-$PROJECT_DIR/$BUILD_DIR/extern/aim_simulator/ramulator2}"
CONFIG_M1="${CONFIG_M1:-$PROJECT_DIR/src/configs/aim_timing_only.yaml}"
CONFIG_M2="${CONFIG_M2:-$PROJECT_DIR/src/configs/aim_rtl.yaml}"
TRACE_GEN="$PROJECT_DIR/scripts/generate_matmul_trace.py"
EXTRACT="$PROJECT_DIR/scripts/extract_matmul_results.py"
RESULT_DIR="${RESULT_DIR:-$PROJECT_DIR/tests/result/matmul_regression}"
FUNC_DIR="$RESULT_DIR/functional"
PYTHON="${PYTHON:-/home/jhlee/miniconda3/bin/python3}"

mkdir -p "$RESULT_DIR"
mkdir -p "$FUNC_DIR"

P="${P:-1}"
Q_LIST="${Q_LIST:-256 512 1024 2048 4096 8192}"
R_LIST="${R_LIST:-256 512 1024 2048 4096 8192}"

PASS=0
FAIL=0
TOTAL=0

echo "================================================================"
echo "Matmul Regression Test (P=$P, batch=1)"
echo "Q: $Q_LIST"
echo "R: $R_LIST"
echo "================================================================"
echo ""
printf "%-20s %10s %12s %12s %10s %10s %8s\n" \
    "Shape(P×Q×R)" "ISR" "Mode1_cyc" "Mode2_cyc" "M1_time" "M2_time" "Result"
printf "%-20s %10s %12s %12s %10s %10s %8s\n" \
    "--------------------" "----------" "------------" "------------" "----------" "----------" "--------"

for Q in $Q_LIST; do
    for R in $R_LIST; do
        TOTAL=$((TOTAL+1))
        name="p${P}_q${Q}_r${R}"
        trace="$RESULT_DIR/${name}.trace"
        m1_out="$RESULT_DIR/${name}_mode1.txt"
        m2_out="$RESULT_DIR/${name}_mode2.txt"
        m2_cfg="$RESULT_DIR/${name}_aim_rtl.yaml"
        shape="${P}×${Q}×${R}"
        tiles_per_row=$((Q / 16))
        if [ "$tiles_per_row" -lt 128 ]; then
            tiles_per_row=128
        fi

        # Generate trace
        $PYTHON "$TRACE_GEN" --p $P --q $Q --r $R -o "$trace" > /dev/null 2>&1

        # Mode 2 memory manager must cover Q/16 tiles. Generate a per-shape
        # config so Q=4096/8192 cases do not reuse the default 128-tile config.
        sed -E \
            -e "s/(tiles_per_row:).*/\1 ${tiles_per_row}/" \
            -e "s#(result_log:).*#\1 \"${FUNC_DIR}/${name}_rtl_results.csv\"#" \
            "$CONFIG_M2" > "$m2_cfg"

        isr_count=$((P * R * 5 + 1))

        # Run Mode 1 with timing
        m1_start=$(date +%s%N)
        if ! "$RAMULATOR" -f "$CONFIG_M1" -t "$trace" > "$m1_out" 2>&1; then
            printf "%-20s %10d %12s %12s %10s %10s %8s\n" "$shape" "$isr_count" "ERR" "-" "-" "-" "SKIP"
            continue
        fi
        m1_end=$(date +%s%N)
        m1_ms=$(( (m1_end - m1_start) / 1000000 ))

        # Run Mode 2 with timing
        m2_start=$(date +%s%N)
        if ! "$RAMULATOR" -f "$m2_cfg" -t "$trace" > "$m2_out" 2>&1; then
            printf "%-20s %10d %12s %12s %10s %10s %8s\n" "$shape" "$isr_count" "-" "ERR" "-" "-" "SKIP"
            continue
        fi
        m2_end=$(date +%s%N)
        m2_ms=$(( (m2_end - m2_start) / 1000000 ))

        # Extract cycle counts
        m1_cycles=$(grep "memory_system_cycles:" "$m1_out" | head -1 | awk '{print $2}')
        m2_cycles=$(grep "memory_system_cycles:" "$m2_out" | head -1 | awk '{print $2}')

        if [ "$m1_cycles" = "$m2_cycles" ]; then
            result="PASS"
            PASS=$((PASS+1))
        else
            result="FAIL"
            FAIL=$((FAIL+1))
        fi

        # Format times (bc-free)
        if [ "$m1_ms" -ge 1000 ]; then
            m1_fmt="$((m1_ms / 1000)).$((m1_ms % 1000 / 100))s"
        else
            m1_fmt="${m1_ms}ms"
        fi
        if [ "$m2_ms" -ge 1000 ]; then
            m2_fmt="$((m2_ms / 1000)).$((m2_ms % 1000 / 100))s"
        else
            m2_fmt="${m2_ms}ms"
        fi

        printf "%-20s %10d %12s %12s %10s %10s %8s\n" \
            "$shape" "$isr_count" "$m1_cycles" "$m2_cycles" "$m1_fmt" "$m2_fmt" "$result"

        # Save functional results from Mode 2 CSV (channel 0)
        mode2_csv="${FUNC_DIR}/${name}_rtl_results_ch0.csv"
        if [ -f "$mode2_csv" ]; then
            func_out="$FUNC_DIR/${name}_functional.txt"
            $PYTHON "$EXTRACT" --csv "$mode2_csv" \
                --p $P --q $Q --r $R -o "$func_out" 2>/dev/null
        fi

        # Cleanup remaining CSV files from CWD
        rm -f rtl_results_ch*.csv 2>/dev/null || true
    done
done

echo ""
echo "================================================================"
echo "RESULTS: $PASS passed, $FAIL failed (of $TOTAL)"
echo "================================================================"
echo ""
echo "Note: P=1 (single-batch). Multi-batch (P>1) is TODO."

exit $FAIL
