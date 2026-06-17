# =============================================================================
# AiM-Cosim - AiM Simulator + Verilator Co-verification Framework
# =============================================================================

.PHONY: all build build-debug clean clean-all docker-build docker-run docker-shell docker-stop docker-clean run-timing run-rtl compare run-llama-3.2-1B-decoder-output test help

BUILD_DIR ?= build_mag32
CMAKE_ARGS ?= -DACC_MAG_WIDTH=32
RAMULATOR ?= ./$(BUILD_DIR)/extern/aim_simulator/ramulator2
PYTHON ?= python3
MODEL_ID ?= meta-llama/Llama-3.2-1B-Instruct
MODEL_DIR ?=
MODEL_DIR_FLAG := $(if $(strip $(MODEL_DIR)),--model-dir $(MODEL_DIR) ,)
LLAMA321B_SEQ_LEN ?= 100
LLAMA321B_DATASET_SAMPLES ?= 64
LLAMA321B_OUT_DIR ?= tests/result/llama32_error_accumulation/summary/decoder_layer_output
LLAMA321B_SUFFIX ?= _v2
LLAMA321B_OFFLINE ?= 0
SEQ_LEN ?= $(LLAMA321B_SEQ_LEN)
DATASET_SAMPLES ?= $(LLAMA321B_DATASET_SAMPLES)
OUT_DIR ?= $(LLAMA321B_OUT_DIR)
SUFFIX ?= $(LLAMA321B_SUFFIX)
OFFLINE ?= $(LLAMA321B_OFFLINE)
OFFLINE_FLAG := $(if $(filter 1 true yes,$(OFFLINE)), --offline,)

# Default target
all: help

# =============================================================================
# Docker Commands
# =============================================================================

docker-build:
	@echo "Building Docker image..."
	cd docker && docker compose build aim-cosim

docker-run:
	@echo "Starting Docker container..."
	cd docker && docker compose up -d aim-cosim

docker-shell:
	@echo "Opening shell in Docker container..."
	cd docker && docker compose exec aim-cosim /bin/bash

docker-stop:
	@echo "Stopping Docker container..."
	cd docker && docker compose down

docker-clean:
	@echo "Removing Docker containers and volumes..."
	cd docker && docker compose down -v --rmi local

# =============================================================================
# Build Commands (run inside Docker or local environment)
# =============================================================================

build:
	@echo "Building AiM-Cosim..."
	cmake -S . -B $(BUILD_DIR) -GNinja $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) -j$$(nproc)

build-debug:
	@echo "Building AiM-Cosim (debug)..."
	cmake -S . -B $(BUILD_DIR) -GNinja $(CMAKE_ARGS) -DCMAKE_BUILD_TYPE=Debug
	cmake --build $(BUILD_DIR) -j$$(nproc)

$(RAMULATOR):
	$(MAKE) build

clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf *.vcd
	rm -rf *.log
	rm -rf *.result

clean-all:
	@echo "Cleaning all generated build/result artifacts..."
	bash scripts/cleanup_for_github.sh --apply

# =============================================================================
# Run Commands
# =============================================================================

# Mode 1: Timing-only (baseline)
run-timing:
	@echo "Running Mode 1 (timing-only)..."
	$(RAMULATOR) \
		-f src/configs/aim_timing_only.yaml \
		-t $(TRACE)

# Mode 2: Timing + Functional (co-verification)
run-rtl:
	@echo "Running Mode 2 (timing+functional)..."
	$(RAMULATOR) \
		-f src/configs/aim_rtl.yaml \
		-t $(TRACE)

# Compare results
compare:
	@echo "Comparing Mode 1 vs Mode 2..."
	python3 scripts/compare_results.py \
		--mode1 $(MODE1_LOG) \
		--mode2 $(MODE2_LOG)

# Llama3.2-1B full PIM-injected decoder-layer output delta run.
# Missing HF model/tokenizer files are downloaded into weight/ at runtime.
run-llama-3.2-1B-decoder-output: $(RAMULATOR)
	@echo "Running Llama3.2-1B full PIM decoder-layer output experiment..."
	$(PYTHON) scripts/run_wikitext_full_pim_decoder_outputs.py \
		--seq-len $(SEQ_LEN) \
		--dataset-samples $(DATASET_SAMPLES) \
		--model-id $(MODEL_ID) \
		$(MODEL_DIR_FLAG)--all-tokens \
		--resume$(OFFLINE_FLAG) \
		--summary-output $(OUT_DIR)/full_pim_decoder_layer_output_by_layer_wiki$(SUFFIX).csv \
		--elementwise-output $(OUT_DIR)/full_pim_decoder_layer_output_elementwise_wiki$(SUFFIX).csv \
		--metadata-output $(OUT_DIR)/full_pim_decoder_layer_output_wiki$(SUFFIX).metadata.json

# =============================================================================
# Test Commands
# =============================================================================

test:
	@echo "Running tests..."
	cd $(BUILD_DIR) && ctest --output-on-failure

# =============================================================================
# Help
# =============================================================================

help:
	@echo "AiM-Cosim - AiM Simulator + Verilator Co-verification Framework"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Start Docker container"
	@echo "  docker-shell   - Open shell in container"
	@echo "  docker-stop    - Stop Docker container"
	@echo "  docker-clean   - Remove containers and volumes"
	@echo ""
	@echo "Build Commands:"
	@echo "  build          - Build (Release)"
	@echo "  build-debug    - Build (Debug)"
	@echo "  clean          - Clean selected build dir ($(BUILD_DIR))"
	@echo "  clean-all      - Move generated artifacts out of the repo for GitHub"
	@echo ""
	@echo "Run Commands:"
	@echo "  run-timing TRACE=<path>    - Run Mode 1 (timing-only)"
	@echo "  run-rtl TRACE=<path>       - Run Mode 2 (timing+functional)"
	@echo "  compare MODE1_LOG=<path> MODE2_LOG=<path>"
	@echo "  run-llama-3.2-1B-decoder-output - Run WikiText Llama3.2 decoder-output delta experiment"
	@echo "                                      Set MODEL_ID=<hf/repo> for a different compatible model"
	@echo "                                      Omit MODEL_DIR for online/default cache"
	@echo "                                      Set MODEL_DIR=<path> to use a local model path"
	@echo "                                      Set OFFLINE=1 to require local model/dataset caches"
	@echo ""
	@echo "Test Commands:"
	@echo "  test           - Run tests"
