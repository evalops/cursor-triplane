# Cursor Triplane RL Stack

This repository implements a Cursor-style tri-plane composed of an FP8 MoE trainer, a Ray-based inference orchestrator, and Firecracker-isolated environment servers that expose a unified tool API.

## Architecture Overview

- **Environment Fleet**: `envd/server.py` provides the gRPC tool surface (read/edit/search/lint/exec) and optional semantic search backed by Qdrant; Firecracker launch scripts in `scripts/firecracker/` create snapshot-based microVMs.
- **Inference**: `inference/serve.py` bootstraps Ray actors (controller, samplers, env clients) to execute parallel tool plans with straggler mitigation and speculative rollouts.
- **Trainer**: `trainer/` contains a PPO loop over a lightweight MoE transformer policy, reward shaping utilities, and data helpers suitable for integration with DeepSpeed/Megatron FP8 stacks.

## Getting Started

1. **Install dependencies** (Python ≥3.10):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Generate gRPC bindings**:
   ```bash
   ./scripts/gen_protos.sh
   ```
3. **Run proto-environment server** (inside Firecracker VM or locally):
   ```bash
   python -m envd
   ```
4. **Demo rollouts** (requires Ray runtime):
   ```bash
   python experiments/run_inference_demo.py
   ```
5. **Dry-run trainer** using sample rollouts:
   ```bash
   ./experiments/run_training.sh
   ```

## Firecracker Workflow

1. Build the base image and import into Ignite:
   ```bash
   ./scripts/firecracker/build_base.sh
   ```
2. Create a snapshot template per task family:
   ```bash
   ./scripts/firecracker/create_template.sh
   ```
3. Launch disposable microVMs for batched rollouts:
   ```bash
   COUNT=50 SEED_REPO=./seed_repo ./scripts/firecracker/launch_envs.sh
   ```

## Training Notes

- `trainer/train.py` uses a configurable PPO loop with checkpoint emission every 50 steps.
- Swap the placeholder `PolicyModel` with a Megatron-Core/DeepSpeed MoE transformer initialized with TransformerEngine FP8 kernels to enable MXFP8 microscaling.
- Rollout logs (JSONL) should include `obs`, `actions`, `logprob`, `return`, `advantage`, and metric payloads; see `experiments/sample_rollouts.jsonl` for the expected schema.

## Roadmap

- Integrate semantic code search using a production-grade embedding model.
- Replace the stub model sampler with a vLLM or serve-hosted policy endpoint.
- Wire reward streaming to an external registry (e.g., ClickHouse + S3 checkpoint sync).
