# WB-MicroJEPA Starter

Whitebox MicroBrain Joint Embedding Predictive Architecture.

## Core rules

1. Deterministic first.
2. No hardcoded task solutions inside the model.
3. World rules are allowed only in worlds/verifiers/data generation.
4. The model must learn transition structure.
5. Every prediction exposes a trace.

## First prototype

WB-MicroJEPA-0A:

- NumberLineWorld
- ModuloWorld
- State encoder
- Action encoder
- World/context encoder
- Cortex micro-brain swarm
- Confidence-weighted voting aggregator
- Decoder probe
- Verifier
- Fixed probe set
- Diagnostics logger

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.wb_microjepa.train --config configs/wb_microjepa_0a.yaml
```

## Output

Runs are saved under:

```text
runs/<run_name>/
├── config_resolved.json
├── metrics.csv
├── probe_traces.jsonl
├── microbrain_stats.csv
├── checkpoints/
└── plots/
```

## First research question

Does arranging parameters into many inspectable tiny predictive units produce more interpretable world-structure learning than a dense model with the same parameter budget?
