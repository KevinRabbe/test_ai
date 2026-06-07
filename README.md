# WB-MicroJEPA Starter

Whitebox MicroBrain Joint Embedding Predictive Architecture.

## Core rules

1. Deterministic first.
2. No hardcoded task solutions inside the model.
3. World rules are allowed only in worlds/verifiers/data generation.
4. The model must learn transition structure.
5. Every prediction exposes a trace.

See:

- `docs/PROJECT_RULES.md`
- `docs/ROADMAP.md`

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
- Verifier/data oracle outside the model
- Fixed probe set
- Diagnostics logger
- Evaluation grid
- Static HTML report
- Dense JEPA baseline scaffold

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.wb_microjepa.train --config configs/wb_microjepa_0a.yaml
```

## Evaluate

```bash
python -m src.wb_microjepa.evaluate runs/wb_microjepa_0a
```

## Generate report

```bash
python -m src.wb_microjepa.report runs/wb_microjepa_0a
```

Open:

```text
runs/wb_microjepa_0a/report.html
```

## Monitoring dashboard

```bash
streamlit run src/wb_microjepa/monitor_dashboard.py
```

## Compare runs

```bash
python -m src.wb_microjepa.compare_runs
```

## Analyze cases

```bash
python -m src.wb_microjepa.analyze_cases runs/wb_microjepa_0b_braingraph_train_numeric_ae
```

## Staged curriculum frontier

```powershell
./scripts/run_braingraph_staged_curriculum.ps1
```

## Soft curriculum frontier

```powershell
./scripts/run_braingraph_soft_curriculum.ps1
```

## Specialization loss frontier

```powershell
./scripts/run_braingraph_specialization_loss.ps1
```

## World modulation sweep

```powershell
./scripts/run_braingraph_worldmod_sweep.ps1
```

## Windows one-shot run

```powershell
./scripts/run_0a.ps1
```

## Output

Runs are saved under:

```text
runs/<run_name>/
├── config_resolved.json
├── metrics.csv
├── evaluation.csv
├── probe_traces.jsonl
├── microbrain_stats.csv
├── report.html
├── checkpoints/
└── plots/
```

## First research question

Does arranging parameters into many inspectable tiny predictive units produce more interpretable world-structure learning than a dense model with the same parameter budget?

## Important note

This is only version 0A. It is deliberately small. The current goal is not to be smart yet; the current goal is to make training, tracing, evaluation, and inspection work reliably before adding more brain modules.
