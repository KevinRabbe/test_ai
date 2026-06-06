# WB-MicroJEPA Project Rules

## 1. Deterministic first

Same seed, same config, same code, and same data generation path should produce the same run where PyTorch allows it.

Required:

- seed Python `random`
- seed NumPy
- seed PyTorch CPU/CUDA
- disable cuDNN benchmark mode
- store resolved config per run

## 2. No hardcoded task solutions inside the model

The learned model must not contain hand-written solution logic such as:

- if world is modulo, compute wrapped result directly
- if number line, use `state + action` inside the model
- if grid world, use hand-coded pathfinding as model output
- if logic world, use hardcoded truth-table answer as model output

The model may receive state/action/context embeddings and must learn transition structure from data.

## 3. World rules belong outside the model

Allowed hardcoded components:

- world simulator
- verifier/oracle
- data generator
- evaluation metrics
- baselines
- logging and visualization

Not allowed inside the learned model:

- answer formulas
- task-specific shortcuts
- manually assigned micro-brain roles
- manually forced routing decisions long-term

## 4. Whitebox by construction

Every trainable module should expose diagnostics.

Minimum trace fields:

- active micro-brains
- activation scores
- confidence scores
- vote vectors
- aggregation weights
- decoded prediction
- verifier result where available

A prediction without a trace is considered incomplete.

## 5. Learn mechanisms, not examples only

The goal is not only high training accuracy.

The system should be evaluated on:

- unseen ranges
- world switching
- boundary behavior
- specialization
- embedding geometry
- dense baseline comparison

## 6. Baselines are mandatory

At minimum:

- Dense JEPA-style baseline
- WB-MicroJEPA swarm model
- same parameter scale where possible
- same training data
- same probes

## 7. Every shortcut must be labeled

If a component has explicit rules, label it as one of:

- world rule
- verifier rule
- dataset rule
- baseline rule
- learned model behavior

This keeps the research clean.
