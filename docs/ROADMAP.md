# WB-MicroJEPA Roadmap

## WB-MicroJEPA-0A

Goal: prove the training, whitebox trace, and monitoring loop.

Worlds:

- NumberLineWorld
- ModuloWorld

Modules:

- State encoder
- Action encoder
- World/context encoder
- Cortex micro-brain swarm
- Voting aggregator
- Decoder probe
- Verifier/data oracle outside model
- Diagnostics logger

Success:

- training runs end-to-end
- probe traces are saved
- activation heatmaps are generated
- embedding plots are generated
- dense baseline can be compared

## WB-MicroJEPA-0B

Goal: improve the first experiment quality.

Add:

- deterministic train/test evaluation
- dense baseline
- HTML run report
- specialization analysis
- stronger probe suite
- CLI smoke test

Success:

- compare swarm vs dense model
- detect dead/dominant/redundant micro-brains
- inspect whether modulo and number-line probes activate different units

## WB-MicroJEPA-1

Goal: add action selection.

Add:

- Basal Ganglia swarm
- goal state representation
- multi-step planning episodes
- STOP action

Example:

```text
start 12, target 18 -> +3, +3
```

## WB-MicroJEPA-2

Goal: add procedure compression.

Add:

- Cerebellum swarm
- primitive action sequence compression
- repeated action procedures

Example:

```text
+3,+3,+3,+3 -> repeat(+3,4)
```

## WB-MicroJEPA-3

Goal: add fast memory and consolidation.

Add:

- Hippocampus swarm
- episodic memory
- retrieval scoring
- replay queue
- consolidation into cortex/procedure modules

## WB-MicroJEPA-4

Goal: add routing and priority.

Add:

- Thalamus/router swarm
- Priority/surprise swarm
- replay priority
- module conflict tracking

## WB-MicroJEPA-5

Goal: multiple worlds and abstraction transfer.

Add:

- normal line
- modulo N
- clamp world
- negative line
- grid world
- logic world

## WB-MicroJEPA-6

Goal: unknowns and primitive algebra.

Add:

- missing target
- missing action
- missing start state
- candidate sets for ambiguous inverse tasks
