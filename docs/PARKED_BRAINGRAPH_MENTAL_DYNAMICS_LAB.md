# Parked Future Idea: BrainGraph Mental Dynamics Lab

## Status

Parked for later. This is not part of the current WB-MicroJEPA training milestone.

Current active focus remains:

1. make BrainGraph learn clean transition structure
2. improve delta/action/world-context learning
3. build better monitoring for artificial region specialization
4. compare runs honestly

## Core idea

Use the whitebox BrainGraph as a visual simulator for artificial brain-like dynamics.

The goal is not diagnosis and not a claim that the model is a real human brain.

The goal is:

```text
Show how different control, attention, memory, routing, action-selection,
and error/surprise settings change activity inside an artificial brain-shaped graph.
```

## Framing

Correct framing:

```text
Disorder-inspired artificial dynamics visualization.
```

Incorrect framing:

```text
Medical diagnosis.
Proof of how a specific human brain works.
A biological one-to-one simulation of BPD, ADHD, or depression.
```

## Possible future modes

### Normal profile

Reference run with default routing, attention, memory replay, action selection, and error/surprise settings.

### ADHD-inspired profile

Possible artificial dynamics:

- weaker sustained attention gate
- higher routing variability
- more frequent task switching
- stronger distractor activation
- less stable workspace/task-state maintenance

Dashboard views:

- attention/router switching frequency
- task-focus stability
- region activation variability
- distractor vs target activation

### Depression-inspired profile

Possible artificial dynamics:

- lower action-initiation signal
- stronger negative/error replay weighting
- sticky replay loops
- reduced exploration
- reduced transition/action energy

Dashboard views:

- action-initiation strength
- replay pressure
- repeated error loop count
- exploration vs exploitation balance

### BPD-inspired profile

Possible artificial dynamics:

- stronger surprise/threat spikes
- unstable valuation
- faster safe/unsafe interpretation switching
- weaker top-down regulation during conflict
- stronger priority/salience bursts

Dashboard views:

- surprise/threat spike timeline
- valuation instability
- top-down regulation strength
- memory-triggered state shifts

## Possible dashboard name

```text
BrainGraph Mental Dynamics Lab
```

## Possible UI sections

```text
1. Profile selector
   normal / ADHD-inspired / depression-inspired / BPD-inspired

2. 3D BrainGraph activation viewer
   region activation over time

3. Connection dynamics
   stronger/weaker artificial pathways

4. Router stability
   thalamus/router switching behavior

5. Memory replay pressure
   hippocampus/replay activity

6. Action initiation
   basal-ganglia/action-selection strength

7. Error/surprise timeline
   priority/salience spikes

8. Side-by-side comparison
   normal vs altered artificial dynamics
```

## Important project boundary

This future branch must not distract from the current architecture work.

Do not start this until WB-MicroJEPA can reliably learn simple transition structures and the monitoring dashboard can already inspect artificial region/micro-unit specialization.

## Activation condition

Revisit after:

```text
WB-MicroJEPA-0C or later
- delta-structured prediction works
- world-context modulation works
- monitoring dashboard exists
- region/micro-unit specialization can be inspected
```
