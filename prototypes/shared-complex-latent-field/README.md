# Shared Complex Latent Field — Minimal PoC

A NumPy-only design-pattern demo: one L2-normalized complex latent vector drives three modality projections (text, image, physics). The projection matrices are fixed random matrices; **nothing is trained**.

## Run

```bash
python main.py
```

Requires Python 3 and NumPy. No weights, downloads, or GPU are required.

## Mechanism

- Latent state: `z ∈ C^64`, normalized to unit L2 norm.
- Fixed real Gaussian projections, each scaled by `1/sqrt(64)`: text `(16, 64)`, image `(64, 64)`, physics `(4, 64)`.
- Each modality is `real(W @ z)`.
- A random complex direction is projected onto the tangent space at `z`, normalized, and applied as a small edit.
- Assertions check tangent orthogonality and every projection shape.

## Reproducibility

All random draws use NumPy's `default_rng(2099)`. The script prints the measured output of the seeded run.

## Interpretation and limitations

A single edit changes all three projected outputs. The image/physics shift correlation repeats the 4-element physics shift to length 64 so it can be compared with the 64-element image shift. This is a descriptive statistic, not a canonical cross-modal similarity measure; repetition introduces dependence and it should not be interpreted as semantic coherence. Random projections do not encode modality semantics. This PoC does not establish that a learned shared representation is possible or useful.


## Controlled shared-vs-independent experiment

Run the multi-seed intervention comparison:

```bash
python controlled_experiment.py
```

The experiment uses 20 deterministic seeds and 30 perturbations per seed (600 interventions). For each seed, both conditions reuse the same fixed modality projection matrices and the same dimensions.

- **Shared condition:** one complex state is edited along a normalized tangent direction; all three modality projections are recomputed.
- **Independent targeted condition:** one modality's private complex state is edited per intervention; the other two states are held fixed. The target rotates across text, image, and physics.
- **Independent all-local control:** all three private states are independently edited in the same trial. This verifies that independent systems can move every output when given three separate edits.

Primary metric: fraction of modality outputs that change under one intervention. Off-target leakage in the independent targeted condition should be zero because its modality states are disconnected. Per-modality independent shift norms are averaged only over interventions targeting that modality; shared shift norms are averaged over all shared interventions.

### Executed reference output

```text
Controlled comparison: shared vs independent latent states
seeds=20  perturbations/seed=30  interventions=600
Metric: fraction of modality outputs changed by one intervention
shared single-state edit       : 1.000
independent single-target edit : 0.333
independent all-local edits    : 1.000
independent off-target leakage : 0.000e+00
Mean output-shift L2 by modality (shared | independent-target):
  text   : 0.034288 | 0.033945
  image  : 0.070474 | 0.071365
  physics: 0.016231 | 0.016327
```

These results demonstrate **causal coupling by construction**, not learned semantic coherence or a performance advantage. The independent all-local control reaches full output coverage by spending three independent edits; it does not have one common latent cause. The projection maps are random and untrained. The experiment does not compare semantic quality, task accuracy, robustness, or sample efficiency.

## Scope

An executable architecture sketch, not a trained multimodal model, benchmark, or scientific claim.
