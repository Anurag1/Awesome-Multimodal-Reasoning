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


## Real-only vs complex-preserving projection benchmark

Run:

```bash
python complex_projection_benchmark.py
```

The benchmark compares the current map

```python
y_real = real(W @ z)
```

with a complex projection that retains both output channels:

```python
A = (W + 1j * V) / sqrt(2)
y_complex = A @ z
```

Here `W` and `V` are independent real Gaussian matrices with entries scaled by `1/sqrt(D)`. Thus `E|A_ij|^2 = 1/D`. The complex result is kept complex; neither its real nor imaginary output is discarded. To compare output energy fairly, RMS is normalized by the number of real scalar channels: `sqrt(m)` for the real map and `sqrt(2*m)` for the complex map.

**Matched protocol:** 32 deterministic seeds, 50 common random tangent directions per seed (1,600 interventions). Within each seed, both methods use the same normalized complex latent state and the same tangent directions. Projection dimensions remain text `(16,64)`, image `(64,64)`, and physics `(4,64)`.

Two measurements are reported:

1. **Modality coverage:** fraction of the three outputs whose directional derivative is nonzero after one shared-state edit.
2. **Sensitivity-profile concordance:** for each seed, Pearson correlations between the three modality response-RMS profiles across 50 perturbations; report the average of the three modality pairs. This measures whether modalities' response magnitudes co-vary across perturbations. It is not a measure of semantic alignment.

A separate **imaginary-only tangent probe** constructs unit tangent directions of the form `t = i*v`, with real `v` orthogonal to both `Re(z)` and `Im(z)`. For the current real-only map, `Re(W @ t)` is exactly zero. A generic complex projection responds to this direction. This directly tests the information blind spot rather than inferring it from random perturbations.

### Executed reference output

```text
Matched projection benchmark: real-only vs complex-preserving
D=64  seeds=32  perturbations/seed=50  total=1600
General tangent edits: modality output coverage
  real-only : 1.000
  complex   : 1.000
Sensitivity-profile concordance (mean +/- SD over 32 seed-level averages)
  real-only : 0.0901 +/- 0.0957
  complex   : 0.0125 +/- 0.0881
  complex - real: -0.0777
Imaginary-only tangent probe: RMS directional derivative
  real-only : 0.00000000
  complex   : 0.087424
  max real-only imaginary leakage: 0.000e+00
General-edit sensitivity RMS per real output channel:
  text   : real=0.087233  complex=0.087689
  image  : real=0.088280  complex=0.088219
  physics: real=0.083962  complex=0.086687
```

### Interpretation

- **No coverage gain:** both designs already transmit one shared-latent edit to all three outputs (100% coverage).
- **Clear observability gain:** the real-only map has an exact null response to the tested imaginary-only tangent directions; the complex map does not.
- **No evidence of stronger cross-modal coupling:** complex sensitivity-profile concordance was lower in this random-projection experiment (0.0125 vs 0.0901). The difference is descriptive; this small random-projection study does not establish a statistically robust disadvantage.
- **Comparable ordinary sensitivity:** per-real-channel RMS responses are similar for all three modalities under general tangent edits.

The result is therefore: complex projections preserve access to both latent quadratures, but random complex matrices do not automatically make modality responses more coordinated or semantically aligned. The projection is still compressive (especially physics: 4 complex outputs from 64 complex latent dimensions), and it is not an invertible encoding of the entire latent state.

A stronger-coupling hypothesis needs an additional mechanism—such as shared structured projections, cross-modal alignment losses, or decoders trained on paired observations—and a task-level benchmark using actual aligned data. This experiment isolates projection algebra only; it does not test training, semantics, or downstream accuracy.

## Scope

An executable architecture sketch, not a trained multimodal model, benchmark, or scientific claim.
