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

## Scope

An executable architecture sketch, not a trained multimodal model, benchmark, or scientific claim.
