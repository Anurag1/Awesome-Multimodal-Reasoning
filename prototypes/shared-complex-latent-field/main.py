"""One complex latent vector projected into text, image, and physics spaces."""
import numpy as np

SEED, D, EPS = 2099, 64, 0.10
RNG = np.random.default_rng(SEED)


def unit(x):
    return x / np.linalg.norm(x)


def projection(out_dim):
    """Fixed real projection scaled by 1/sqrt(D)."""
    return RNG.standard_normal((out_dim, D)) / np.sqrt(D)


z = unit(RNG.standard_normal(D) + 1j * RNG.standard_normal(D))
W = {"text": projection(16), "image": projection(64),
     "physics": projection(4)}


def validate(state, tangent, matrices):
    """Fail loudly on a non-tangent edit or malformed projection."""
    inner = np.vdot(state, tangent)
    assert abs(inner) < 1e-10, f"edit is not orthogonal to z: {inner}"
    expected = {"text": (16, D), "image": (64, D), "physics": (4, D)}
    for name, shape in expected.items():
        assert matrices[name].shape == shape, (
            f"{name} projection shape {matrices[name].shape}; expected {shape}")


def modalities(state):
    return {name: np.real(matrix @ state) for name, matrix in W.items()}


direction = RNG.standard_normal(D) + 1j * RNG.standard_normal(D)
tangent = direction - z * np.vdot(z, direction)
tangent = unit(tangent)
validate(z, tangent, W)
z_edited = unit(z + EPS * tangent)

before, after = modalities(z), modalities(z_edited)
shifts = {name: after[name] - before[name] for name in W}
shift_norms = {name: np.linalg.norm(value) for name, value in shifts.items()}
# The physics vector has 4 entries; resize repeats it to compare with 64 image entries.
correlation = np.corrcoef(shifts["image"],
                          np.resize(shifts["physics"], 64))[0, 1]

print(f"seed={SEED}  D={D}  |z|={np.linalg.norm(z):.12f}")
print(f"tangent inner product |<z,t>|={abs(np.vdot(z, tangent)):.3e}")
for name in ("text", "image", "physics"):
    print(f"{name:7s} shift L2 = {shift_norms[name]:.6f}")
print(f"image/physics shift correlation (resized physics) = {correlation:.6f}")
print("Interpretation: all three projections move after one latent edit; "
      "correlation measures alignment, not semantic coherence.")
