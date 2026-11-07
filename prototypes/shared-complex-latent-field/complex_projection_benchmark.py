"""Matched benchmark: real-only vs complex projections of one complex latent state."""
import numpy as np

D = 64
OUT = {"text": 16, "image": 64, "physics": 4}
N_SEEDS, N_PERTURBATIONS = 32, 50
NAMES = tuple(OUT)


def unit(x):
    norm = np.linalg.norm(x)
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return x / norm


def random_tangent(z, rng):
    d = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    t = d - z * np.vdot(z, d)
    t = unit(t)
    assert abs(np.vdot(z, t)) < 1e-10
    return t


def imaginary_tangent(z, rng):
    """Return t=i*v with real v orthogonal to Re(z) and Im(z)."""
    v = rng.standard_normal(D)
    basis = np.column_stack((z.real, z.imag))
    v -= basis @ np.linalg.lstsq(basis, v, rcond=None)[0]
    v = unit(v)
    t = 1j * v
    assert np.linalg.norm(t.real) == 0.0
    assert abs(np.vdot(z, t)) < 1e-10
    return t


def correlation(a, b):
    if np.std(a) < 1e-14 or np.std(b) < 1e-14:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def run():
    coverage_real, coverage_complex = [], []
    sensitivity_real = {n: [] for n in NAMES}
    sensitivity_complex = {n: [] for n in NAMES}
    corr_real, corr_complex = [], []
    imag_real, imag_complex = [], []
    max_real_imag_leak = 0.0

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        z = unit(rng.standard_normal(D) + 1j * rng.standard_normal(D))
        w_real, w_imag = {}, {}
        for name, dim in OUT.items():
            base = rng.standard_normal((dim, D)) / np.sqrt(D)
            quad = rng.standard_normal((dim, D)) / np.sqrt(D)
            w_real[name] = base
            # E|A_ij|^2 = 1/D; both real and imaginary channels are retained.
            w_imag[name] = (base + 1j * quad) / np.sqrt(2.0)
            assert base.shape == (dim, D) and w_imag[name].shape == (dim, D)

        seed_real = {n: [] for n in NAMES}
        seed_complex = {n: [] for n in NAMES}
        for _ in range(N_PERTURBATIONS):
            t = random_tangent(z, rng)
            active_real, active_complex = [], []
            for name in NAMES:
                dr = np.real(w_real[name] @ t)
                dc = w_imag[name] @ t
                # RMS per real scalar channel: m channels vs 2m complex channels.
                er = np.linalg.norm(dr) / np.sqrt(OUT[name])
                ec = np.linalg.norm(dc) / np.sqrt(2 * OUT[name])
                seed_real[name].append(er)
                seed_complex[name].append(ec)
                active_real.append(np.linalg.norm(dr) > 1e-12)
                active_complex.append(np.linalg.norm(dc) > 1e-12)
            coverage_real.append(np.mean(active_real))
            coverage_complex.append(np.mean(active_complex))

            ti = imaginary_tangent(z, rng)
            for name in NAMES:
                di_real = np.real(w_real[name] @ ti)
                di_complex = w_imag[name] @ ti
                imag_real.append(np.linalg.norm(di_real) / np.sqrt(OUT[name]))
                imag_complex.append(
                    np.linalg.norm(di_complex) / np.sqrt(2 * OUT[name])
                )
                max_real_imag_leak = max(max_real_imag_leak,
                                         float(np.linalg.norm(di_real)))

        # Correlate cross-modal sensitivity profiles across perturbations.
        seed_corr_real, seed_corr_complex = [], []
        for i in range(len(NAMES)):
            for j in range(i + 1, len(NAMES)):
                a, b = NAMES[i], NAMES[j]
                seed_corr_real.append(correlation(seed_real[a], seed_real[b]))
                seed_corr_complex.append(
                    correlation(seed_complex[a], seed_complex[b])
                )
        corr_real.append(float(np.nanmean(seed_corr_real)))
        corr_complex.append(float(np.nanmean(seed_corr_complex)))

        for name in NAMES:
            sensitivity_real[name].extend(seed_real[name])
            sensitivity_complex[name].extend(seed_complex[name])

    mean = lambda x: float(np.nanmean(x))
    print("Matched projection benchmark: real-only vs complex-preserving")
    print(f"D={D}  seeds={N_SEEDS}  perturbations/seed={N_PERTURBATIONS}  "
          f"total={N_SEEDS * N_PERTURBATIONS}")
    print("General tangent edits: modality output coverage")
    print(f"  real-only : {mean(coverage_real):.3f}")
    print(f"  complex   : {mean(coverage_complex):.3f}")
    print("Sensitivity-profile concordance (mean +/- SD over 32 seed-level averages)")
    print(f"  real-only : {mean(corr_real):.4f} +/- {float(np.nanstd(corr_real)):.4f}")
    print(f"  complex   : {mean(corr_complex):.4f} +/- {float(np.nanstd(corr_complex)):.4f}")
    delta = np.asarray(corr_complex) - np.asarray(corr_real)
    print(f"  complex - real: {mean(delta):+.4f}")
    print("Imaginary-only tangent probe: RMS directional derivative")
    print(f"  real-only : {mean(imag_real):.8f}")
    print(f"  complex   : {mean(imag_complex):.6f}")
    print(f"  max real-only imaginary leakage: {max_real_imag_leak:.3e}")
    print("General-edit sensitivity RMS per real output channel:")
    for name in NAMES:
        print(f"  {name:7s}: real={mean(sensitivity_real[name]):.6f}  "
              f"complex={mean(sensitivity_complex[name]):.6f}")
    print("Conclusion: complex projections expose imaginary latent directions; "
          "they do not automatically improve cross-modal coupling or semantics.")


if __name__ == "__main__":
    run()
