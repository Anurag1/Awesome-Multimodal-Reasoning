"""Controlled intervention-scope comparison: shared vs independent latent states."""
import numpy as np

D = 64
EPS = 0.10
OUT = {"text": 16, "image": 64, "physics": 4}
N_SEEDS, N_PERTURBATIONS = 20, 30
NAMES = tuple(OUT)


def unit(x):
    norm = np.linalg.norm(x)
    assert norm > 0
    return x / norm


def tangent_edit(z, rng):
    direction = rng.standard_normal(D) + 1j * rng.standard_normal(D)
    tangent = direction - z * np.vdot(z, direction)
    tangent = unit(tangent)
    assert abs(np.vdot(z, tangent)) < 1e-10
    return unit(z + EPS * tangent)


def project(z, matrix):
    return np.real(matrix @ z)


def changed(before, after):
    return np.linalg.norm(after - before) > 1e-12


def run():
    shared_cover, independent_cover, local_cover = [], [], []
    shared_norms = {name: [] for name in NAMES}
    independent_norms = {name: [] for name in NAMES}
    leakage = []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        matrices = {
            name: rng.standard_normal((dim, D)) / np.sqrt(D)
            for name, dim in OUT.items()
        }
        shared_z = unit(rng.standard_normal(D) + 1j * rng.standard_normal(D))
        independent_z = {
            name: unit(rng.standard_normal(D) + 1j * rng.standard_normal(D))
            for name in NAMES
        }
        shared_base = {n: project(shared_z, matrices[n]) for n in NAMES}
        independent_base = {
            n: project(independent_z[n], matrices[n]) for n in NAMES
        }

        for step in range(N_PERTURBATIONS):
            target = NAMES[step % len(NAMES)]

            # One intervention on the common state propagates through every W.
            edited_shared = tangent_edit(shared_z, rng)
            shared_after = {n: project(edited_shared, matrices[n]) for n in NAMES}
            shared_delta = {n: shared_after[n] - shared_base[n] for n in NAMES}
            shared_cover.append(sum(changed(shared_base[n], shared_after[n])
                                    for n in NAMES) / len(NAMES))
            for n in NAMES:
                shared_norms[n].append(np.linalg.norm(shared_delta[n]))

            # Matched intervention: edit only the selected modality's private state.
            edited_private = tangent_edit(independent_z[target], rng)
            independent_after = dict(independent_base)
            independent_after[target] = project(edited_private, matrices[target])
            independent_cover.append(sum(
                changed(independent_base[n], independent_after[n]) for n in NAMES
            ) / len(NAMES))
            for n in NAMES:
                independent_norms[n].append(
                    np.linalg.norm(independent_after[n] - independent_base[n])
                )
            total = sum(np.linalg.norm(independent_after[n] - independent_base[n])
                        for n in NAMES)
            off_target = sum(
                np.linalg.norm(independent_after[n] - independent_base[n])
                for n in NAMES if n != target
            )
            leakage.append(off_target / total if total else 0.0)

            # Independent all-local control: edit all three private states separately.
            edited_all = {
                n: tangent_edit(independent_z[n], rng) for n in NAMES
            }
            local_after = {n: project(edited_all[n], matrices[n]) for n in NAMES}
            local_cover.append(sum(changed(independent_base[n], local_after[n])
                                   for n in NAMES) / len(NAMES))

    mean = lambda values: float(np.mean(values))
    print("Controlled comparison: shared vs independent latent states")
    print(f"seeds={N_SEEDS}  perturbations/seed={N_PERTURBATIONS}  "
          f"interventions={N_SEEDS * N_PERTURBATIONS}")
    print("Metric: fraction of modality outputs changed by one intervention")
    print(f"shared single-state edit       : {mean(shared_cover):.3f}")
    print(f"independent single-target edit : {mean(independent_cover):.3f}")
    print(f"independent all-local edits    : {mean(local_cover):.3f}")
    print(f"independent off-target leakage : {mean(leakage):.3e}")
    print("Mean output-shift L2 by modality (shared | independent-target):")
    for n in NAMES:
        print(f"  {n:7s}: {mean(shared_norms[n]):.6f} | "
              f"{mean(independent_norms[n]):.6f}")
    print("Interpretation: a shared state couples modalities under one edit; "
          "independent states require separate edits. All-local edits can move "
          "every output, but do not create a shared causal latent.")


if __name__ == "__main__":
    run()
