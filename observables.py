"""
observables.py

Minimal observable routines for the Topological Impurity Bands project.

Designed to work with the current models.py file:
    - QWZModel
    - HaldaneModel clean model

Main routines
-------------
diagonalize(H)
spectral_gap(evals, fermi_energy=0.0)
near_fermi_indices(evals, fermi_energy=0.0, n_states=4)
bott_index(evals, evecs, positions, Lx, Ly, fermi_energy=0.0)
ipr(vec)
site_probability(vec, model)
impurity_weight(vec, model, impurities, radius=1.5)

Typical usage
-------------
from models import QWZModel, sample_impurity_sites
from observables import diagonalize, bott_index, spectral_gap, ipr, impurity_weight

model = QWZModel(Lx=24, Ly=24, u=-2.05, A=1.5)
impurities = sample_impurity_sites(24, 24, density=0.03, seed=0)
H = model.hamiltonian(impurities=impurities, h=-1.7)

evals, evecs = diagonalize(H)
B = bott_index(evals, evecs, model.positions(), model.Lx, model.Ly)
gap_info = spectral_gap(evals)

idx0 = near_fermi_indices(evals, n_states=1)[0]
vec0 = evecs[:, idx0]
print(B, gap_info["gap"], ipr(vec0), impurity_weight(vec0, model, impurities))
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


Site = Tuple[int, int]


def _as_dense(H) -> np.ndarray:
    """Convert dense or sparse matrix to dense ndarray."""
    if sp is not None and sp.issparse(H):
        return H.toarray()
    return np.asarray(H)


def diagonalize(H, sort: bool = True):
    """
    Dense Hermitian diagonalization.

    Parameters
    ----------
    H:
        Dense ndarray or scipy sparse matrix.
    sort:
        If True, return eigenvalues/eigenvectors sorted ascending.

    Returns
    -------
    evals:
        Eigenvalues.
    evecs:
        Eigenvectors as columns: evecs[:, n].
    """
    Hd = _as_dense(H)
    evals, evecs = np.linalg.eigh(Hd)

    if sort:
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

    return evals, evecs


def occupation_mask(evals: np.ndarray, fermi_energy: float = 0.0) -> np.ndarray:
    """
    Boolean mask for occupied states E < E_F.
    """
    return np.asarray(evals) < fermi_energy


def spectral_gap(evals: np.ndarray, fermi_energy: float = 0.0) -> Dict[str, float]:
    """
    Compute the finite-size spectral gap around a chosen Fermi energy.

    Returns a dict with:
        gap: E_lumo - E_homo
        homo: largest occupied eigenvalue
        lumo: smallest unoccupied eigenvalue
        n_occ: number of occupied states

    If the Fermi energy lies outside the spectrum, gap is NaN.
    """
    evals = np.sort(np.asarray(evals, dtype=float))
    occ = evals[evals < fermi_energy]
    unocc = evals[evals >= fermi_energy]

    if len(occ) == 0 or len(unocc) == 0:
        return {
            "gap": np.nan,
            "homo": np.nan,
            "lumo": np.nan,
            "n_occ": int(len(occ)),
        }

    homo = float(occ[-1])
    lumo = float(unocc[0])
    return {
        "gap": float(lumo - homo),
        "homo": homo,
        "lumo": lumo,
        "n_occ": int(len(occ)),
    }


def near_fermi_indices(
    evals: np.ndarray,
    fermi_energy: float = 0.0,
    n_states: int = 4,
) -> np.ndarray:
    """
    Indices of eigenstates closest to the Fermi energy.
    """
    evals = np.asarray(evals)
    n_states = min(int(n_states), len(evals))
    return np.argsort(np.abs(evals - fermi_energy))[:n_states]


def _unitarize_by_svd(M: np.ndarray) -> np.ndarray:
    """
    Return the unitary polar factor of M using SVD.

    This improves Bott-index stability when the projected position operators
    are not perfectly unitary at finite size / finite precision.
    """
    U, _s, Vh = np.linalg.svd(M, full_matrices=False)
    return U @ Vh


def bott_index(
    evals: np.ndarray,
    evecs: np.ndarray,
    positions: np.ndarray,
    Lx: int,
    Ly: int,
    fermi_energy: float = 0.0,
    unitarize: bool = True,
    return_details: bool = False,
):
    """
    Compute the real-space Bott index for a finite system with PBC.

    Convention
    ----------
    Let Q be the occupied-state matrix with columns |psi_n>, E_n < E_F.
    Define projected position phase operators

        U = Q^† exp(i 2π X/Lx) Q
        V = Q^† exp(i 2π Y/Ly) Q

    Then

        B = (1/2π) Im Tr log( V U V^† U^† )

    We compute Tr log by summing the phases of eigenvalues.

    Parameters
    ----------
    evals, evecs:
        Output from diagonalize(H). Eigenvectors are columns.
    positions:
        Array of shape (dim, 2), usually model.positions().
    Lx, Ly:
        System lengths used in the phase factors.
    fermi_energy:
        Occupied states have E < E_F.
    unitarize:
        If True, replace U,V by their unitary polar factors via SVD.
        This is usually more stable for finite-size numerics.
    return_details:
        If True, return dict. Otherwise return float Bott index.

    Returns
    -------
    B or details:
        Bott index as float, or dict containing bott, bott_round, n_occ, etc.
    """
    evals = np.asarray(evals)
    evecs = np.asarray(evecs)
    positions = np.asarray(positions)

    if positions.shape[0] != evecs.shape[0] or positions.shape[1] != 2:
        raise ValueError(
            f"positions must have shape (dim,2). Got {positions.shape}, "
            f"with evecs shape {evecs.shape}."
        )

    occ_mask = occupation_mask(evals, fermi_energy=fermi_energy)
    n_occ = int(np.count_nonzero(occ_mask))

    if n_occ == 0 or n_occ == len(evals):
        B = np.nan
        details = {
            "bott": B,
            "bott_round": np.nan,
            "n_occ": n_occ,
            "n_total": int(len(evals)),
            "fermi_energy": float(fermi_energy),
            "unitarized": bool(unitarize),
            "eig_min_abs_W": np.nan,
            "eig_max_abs_W": np.nan,
        }
        return details if return_details else B

    Q = evecs[:, occ_mask]

    phase_x = np.exp(2j * np.pi * positions[:, 0] / float(Lx))
    phase_y = np.exp(2j * np.pi * positions[:, 1] / float(Ly))

    # Since phase operators are diagonal in the real-space basis:
    # Q^† diag(phase) Q = Q^† (phase[:, None] * Q)
    U = Q.conj().T @ (phase_x[:, None] * Q)
    V = Q.conj().T @ (phase_y[:, None] * Q)

    if unitarize:
        U = _unitarize_by_svd(U)
        V = _unitarize_by_svd(V)

    W = V @ U @ V.conj().T @ U.conj().T
    w_eigs = np.linalg.eigvals(W)

    B = float(np.sum(np.angle(w_eigs)) / (2.0 * np.pi))

    details = {
        "bott": B,
        "bott_round": int(np.rint(B)),
        "n_occ": n_occ,
        "n_total": int(len(evals)),
        "fermi_energy": float(fermi_energy),
        "unitarized": bool(unitarize),
        "eig_min_abs_W": float(np.min(np.abs(w_eigs))),
        "eig_max_abs_W": float(np.max(np.abs(w_eigs))),
    }

    return details if return_details else B


def ipr(vec: np.ndarray, normalize: bool = True) -> float:
    """
    Inverse participation ratio in the full basis.

        IPR = sum_i |psi_i|^4

    If normalize=True, vec is normalized first.
    """
    vec = np.asarray(vec)
    if normalize:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.nan
        vec = vec / norm
    prob = np.abs(vec) ** 2
    return float(np.sum(prob ** 2))


def participation_ratio(vec: np.ndarray, normalize: bool = True) -> float:
    """
    Participation ratio PR = 1 / IPR.
    """
    val = ipr(vec, normalize=normalize)
    if val == 0 or np.isnan(val):
        return np.nan
    return float(1.0 / val)


def _internal_dim(model) -> int:
    """
    Number of internal states per site/unit cell.

    QWZModel has norb=2.
    HaldaneModel has nsub=2.
    """
    if hasattr(model, "norb"):
        return int(model.norb)
    if hasattr(model, "nsub"):
        return int(model.nsub)
    raise AttributeError("model must have either norb or nsub attribute.")


def site_probability(vec: np.ndarray, model, normalize: bool = True) -> np.ndarray:
    """
    Sum wavefunction probability over internal degrees of freedom at each site.

    Returns
    -------
    prob_site:
        Array shape (Lx, Ly), where prob_site[x,y] is the probability on
        that site/unit cell summed over orbitals/sublattices.
    """
    vec = np.asarray(vec)
    if normalize:
        norm = np.linalg.norm(vec)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        vec = vec / norm

    nint = _internal_dim(model)
    prob_site = np.zeros((model.Lx, model.Ly), dtype=float)

    for y in range(model.Ly):
        for x in range(model.Lx):
            total = 0.0
            for a in range(nint):
                total += float(np.abs(vec[model.idx(x, y, a)]) ** 2)
            prob_site[x, y] = total

    return prob_site


def _periodic_delta(d: int, L: int, pbc: bool) -> int:
    """
    Minimum-image coordinate difference.
    """
    if not pbc:
        return d
    return min(abs(d), L - abs(d))


def impurity_mask(model, impurities: Sequence[Site], radius: float = 1.5) -> np.ndarray:
    """
    Boolean mask on sites/unit cells within distance <= radius of any impurity.

    Distance is measured in the square/unit-cell coordinate system. For QWZ this
    is the physical square lattice. For Haldane this is a coarse unit-cell
    distance, mainly useful for simple diagnostics.
    """
    mask = np.zeros((model.Lx, model.Ly), dtype=bool)
    if impurities is None or len(impurities) == 0:
        return mask

    pbc = bool(getattr(model, "pbc", True))
    r2 = float(radius) ** 2

    for x in range(model.Lx):
        for y in range(model.Ly):
            for xi, yi in impurities:
                dx = _periodic_delta(abs(x - int(xi) % model.Lx), model.Lx, pbc)
                dy = _periodic_delta(abs(y - int(yi) % model.Ly), model.Ly, pbc)
                if dx * dx + dy * dy <= r2:
                    mask[x, y] = True
                    break

    return mask


def impurity_weight(
    vec: np.ndarray,
    model,
    impurities: Sequence[Site],
    radius: float = 1.5,
    normalize: bool = True,
) -> float:
    """
    Wavefunction probability weight near impurity sites.

        W_imp = sum_{sites within radius of impurities} |psi(site)|^2

    For QWZ, both orbitals are included at every selected site.
    """
    prob_site = site_probability(vec, model, normalize=normalize)
    mask = impurity_mask(model, impurities, radius=radius)
    return float(np.sum(prob_site[mask]))


def ldos_near_fermi(
    evals: np.ndarray,
    evecs: np.ndarray,
    model,
    fermi_energy: float = 0.0,
    eta: float = 0.05,
    normalize_states: bool = True,
) -> np.ndarray:
    """
    Simple LDOS-like spatial weight from states within |E - E_F| <= eta.

        rho_i = sum_{|E_n-E_F| <= eta} sum_a |psi_n(i,a)|^2

    Returns
    -------
    rho:
        Array shape (Lx, Ly).
    """
    evals = np.asarray(evals)
    selected = np.where(np.abs(evals - fermi_energy) <= eta)[0]
    rho = np.zeros((model.Lx, model.Ly), dtype=float)

    for n in selected:
        rho += site_probability(evecs[:, n], model, normalize=normalize_states)

    return rho


def summarize_near_fermi_state(
    evals: np.ndarray,
    evecs: np.ndarray,
    model,
    impurities: Optional[Sequence[Site]] = None,
    fermi_energy: float = 0.0,
    radius: float = 1.5,
) -> Dict[str, float]:
    """
    Convenience diagnostic for the single eigenstate closest to E_F.
    """
    idx0 = int(near_fermi_indices(evals, fermi_energy=fermi_energy, n_states=1)[0])
    vec0 = evecs[:, idx0]

    out = {
        "idx0": idx0,
        "E0": float(evals[idx0]),
        "abs_E0_minus_EF": float(abs(evals[idx0] - fermi_energy)),
        "ipr0": ipr(vec0),
        "pr0": participation_ratio(vec0),
    }

    if impurities is not None:
        out["wimp0"] = impurity_weight(
            vec0,
            model=model,
            impurities=impurities,
            radius=radius,
        )
    else:
        out["wimp0"] = np.nan

    return out


if __name__ == "__main__":
    # Minimal smoke test if models.py is in the same folder.
    from models import QWZModel, sample_impurity_sites

    model = QWZModel(Lx=8, Ly=8, u=-1.0, A=1.5)
    impurities = sample_impurity_sites(8, 8, density=0.03, seed=0)
    H = model.hamiltonian(impurities=impurities, h=-1.7)

    evals, evecs = diagonalize(H)
    B = bott_index(evals, evecs, model.positions(), model.Lx, model.Ly)
    gap = spectral_gap(evals)
    diag = summarize_near_fermi_state(evals, evecs, model, impurities)

    print("dim:", model.dim)
    print("n_imp:", len(impurities))
    print("Bott:", B)
    print("gap:", gap)
    print("near-EF diagnostics:", diag)
