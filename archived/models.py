"""
models.py

Minimal model definitions for the Topological Impurity Bands project.

This file intentionally stays small:
    - QWZModel: fully usable for the first-stage Bott-index scans.
    - HaldaneModel: clean Haldane model implemented with a standard convention;
      draft-specific hexagon impurities are left as an explicit TODO, because
      their bond convention should be sanity-checked carefully before use.

Main usage example
------------------
from models import QWZModel, sample_impurity_sites

model = QWZModel(Lx=32, Ly=32, t=1.0, u=-2.05, A=1.5, pbc=True)
impurities = sample_impurity_sites(32, 32, density=0.03, seed=0)

H0 = model.clean_hamiltonian(sparse=False)
H  = model.hamiltonian(impurities=impurities, h=-1.7, sparse=False)
positions = model.positions()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None


Site = Tuple[int, int]


def _empty_matrix(dim: int, sparse: bool):
    """Create an empty complex matrix container."""
    if sparse:
        if sp is None:
            raise ImportError("scipy is required for sparse=True")
        return sp.lil_matrix((dim, dim), dtype=np.complex128)
    return np.zeros((dim, dim), dtype=np.complex128)


def _finalize_matrix(H, sparse: bool):
    """Convert sparse matrix to CSR; leave dense matrix unchanged."""
    if sparse:
        return H.tocsr()
    return H


def _wrap_or_skip(x: int, L: int, pbc: bool) -> Optional[int]:
    """Return periodic coordinate or None if open boundary and outside."""
    if 0 <= x < L:
        return x
    if pbc:
        return x % L
    return None


def sample_impurity_sites(
    Lx: int,
    Ly: int,
    density: Optional[float] = None,
    n_imp: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Site]:
    """
    Sample a fixed number of onsite impurities on an Lx x Ly grid.

    Use fixed N_imp rather than Bernoulli sampling so that different disorder
    configurations have the same impurity density.
    """
    if n_imp is None:
        if density is None:
            raise ValueError("Either density or n_imp must be provided.")
        n_imp = int(round(float(density) * Lx * Ly))

    if n_imp < 0 or n_imp > Lx * Ly:
        raise ValueError(f"Invalid n_imp={n_imp} for Lx*Ly={Lx*Ly}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(Lx * Ly, size=n_imp, replace=False)

    impurities: List[Site] = []
    for s in chosen:
        x = int(s % Lx)
        y = int(s // Lx)
        impurities.append((x, y))
    return impurities


@dataclass
class QWZModel:
    """
    Qi-Wu-Zhang model with onsite mass impurities.

    Real-space convention follows the draft form

        H0 onsite: u sigma_z
        NN hopping along bond angle theta:
            (1/2) [[ t, i A exp(-i theta) ],
                   [ i A exp(+i theta), -t ]]

        impurity at R:
            V_R = -h sigma_z

    With the oriented +x/+y bond convention used here, the clean Bloch
    Hamiltonian is equivalent to

        H(k) = -A sin(kx) sigma_x - A sin(ky) sigma_y
             + [u + t cos(kx) + t cos(ky)] sigma_z.

    The sign convention of the sin terms can flip the sign of the Chern number
    but not the existence/location of the transitions we are targeting.
    """

    Lx: int
    Ly: int
    t: float = 1.0
    u: float = -2.05
    A: float = 1.5
    pbc: bool = True

    norb: int = 2

    @property
    def dim(self) -> int:
        return self.norb * self.Lx * self.Ly

    def idx(self, x: int, y: int, orb: int) -> int:
        """Basis index for |x, y, orb>, orb=0,1."""
        if orb not in (0, 1):
            raise ValueError("QWZ orbital index must be 0 or 1.")
        x = x % self.Lx
        y = y % self.Ly
        return self.norb * (x + self.Lx * y) + orb

    def positions(self) -> np.ndarray:
        """
        Return positions for every basis state.

        Shape: (dim, 2). Both orbitals at the same site share the same position.
        This is the input needed by a real-space Bott-index routine.
        """
        pos = np.zeros((self.dim, 2), dtype=float)
        for y in range(self.Ly):
            for x in range(self.Lx):
                for orb in range(self.norb):
                    pos[self.idx(x, y, orb)] = (float(x), float(y))
        return pos

    def _add_onsite_block(self, H, x: int, y: int, block: np.ndarray):
        for a in range(self.norb):
            ia = self.idx(x, y, a)
            for b in range(self.norb):
                ib = self.idx(x, y, b)
                H[ia, ib] += block[a, b]

    def _add_hopping_block(self, H, x1: int, y1: int, x2: int, y2: int, block: np.ndarray):
        """Add c_{r1}^dagger block c_{r2} + h.c."""
        for a in range(self.norb):
            i = self.idx(x1, y1, a)
            for b in range(self.norb):
                j = self.idx(x2, y2, b)
                H[i, j] += block[a, b]
                H[j, i] += np.conjugate(block[a, b])

    def clean_hamiltonian(self, sparse: bool = False):
        """Build the clean finite-size QWZ Hamiltonian."""
        H = _empty_matrix(self.dim, sparse=sparse)

        onsite = np.array([[self.u, 0.0], [0.0, -self.u]], dtype=np.complex128)
        for y in range(self.Ly):
            for x in range(self.Lx):
                self._add_onsite_block(H, x, y, onsite)

        # +x bond: theta=0
        hop_x = 0.5 * np.array(
            [[self.t, 1j * self.A], [1j * self.A, -self.t]],
            dtype=np.complex128,
        )

        # +y bond: theta=pi/2
        # i A exp(-i pi/2) = +A, i A exp(+i pi/2) = -A
        hop_y = 0.5 * np.array(
            [[self.t, self.A], [-self.A, -self.t]],
            dtype=np.complex128,
        )

        for y in range(self.Ly):
            for x in range(self.Lx):
                xp = _wrap_or_skip(x + 1, self.Lx, self.pbc)
                if xp is not None:
                    self._add_hopping_block(H, x, y, xp, y, hop_x)

                yp = _wrap_or_skip(y + 1, self.Ly, self.pbc)
                if yp is not None:
                    self._add_hopping_block(H, x, y, x, yp, hop_y)

        return _finalize_matrix(H, sparse=sparse)

    def impurity_potential(
        self,
        impurities: Optional[Sequence[Site]],
        h: float,
        sparse: bool = False,
    ):
        """Build V = sum_R -h sigma_z on impurity sites."""
        V = _empty_matrix(self.dim, sparse=sparse)
        if impurities is None:
            return _finalize_matrix(V, sparse=sparse)

        block = np.array([[-h, 0.0], [0.0, h]], dtype=np.complex128)
        for x, y in impurities:
            self._add_onsite_block(V, int(x) % self.Lx, int(y) % self.Ly, block)

        return _finalize_matrix(V, sparse=sparse)

    def hamiltonian(
        self,
        impurities: Optional[Sequence[Site]] = None,
        h: float = 0.0,
        sparse: bool = False,
    ):
        """Build H = H0 + V_imp."""
        H = self.clean_hamiltonian(sparse=sparse)
        if impurities is None or len(impurities) == 0 or abs(h) == 0:
            return H
        V = self.impurity_potential(impurities=impurities, h=h, sparse=sparse)
        return H + V


@dataclass
class HaldaneModel:
    """
    Clean Haldane model on a honeycomb lattice.

    This class is included for later Bott-index checks, but the draft-specific
    hexagon impurity is intentionally not implemented yet. The hexagon impurity
    changes the oriented next-nearest-neighbor hoppings on one hexagonal
    plaquette, so we should add it only after a separate geometry/sign sanity
    check.

    Current clean convention:
        - two sublattices A=0, B=1 per unit cell (x,y)
        - NN bonds from A(x,y) to B(x,y), B(x-1,y), B(x,y-1)
        - NNN vectors (1,0), (0,1), (-1,1)
        - A sublattice NNN hopping: +i t2 along these vectors
        - B sublattice NNN hopping: -i t2 along these vectors
        - staggered mass: +m/2 on A, -m/2 on B

    A different convention may flip the Chern-number sign.
    """

    Lx: int
    Ly: int
    t1: float = 1.0
    t2: float = 0.1
    m: float = 0.0
    pbc: bool = True

    nsub: int = 2

    @property
    def dim(self) -> int:
        return self.nsub * self.Lx * self.Ly

    def idx(self, x: int, y: int, sub: int) -> int:
        """Basis index for |x, y, sub>, sub=0(A),1(B)."""
        if sub not in (0, 1):
            raise ValueError("Haldane sublattice index must be 0(A) or 1(B).")
        x = x % self.Lx
        y = y % self.Ly
        return self.nsub * (x + self.Lx * y) + sub

    def positions(self) -> np.ndarray:
        """
        Return positions in unit-cell coordinates.

        For Bott-index purposes, using unit-cell coordinates is usually enough.
        Small sublattice offsets are included to avoid exactly coincident A/B
        positions in diagnostic plots.
        """
        pos = np.zeros((self.dim, 2), dtype=float)
        offsets = {0: (0.0, 0.0), 1: (1.0 / 3.0, 1.0 / 3.0)}
        for y in range(self.Ly):
            for x in range(self.Lx):
                for sub in (0, 1):
                    ox, oy = offsets[sub]
                    pos[self.idx(x, y, sub)] = (float(x) + ox, float(y) + oy)
        return pos

    def _valid_neighbor(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        xx = _wrap_or_skip(x, self.Lx, self.pbc)
        yy = _wrap_or_skip(y, self.Ly, self.pbc)
        if xx is None or yy is None:
            return None
        return xx, yy

    def _add_hop(self, H, i: int, j: int, amp: complex):
        H[i, j] += amp
        H[j, i] += np.conjugate(amp)

    def clean_hamiltonian(self, sparse: bool = False):
        """Build the clean finite-size Haldane Hamiltonian."""
        H = _empty_matrix(self.dim, sparse=sparse)

        for y in range(self.Ly):
            for x in range(self.Lx):
                H[self.idx(x, y, 0), self.idx(x, y, 0)] += 0.5 * self.m
                H[self.idx(x, y, 1), self.idx(x, y, 1)] += -0.5 * self.m

        # NN hopping: A(x,y) to B(x,y), B(x-1,y), B(x,y-1)
        nn_offsets = [(0, 0), (-1, 0), (0, -1)]
        for y in range(self.Ly):
            for x in range(self.Lx):
                iA = self.idx(x, y, 0)
                for dx, dy in nn_offsets:
                    nb = self._valid_neighbor(x + dx, y + dy)
                    if nb is None:
                        continue
                    xb, yb = nb
                    iB = self.idx(xb, yb, 1)
                    self._add_hop(H, iA, iB, -self.t1)

        # NNN imaginary hopping.
        nnn_vecs = [(1, 0), (0, 1), (-1, 1)]
        for y in range(self.Ly):
            for x in range(self.Lx):
                for dx, dy in nnn_vecs:
                    nb = self._valid_neighbor(x + dx, y + dy)
                    if nb is None:
                        continue
                    xp, yp = nb

                    i = self.idx(x, y, 0)
                    j = self.idx(xp, yp, 0)
                    self._add_hop(H, i, j, 1j * self.t2)

                    i = self.idx(x, y, 1)
                    j = self.idx(xp, yp, 1)
                    self._add_hop(H, i, j, -1j * self.t2)

        return _finalize_matrix(H, sparse=sparse)

    def impurity_potential(
        self,
        impurities: Optional[Sequence[Site]],
        h: float,
        sparse: bool = False,
    ):
        """
        Draft-specific Haldane hexagon impurity.

        Not implemented yet on purpose. We should add this after checking the
        hexagon-bond convention against a drawing / small finite lattice.
        """
        if impurities is None or len(impurities) == 0 or abs(h) == 0:
            return _empty_matrix(self.dim, sparse=sparse)
        raise NotImplementedError(
            "Haldane hexagon impurity is not implemented yet. "
            "Clean HaldaneModel is available; use QWZModel for first scans."
        )

    def hamiltonian(
        self,
        impurities: Optional[Sequence[Site]] = None,
        h: float = 0.0,
        sparse: bool = False,
    ):
        """Build clean Haldane Hamiltonian; impurities currently not implemented."""
        H = self.clean_hamiltonian(sparse=sparse)
        if impurities is None or len(impurities) == 0 or abs(h) == 0:
            return H
        V = self.impurity_potential(impurities=impurities, h=h, sparse=sparse)
        return H + V


def check_hermitian(H, atol: float = 1e-10) -> bool:
    """Quick Hermiticity check for dense or sparse matrices."""
    if sp is not None and sp.issparse(H):
        diff = H - H.getH()
        if diff.nnz == 0:
            return True
        return np.max(np.abs(diff.data)) < atol
    return np.allclose(H, H.conj().T, atol=atol)


if __name__ == "__main__":
    model = QWZModel(Lx=6, Ly=5, t=1.0, u=-2.05, A=1.5)
    impurities = sample_impurity_sites(6, 5, density=0.03, seed=0)
    H = model.hamiltonian(impurities=impurities, h=-1.7, sparse=False)
    print("QWZ dim:", model.dim)
    print("QWZ n_imp:", len(impurities))
    print("QWZ Hermitian:", check_hermitian(H))

    hmodel = HaldaneModel(Lx=6, Ly=5, t1=1.0, t2=0.1, m=0.2)
    HH = hmodel.clean_hamiltonian(sparse=False)
    print("Haldane dim:", hmodel.dim)
    print("Haldane Hermitian:", check_hermitian(HH))
