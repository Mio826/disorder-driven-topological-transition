"""QWZmodel.py
================
Production Hamiltonian definition for the impurity-driven QWZ project.

This module contains only the microscopic Qi--Wu--Zhang model and matrix
construction utilities.  Disorder *sampling* (fixed-count, Bernoulli, or
correlated ensembles), finite-size scaling, observables, fitting, and
plotting intentionally belong in other modules.

Real-space convention
---------------------
Each square-lattice site carries two orbitals.  With site-major ordering
``|x, y, orbital>`` and lattice spacing set to one,

    H_0 = sum_r c_r^dagger (u sigma_z) c_r
          + sum_r [c_r^dagger T_x c_{r+x} + h.c.]
          + sum_r [c_r^dagger T_y c_{r+y} + h.c.],

where

    T_x = (t sigma_z + i A sigma_x) / 2,
    T_y = (t sigma_z + i A sigma_y) / 2.

The corresponding Bloch Hamiltonian is

    H_0(k) = -A sin(k_x) sigma_x - A sin(k_y) sigma_y
             + [u + t cos(k_x) + t cos(k_y)] sigma_z.

An onsite mass impurity at site R is

    V_R = -h sigma_z,

so the local mass changes from ``u`` to ``u - h``.

Notes
-----
* The signs of the sine terms are a convention.  Reversing both signs can
  flip the Chern-number sign but does not move the phase boundaries studied
  in this project.
* ``pbc`` accepts either one bool (same boundary condition in both axes) or
  ``(pbc_x, pbc_y)``.
* Impurity configurations are supplied explicitly.  This file does not
  choose how they are sampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, Optional, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - dense functionality remains usable
    sp = None


Site: TypeAlias = tuple[int, int]
BoundaryCondition: TypeAlias = bool | tuple[bool, bool]
DenseMatrix: TypeAlias = npt.NDArray[np.complex128]


# Pauli matrices are module constants so every numerical method uses exactly
# the same convention.
SIGMA_0: DenseMatrix = np.eye(2, dtype=np.complex128)
SIGMA_X: DenseMatrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y: DenseMatrix = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z: DenseMatrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _empty_matrix(dim: int, sparse: bool):
    """Return an empty complex dense matrix or sparse LIL matrix."""
    if sparse:
        if sp is None:
            raise ImportError("scipy is required when sparse=True")
        return sp.lil_matrix((dim, dim), dtype=np.complex128)
    return np.zeros((dim, dim), dtype=np.complex128)


def _finalize_matrix(matrix, sparse: bool):
    """Return sparse matrices in CSR format; leave dense matrices unchanged."""
    return matrix.tocsr() if sparse else matrix


def _coerce_integer(value: int, name: str) -> int:
    """Convert an integer-like scalar while rejecting booleans and floats."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; received {value!r}")
    return int(value)


def _max_abs(matrix) -> float:
    """Maximum absolute matrix entry for dense or sparse matrices."""
    if sp is not None and sp.issparse(matrix):
        if matrix.nnz == 0:
            return 0.0
        return float(np.max(np.abs(matrix.data)))
    array = np.asarray(matrix)
    return 0.0 if array.size == 0 else float(np.max(np.abs(array)))


@dataclass
class QWZModel:
    """Finite square-lattice Qi--Wu--Zhang Hamiltonian.

    Parameters
    ----------
    Lx, Ly:
        Numbers of lattice sites along the x and y directions.
    t:
        Wilson-mass hopping coefficient.
    u:
        Uniform onsite Dirac mass.
    A:
        Dirac hopping/velocity coefficient.
    pbc:
        ``True``/``False`` for the same boundary condition in both axes, or
        ``(pbc_x, pbc_y)`` for independent choices.

    Basis convention
    ----------------
    The basis index is

        ``2 * (x + Lx * y) + orbital``,  with ``orbital in {0, 1}``.

    Both orbitals at a site share the same real-space coordinate.
    """

    Lx: int
    Ly: int
    t: float = 1.0
    u: float = -2.05
    A: float = 1.5
    pbc: BoundaryCondition = True

    # Fixed by the physical model; it is deliberately not a dataclass input.
    norb: ClassVar[int] = 2

    def __post_init__(self) -> None:
        self.Lx = _coerce_integer(self.Lx, "Lx")
        self.Ly = _coerce_integer(self.Ly, "Ly")
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError(f"Lx and Ly must be positive; received {(self.Lx, self.Ly)}")

        self.t = float(self.t)
        self.u = float(self.u)
        self.A = float(self.A)
        for name, value in (("t", self.t), ("u", self.u), ("A", self.A)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite; received {value!r}")

        # Validate pbc early.  The original object is retained for backward
        # compatibility, while pbc_x/pbc_y expose normalized booleans.
        _ = self.boundary_conditions

    # ------------------------------------------------------------------
    # Geometry and indexing
    # ------------------------------------------------------------------
    @property
    def n_sites(self) -> int:
        """Number of lattice sites."""
        return self.Lx * self.Ly

    @property
    def dim(self) -> int:
        """Single-particle Hilbert-space dimension."""
        return self.norb * self.n_sites

    @property
    def boundary_conditions(self) -> tuple[bool, bool]:
        """Return normalized ``(pbc_x, pbc_y)`` boundary conditions."""
        if isinstance(self.pbc, (bool, np.bool_)):
            value = bool(self.pbc)
            return value, value

        if not isinstance(self.pbc, tuple) or len(self.pbc) != 2:
            raise TypeError("pbc must be a bool or a two-bool tuple (pbc_x, pbc_y)")

        px, py = self.pbc
        if not isinstance(px, (bool, np.bool_)) or not isinstance(py, (bool, np.bool_)):
            raise TypeError("pbc tuple entries must be booleans")
        return bool(px), bool(py)

    @property
    def pbc_x(self) -> bool:
        """Whether the x direction is periodic."""
        return self.boundary_conditions[0]

    @property
    def pbc_y(self) -> bool:
        """Whether the y direction is periodic."""
        return self.boundary_conditions[1]

    def _normalize_axis(self, value: int, length: int, periodic: bool, name: str) -> int:
        value = _coerce_integer(value, name)
        if periodic:
            return value % length
        if not 0 <= value < length:
            raise IndexError(f"{name}={value} lies outside the open interval [0, {length})")
        return value

    def normalize_site(self, x: int, y: int) -> Site:
        """Validate a site, wrapping only along periodic directions."""
        return (
            self._normalize_axis(x, self.Lx, self.pbc_x, "x"),
            self._normalize_axis(y, self.Ly, self.pbc_y, "y"),
        )

    def _neighbor(self, x: int, y: int, dx: int, dy: int) -> Optional[Site]:
        """Return a neighboring site or ``None`` when an open edge is crossed."""
        xn = x + dx
        yn = y + dy

        if self.pbc_x:
            xn %= self.Lx
        elif not 0 <= xn < self.Lx:
            return None

        if self.pbc_y:
            yn %= self.Ly
        elif not 0 <= yn < self.Ly:
            return None

        return int(xn), int(yn)

    def site_index(self, x: int, y: int) -> int:
        """Linear site index for ``(x, y)``."""
        x, y = self.normalize_site(x, y)
        return x + self.Lx * y

    def idx(self, x: int, y: int, orb: int) -> int:
        """Basis index for ``|x, y, orb>`` with ``orb`` equal to 0 or 1."""
        orb = _coerce_integer(orb, "orb")
        if orb not in (0, 1):
            raise ValueError(f"QWZ orbital index must be 0 or 1; received {orb}")
        return self.norb * self.site_index(x, y) + orb

    # Descriptive alias for new code; ``idx`` is retained for compatibility.
    basis_index = idx

    def site_positions(self) -> npt.NDArray[np.float64]:
        """Return one ``(x, y)`` coordinate per lattice site."""
        positions = np.empty((self.n_sites, 2), dtype=np.float64)
        for y in range(self.Ly):
            for x in range(self.Lx):
                positions[x + self.Lx * y] = (float(x), float(y))
        return positions

    def positions(self) -> npt.NDArray[np.float64]:
        """Return one ``(x, y)`` coordinate per basis state, shape ``(dim, 2)``."""
        return np.repeat(self.site_positions(), self.norb, axis=0)

    # ------------------------------------------------------------------
    # Public Hamiltonian blocks: the single source of truth used by ED,
    # Bott-index, transfer-matrix, DOS, and multifractal calculations.
    # ------------------------------------------------------------------
    @property
    def onsite_block(self) -> DenseMatrix:
        """Clean onsite block ``u sigma_z``."""
        return np.asarray(self.u * SIGMA_Z, dtype=np.complex128)

    @property
    def hopping_x(self) -> DenseMatrix:
        """Oriented +x hopping block ``(t sigma_z + i A sigma_x)/2``."""
        return np.asarray(0.5 * (self.t * SIGMA_Z + 1.0j * self.A * SIGMA_X))

    @property
    def hopping_y(self) -> DenseMatrix:
        """Oriented +y hopping block ``(t sigma_z + i A sigma_y)/2``."""
        return np.asarray(0.5 * (self.t * SIGMA_Z + 1.0j * self.A * SIGMA_Y))

    @staticmethod
    def impurity_block(h: float) -> DenseMatrix:
        """Onsite mass-impurity block ``-h sigma_z``."""
        h = float(h)
        if not np.isfinite(h):
            raise ValueError(f"h must be finite; received {h!r}")
        return np.asarray(-h * SIGMA_Z, dtype=np.complex128)

    def local_onsite_block(self, h: float = 0.0, is_impurity: bool = False) -> DenseMatrix:
        """Return the onsite block for a clean or impurity site."""
        block = self.onsite_block.copy()
        if is_impurity:
            block += self.impurity_block(h)
        return block

    # ------------------------------------------------------------------
    # Bloch-space model
    # ------------------------------------------------------------------
    def bloch_hamiltonian(self, kx: float, ky: float) -> DenseMatrix:
        """Return the 2x2 clean Bloch Hamiltonian at momentum ``(kx, ky)``."""
        kx = float(kx)
        ky = float(ky)
        if not np.isfinite(kx) or not np.isfinite(ky):
            raise ValueError("kx and ky must be finite")

        return np.asarray(
            -self.A * np.sin(kx) * SIGMA_X
            - self.A * np.sin(ky) * SIGMA_Y
            + (self.u + self.t * np.cos(kx) + self.t * np.cos(ky)) * SIGMA_Z,
            dtype=np.complex128,
        )

    def dispersion(self, kx: float, ky: float) -> npt.NDArray[np.float64]:
        """Return the ordered clean energies ``[-E(k), +E(k)]``."""
        eigenvalues = np.linalg.eigvalsh(self.bloch_hamiltonian(kx, ky))
        return np.asarray(eigenvalues, dtype=np.float64)

    # ------------------------------------------------------------------
    # Real-space matrix construction
    # ------------------------------------------------------------------
    def _add_onsite_block(self, matrix, x: int, y: int, block: DenseMatrix) -> None:
        if np.shape(block) != (self.norb, self.norb):
            raise ValueError(f"onsite block must have shape (2, 2); received {np.shape(block)}")
        for a in range(self.norb):
            ia = self.idx(x, y, a)
            for b in range(self.norb):
                ib = self.idx(x, y, b)
                matrix[ia, ib] += block[a, b]

    def _add_hopping_block(
        self,
        matrix,
        source: Site,
        target: Site,
        block: DenseMatrix,
    ) -> None:
        """Add ``c_source^dagger block c_target + h.c.``."""
        if np.shape(block) != (self.norb, self.norb):
            raise ValueError(f"hopping block must have shape (2, 2); received {np.shape(block)}")

        x1, y1 = source
        x2, y2 = target
        for a in range(self.norb):
            i = self.idx(x1, y1, a)
            for b in range(self.norb):
                j = self.idx(x2, y2, b)
                amplitude = block[a, b]
                matrix[i, j] += amplitude
                matrix[j, i] += np.conjugate(amplitude)

    def clean_hamiltonian(self, sparse: bool = False):
        """Construct the finite clean Hamiltonian in dense or CSR format."""
        matrix = _empty_matrix(self.dim, sparse=sparse)

        for y in range(self.Ly):
            for x in range(self.Lx):
                self._add_onsite_block(matrix, x, y, self.onsite_block)

                neighbor_x = self._neighbor(x, y, 1, 0)
                if neighbor_x is not None:
                    self._add_hopping_block(matrix, (x, y), neighbor_x, self.hopping_x)

                neighbor_y = self._neighbor(x, y, 0, 1)
                if neighbor_y is not None:
                    self._add_hopping_block(matrix, (x, y), neighbor_y, self.hopping_y)

        return _finalize_matrix(matrix, sparse=sparse)

    def canonicalize_impurities(self, impurities: Optional[Iterable[Site]]) -> tuple[Site, ...]:
        """Validate, wrap where periodic, and deduplicate an impurity configuration.

        Duplicate sites raise ``ValueError`` because an impurity configuration is
        a set of sites; silently counting the same site twice would change its
        physical strength.
        """
        if impurities is None:
            return ()

        canonical: list[Site] = []
        seen: set[Site] = set()
        for item in impurities:
            if not isinstance(item, (tuple, list, np.ndarray)) or len(item) != 2:
                raise TypeError(f"each impurity must be a two-coordinate site; received {item!r}")
            site = self.normalize_site(item[0], item[1])
            if site in seen:
                raise ValueError(f"duplicate impurity site after boundary wrapping: {site}")
            seen.add(site)
            canonical.append(site)
        return tuple(canonical)

    def impurity_mask(self, impurities: Optional[Iterable[Site]]) -> npt.NDArray[np.bool_]:
        """Return a boolean mask of shape ``(Ly, Lx)`` for a configuration."""
        mask = np.zeros((self.Ly, self.Lx), dtype=bool)
        for x, y in self.canonicalize_impurities(impurities):
            mask[y, x] = True
        return mask

    def onsite_mass_profile(
        self,
        impurities: Optional[Iterable[Site]],
        h: float,
    ) -> npt.NDArray[np.float64]:
        """Return the scalar sigma_z mass ``u - h n_R`` on every site."""
        h = float(h)
        if not np.isfinite(h):
            raise ValueError(f"h must be finite; received {h!r}")
        profile = np.full((self.Ly, self.Lx), self.u, dtype=np.float64)
        profile[self.impurity_mask(impurities)] -= h
        return profile

    def impurity_potential(
        self,
        impurities: Optional[Iterable[Site]],
        h: float,
        sparse: bool = False,
    ):
        """Construct ``V = sum_R (-h sigma_z)_R`` in dense or CSR format."""
        matrix = _empty_matrix(self.dim, sparse=sparse)
        sites = self.canonicalize_impurities(impurities)
        if not sites:
            return _finalize_matrix(matrix, sparse=sparse)

        block = self.impurity_block(h)
        for x, y in sites:
            self._add_onsite_block(matrix, x, y, block)
        return _finalize_matrix(matrix, sparse=sparse)

    def hamiltonian(
        self,
        impurities: Optional[Iterable[Site]] = None,
        h: float = 0.0,
        sparse: bool = False,
    ):
        """Construct the full Hamiltonian ``H = H0 + V_imp``."""
        sites = self.canonicalize_impurities(impurities)
        clean = self.clean_hamiltonian(sparse=sparse)
        if not sites or float(h) == 0.0:
            return clean

        disorder = self.impurity_potential(sites, h=h, sparse=sparse)
        full = clean + disorder
        return full.tocsr() if sparse else np.asarray(full, dtype=np.complex128)

    # ------------------------------------------------------------------
    # Symmetry and numerical diagnostics
    # ------------------------------------------------------------------
    def particle_hole_operator(self, sparse: bool = False):
        """Return the unitary part of ``C = U_C K`` with ``U_C=I_sites⊗sigma_x``."""
        if sparse:
            if sp is None:
                raise ImportError("scipy is required when sparse=True")
            return sp.kron(sp.eye(self.n_sites, format="csr"), sp.csr_matrix(SIGMA_X), format="csr")
        return np.kron(np.eye(self.n_sites, dtype=np.complex128), SIGMA_X)

    @staticmethod
    def hermiticity_residual(matrix) -> float:
        """Return ``max|H-H^dagger|`` for a dense or sparse matrix."""
        if sp is not None and sp.issparse(matrix):
            return _max_abs(matrix - matrix.getH())
        array = np.asarray(matrix)
        return _max_abs(array - array.conj().T)

    @staticmethod
    def check_hermitian(matrix, atol: float = 1.0e-10) -> bool:
        """Whether ``matrix`` is Hermitian within absolute tolerance ``atol``."""
        return QWZModel.hermiticity_residual(matrix) <= float(atol)

    def particle_hole_residual(self, matrix) -> float:
        """Return ``max|U_C H* U_C^dagger + H|`` for class-D particle-hole symmetry."""
        if sp is not None and sp.issparse(matrix):
            operator = self.particle_hole_operator(sparse=True)
            transformed = operator @ matrix.conjugate() @ operator.getH()
            return _max_abs(transformed + matrix)

        array = np.asarray(matrix, dtype=np.complex128)
        operator = self.particle_hole_operator(sparse=False)
        transformed = operator @ array.conjugate() @ operator.conj().T
        return _max_abs(transformed + array)

    def check_particle_hole(self, matrix, atol: float = 1.0e-10) -> bool:
        """Whether the matrix satisfies ``C H C^-1 = -H`` within ``atol``."""
        if matrix.shape != (self.dim, self.dim):
            raise ValueError(
                f"matrix shape {matrix.shape} is incompatible with model dimension {self.dim}"
            )
        return self.particle_hole_residual(matrix) <= float(atol)

    def clean_spectrum_from_bloch_grid(self) -> npt.NDArray[np.float64]:
        """Return the clean PBC spectrum from the allowed discrete momenta.

        Both axes must be periodic.  This method is intended as a small-system
        validation of the real-space matrix convention.
        """
        if not (self.pbc_x and self.pbc_y):
            raise ValueError("Bloch-grid spectrum requires periodic boundaries in both axes")

        energies: list[float] = []
        for nx in range(self.Lx):
            kx = 2.0 * np.pi * nx / self.Lx
            for ny in range(self.Ly):
                ky = 2.0 * np.pi * ny / self.Ly
                energies.extend(self.dispersion(kx, ky))
        return np.sort(np.asarray(energies, dtype=np.float64))

    def realspace_bloch_spectrum_residual(self) -> float:
        """Maximum clean-spectrum mismatch between real and Bloch constructions."""
        realspace = np.linalg.eigvalsh(self.clean_hamiltonian(sparse=False))
        bloch = self.clean_spectrum_from_bloch_grid()
        return float(np.max(np.abs(np.sort(realspace) - bloch)))

    def run_sanity_checks(
        self,
        impurities: Optional[Sequence[Site]] = None,
        h: float = 0.0,
        atol: float = 1.0e-10,
        check_bloch_spectrum: bool = True,
    ) -> dict[str, float | bool]:
        """Run lightweight consistency checks and return machine-readable results."""
        dense = self.hamiltonian(impurities=impurities, h=h, sparse=False)
        results: dict[str, float | bool] = {
            "hermiticity_residual": self.hermiticity_residual(dense),
            "is_hermitian": self.check_hermitian(dense, atol=atol),
            "particle_hole_residual": self.particle_hole_residual(dense),
            "has_particle_hole_symmetry": self.check_particle_hole(dense, atol=atol),
        }

        if check_bloch_spectrum and self.pbc_x and self.pbc_y:
            residual = self.realspace_bloch_spectrum_residual()
            results["realspace_bloch_spectrum_residual"] = residual
            results["realspace_bloch_spectrum_matches"] = residual <= float(atol)

        if sp is not None:
            sparse_matrix = self.hamiltonian(impurities=impurities, h=h, sparse=True)
            dense_sparse_residual = _max_abs(dense - sparse_matrix.toarray())
            results["dense_sparse_residual"] = dense_sparse_residual
            results["dense_sparse_match"] = dense_sparse_residual <= float(atol)

        return results


# Module-level compatibility helper.  New code may call either
# ``QWZModel.check_hermitian(H)`` or ``check_hermitian(H)``.
def check_hermitian(matrix, atol: float = 1.0e-10) -> bool:
    """Whether a dense or sparse matrix is Hermitian within ``atol``."""
    return QWZModel.check_hermitian(matrix, atol=atol)


if __name__ == "__main__":
    model = QWZModel(Lx=6, Ly=5, t=1.0, u=-2.05, A=1.5, pbc=True)
    demo_impurities = [(0, 0), (3, 2)]
    checks = model.run_sanity_checks(
        impurities=demo_impurities,
        h=-1.7,
        check_bloch_spectrum=True,
    )

    print("QWZ model self-check")
    print(f"dimension: {model.dim}")
    for key, value in checks.items():
        print(f"{key}: {value}")
