"""
DESI DR1 BAO compilation and χ² computation for the γCDM pipeline.

Source
------
DESI Collaboration 2024, arXiv:2404.03002 (Year-1 BAO results).
Table 1 of that paper gives 12 data points across 7 tracer bins:
  · BGS  (z_eff=0.295):      D_V/r_d only (isotropic)
  · LRG1 (z_eff=0.510):      D_M/r_d + D_H/r_d
  · LRG2 (z_eff=0.706):      D_M/r_d + D_H/r_d
  · LRG3+ELG1 (z_eff=0.930): D_M/r_d + D_H/r_d
  · ELG2 (z_eff=1.317):      D_M/r_d + D_H/r_d
  · QSO  (z_eff=1.491):      D_V/r_d only (isotropic)
  · Lyα  (z_eff=2.330):      D_M/r_d + D_H/r_d

Within-tracer correlations (D_M/r_d ↔ D_H/r_d) are taken from DESI DR1 Table 1.
Cross-tracer correlations are negligible for Year-1 data and ignored here.

Observables
-----------
  D_M(z) :  transverse comoving distance [Mpc]  = D_L(z)/(1+z) for flat FRW
  D_H(z) :  Hubble distance [Mpc]                = c/H(z)
  D_V(z) :  spherically-averaged distance [Mpc]  = (z · D_M² · D_H)^(1/3)
  r_d    :  sound horizon at drag [Mpc]          (from CAMB.get_derived_params())

γCDM phenomenological modification
----------------------------------
In the γCDM/γCDM-LOG²-Decay framework the model predicts an additive
correction Δμ(z) on the distance modulus. Since μ = 5·log₁₀(D_L/Mpc) + 25
and D_M = D_L/(1+z) for flat FRW, the corresponding BAO prediction is

    D_M_model(z) = D_M_LCDM(z) · 10^(Δμ(z)/5)

The Hubble distance D_H(z) = c/H(z) is NOT directly prescribed by the μ-based
correction, so we keep D_H = D_H_LCDM. For D_V we use the same D_M correction:

    D_V_model(z) = D_V_LCDM(z) · 10^(2·Δμ(z)/15)

This is the "μ-only propagation" interpretation — the strictest self-consistency
test of the model (γ and A parameters enter BAO via the D_M channel).

Public API
----------
    Z_EFF_DESI                 ordered unique z_eff values (len == 7)
    DESI_DR1                   list of 12 data rows (z, obs, val, err, tracer)
    build_desi_cov_inv(diag)   returns inverse covariance, 12×12
    compute_chi2_bao(...)      evaluates χ²_BAO against the full 12-point vector
"""
from __future__ import annotations

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DESI DR1 DATA (arXiv:2404.03002, Table 1)
# ─────────────────────────────────────────────────────────────────────────────
# Each row: (z_eff, observable, value, error, tracer_tag)
#   observable ∈ {"DV", "DM", "DH"}  — all normalized by r_d.
# ─────────────────────────────────────────────────────────────────────────────
DESI_DR1: list[tuple[float, str, float, float, str]] = [
    (0.295, "DV",  7.93, 0.15, "BGS"),
    (0.510, "DM", 13.62, 0.25, "LRG1"),
    (0.510, "DH", 20.98, 0.61, "LRG1"),
    (0.706, "DM", 16.85, 0.32, "LRG2"),
    (0.706, "DH", 20.08, 0.60, "LRG2"),
    (0.930, "DM", 21.71, 0.28, "LRG3+ELG1"),
    (0.930, "DH", 17.88, 0.35, "LRG3+ELG1"),
    (1.317, "DM", 27.79, 0.69, "ELG2"),
    (1.317, "DH", 13.82, 0.42, "ELG2"),
    (1.491, "DV", 26.07, 0.67, "QSO"),
    (2.330, "DM", 39.71, 0.94, "Lya"),
    (2.330, "DH",  8.52, 0.17, "Lya"),
]

# Pearson correlations between (D_M/r_d, D_H/r_d) within the same tracer,
# taken from DESI DR1 Table 1. Only defined for anisotropic tracers.
DESI_DR1_RHO: dict[str, float] = {
    "LRG1":      -0.45,
    "LRG2":      -0.41,
    "LRG3+ELG1": -0.39,
    "ELG2":      -0.44,
    "Lya":       -0.48,
}

# Ordered unique z_eff values. The BAO chi2 helper uses this ordering to map
# predicted (D_M, D_H) arrays back into the 12-point DESI vector.
Z_EFF_DESI: np.ndarray = np.array(sorted({row[0] for row in DESI_DR1}))
N_BAO_POINTS: int = len(DESI_DR1)


def build_desi_cov_inv(diagonal: bool = False) -> np.ndarray:
    """Inverse of the 12×12 DESI DR1 covariance matrix.

    Parameters
    ----------
    diagonal : bool
        If True, ignores the (D_M, D_H) within-tracer correlations and uses
        only per-point variances. Default False (full correlations active).

    Returns
    -------
    C_inv : (12, 12) ndarray
    """
    N = N_BAO_POINTS
    C = np.zeros((N, N), dtype=float)
    for i in range(N):
        C[i, i] = DESI_DR1[i][3] ** 2

    if not diagonal:
        # Group rows by tracer to find the (D_M, D_H) pair indices
        by_tracer: dict[str, list[tuple[int, str, float]]] = {}
        for i, (_z, obs, _val, err, tr) in enumerate(DESI_DR1):
            by_tracer.setdefault(tr, []).append((i, obs, err))
        for tr, rows in by_tracer.items():
            if tr in DESI_DR1_RHO and len(rows) == 2:
                (i1, _o1, e1), (i2, _o2, e2) = rows
                rho = DESI_DR1_RHO[tr]
                C[i1, i2] = C[i2, i1] = rho * e1 * e2

    return np.linalg.inv(C)


# Pre-computed inverse covariance (full correlations) — cached at import time
# since it never changes. Cost is one 12×12 inversion; trivial.
_CINV_FULL: np.ndarray = build_desi_cov_inv(diagonal=False)


def compute_chi2_bao(dm_at_zeff: np.ndarray,
                     dh_at_zeff: np.ndarray,
                     rdrag: float,
                     *,
                     cov_inv: np.ndarray | None = None) -> tuple[float, int]:
    """Evaluate the DESI DR1 BAO χ² for the given cosmological background.

    Parameters
    ----------
    dm_at_zeff : (7,) ndarray
        Transverse comoving distance [Mpc], ordered by Z_EFF_DESI. Should
        include any γCDM correction via the caller (see module docstring).
    dh_at_zeff : (7,) ndarray
        Hubble distance c/H(z) [Mpc], ordered by Z_EFF_DESI.
    rdrag : float
        Sound horizon at drag [Mpc]. From CAMB.get_derived_params()["rdrag"].
    cov_inv : optional (12, 12) ndarray
        Override for the DESI covariance inverse (e.g. diagonal approximation).

    Returns
    -------
    chi2 : float  — residualᵀ · C⁻¹ · residual (no log|C| term)
    ndof : int    — number of data points (12). Does NOT subtract fit params.
    """
    if cov_inv is None:
        cov_inv = _CINV_FULL

    model = np.empty(N_BAO_POINTS, dtype=float)
    # Map z_eff value to its index in Z_EFF_DESI
    z_to_idx = {z: i for i, z in enumerate(Z_EFF_DESI)}

    for row_i, (z, obs, _val, _err, _tr) in enumerate(DESI_DR1):
        k = z_to_idx[z]
        DM = dm_at_zeff[k]
        DH = dh_at_zeff[k]
        if obs == "DV":
            model[row_i] = (z * DM * DM * DH) ** (1.0 / 3.0) / rdrag
        elif obs == "DM":
            model[row_i] = DM / rdrag
        elif obs == "DH":
            model[row_i] = DH / rdrag
        else:
            raise ValueError(f"Unknown BAO observable '{obs}' in DESI_DR1 row {row_i}")

    obs_vec = np.array([row[2] for row in DESI_DR1])
    resid = obs_vec - model
    chi2 = float(resid @ cov_inv @ resid)
    return chi2, N_BAO_POINTS


# Convenience for null-test reporting: per-point labels ("BGS D_V", "LRG1 D_M", ...)
def point_labels() -> list[str]:
    """Short labels 'tracer obs' for each of the 12 DESI DR1 points."""
    pretty = {"DV": "D_V", "DM": "D_M", "DH": "D_H"}
    return [f"{row[4]} {pretty[row[1]]}" for row in DESI_DR1]
