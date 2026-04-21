"""
Core numerical utilities shared by the γCDM verification pipeline.

Everything in this module is pure-Python/NumPy and has NO side effects on
global state. It is safe to import from any notebook script (MLE, MCMC,
nested sampling) without risk of divergent numerics between samplers.

Functions
---------
h0_local            Implied local H0 from a model amplitude A evaluated at z_pivot.
apply_correction    Dispatch table (model_type → Δμ(z) function).
build_mu_th         Assemble theoretical μ from CAMB baseline + offsets + correction.
build_err_eff       Compute σ_eff = √(σ_obs² + σ_int²) respecting combined/single mode.
ExceptionCounter    Lightweight tracked counter for CAMB/optimizer failures.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# H0_local helper
# ─────────────────────────────────────────────────────────────────────────────
def h0_local(H0_cosmo: float,
             A: float = 0.0,
             z_b: float = 1.0,
             gamma_0: float = 0.0,
             z_h: float = 1e10,
             z_pivot: float = 0.0) -> float:
    """Implied local H0 given the γCDM-LOG²-Decay correction.

    Derivation (see review notes, April 2026)
    -----------------------------------------
    The full correction is

        Δμ(z) = A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h)

    An additive offset Δμ on the distance modulus is equivalent to rescaling
    d_L → d_L · 10^(Δμ/5). In the linear Hubble regime d_L ≈ c·z/H₀, so an
    observer inferring H₀ from low-z SNe would get

        H₀_local = H₀_cosmo · 10^(-Δμ(z_pivot)/5)

    Parameters
    ----------
    H0_cosmo : float
        Cosmological (high-z) Hubble constant inferred from the fit.
    A, z_b : float
        Amplitude and decay scale of the local-bubble component.
        Set A=0 to suppress the bubble term entirely.
    gamma_0, z_h : float
        Amplitude and decay scale of the long-range LOG² (Kerr) component.
        Its contribution at z_pivot ≈ 0.02 is ~1e-3·|γ₀| mag and
        typically negligible compared to A. Included for completeness.
    z_pivot : float, default 0.0
        Redshift at which the "local" H₀ is evaluated.
        - z_pivot = 0.0       → strict limit, H0_cosmo · 10^(-A/5).
        - z_pivot = 0.023     → effective pivot of the Cepheid+SN anchor
                                (Riess+22). For z_b ≳ 0.1 the difference
                                vs. the strict limit is < 1%.

    Notes
    -----
    To preserve the pre-April-2026 behaviour (strict
    z→0) must pass z_pivot=0.0 (the default). 
    To compare against SH0ES rigorously should pass
    cosmo_constants.SH0ES_Z_PIVOT.
    """
    if z_b <= 0.0:
        bubble = 0.0
    else:
        bubble = A * np.exp(-z_pivot / z_b)
    if z_h <= 0.0:
        kerr = 0.0
    else:
        kerr = gamma_0 * np.log1p(z_pivot)**2 * np.exp(-z_pivot / z_h)
    dmu_pivot = bubble + kerr
    return H0_cosmo * 10.0 ** (-dmu_pivot / 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Correction-function dispatch (model_type → f(z, **params))
# ─────────────────────────────────────────────────────────────────────────────
def correction_lcdm(z: np.ndarray, **_) -> np.ndarray:
    """ΛCDM baseline: no correction."""
    return np.zeros_like(z)


def correction_constant(z: np.ndarray, gamma: float = 0.0, **_) -> np.ndarray:
    """γCDM (constant): Δμ = γ · ln(1+z)."""
    return gamma * np.log1p(z)


def correction_linear(z: np.ndarray, gamma_0: float = 0.0, **_) -> np.ndarray:
    """γCDM-LINEAR: Δμ = γ₀ · (1+z) · ln(1+z)."""
    return gamma_0 * (1.0 + z) * np.log1p(z)


def correction_log2(z: np.ndarray, gamma_0: float = 0.0, **_) -> np.ndarray:
    """γCDM-LOG²: Δμ = γ₀ · [ln(1+z)]²."""
    return gamma_0 * np.log1p(z) ** 2


def correction_log3(z: np.ndarray, gamma_0: float = 0.0, **_) -> np.ndarray:
    """γCDM-LOG³: Δμ = γ₀ · [ln(1+z)]³."""
    return gamma_0 * np.log1p(z) ** 3


def correction_decay(z: np.ndarray, A: float = 0.0, zd: float = 1.0, **_) -> np.ndarray:
    """γCDM-Decay: Δμ = A · exp(-z/zd)."""
    return A * np.exp(-z / zd)


def correction_log_decay(z: np.ndarray,
                         A: float = 0.0, z_b: float = 1.0,
                         gamma_0: float = 0.0, z_h: float = 1e10,
                         no_bubble: bool = False,
                         **_) -> np.ndarray:
    """γCDM-LOG²-Decay: Δμ = A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h).

    With no_bubble=True the local term vanishes (Kerr-Only variant).
    """
    bubble = 0.0 if no_bubble else A * np.exp(-z / z_b)
    kerr = gamma_0 * np.log1p(z) ** 2 * np.exp(-z / z_h)
    return bubble + kerr


CORRECTION_FNS = {
    "lcdm":      correction_lcdm,
    "gcdm":      correction_constant,
    "linear":    correction_linear,
    "log2":      correction_log2,
    "log3":      correction_log3,
    "decay":     correction_decay,
    "log_decay": correction_log_decay,
}


# ─────────────────────────────────────────────────────────────────────────────
# μ-theoretical assembler (handles combined vs. single calibration mode)
# ─────────────────────────────────────────────────────────────────────────────
def build_mu_th(mu_base: np.ndarray,
                correction: np.ndarray,
                combined_mode: bool,
                sne_mask: np.ndarray | None,
                M_sne: float = 0.0,
                M_qso: float = 0.0,
                M_single: float = 0.0) -> np.ndarray:
    """μ_th = mu_base + calibration_offset + Δμ_correction.

    In combined mode (SNe + QSO) the offset is piecewise (M_sne for SNe
    entries, M_qso for QSO entries). In single-sample mode (either SNe-only
    or QSO-only) a single M_single is added uniformly.
    """
    if combined_mode and sne_mask is not None:
        offset = np.where(sne_mask, M_sne, M_qso)
        return mu_base + offset + correction
    return mu_base + M_single + correction


# ─────────────────────────────────────────────────────────────────────────────
# Effective error assembler (σ_obs ⊕ σ_int)
# ─────────────────────────────────────────────────────────────────────────────
def build_err_eff(err_mu: np.ndarray,
                  combined_mode: bool,
                  sne_mask: np.ndarray | None,
                  sigma_int_sne: float,
                  sigma_int_qso: float,
                  sigma_int_single: float | None = None) -> np.ndarray:
    """σ_eff = √(σ_obs² + σ_int²) with the appropriate per-point σ_int."""
    if combined_mode and sne_mask is not None:
        sig_int = np.where(sne_mask, sigma_int_sne, sigma_int_qso)
    else:
        val = (sigma_int_single if sigma_int_single is not None
               else sigma_int_sne)
        sig_int = np.full_like(err_mu, val)
    return np.sqrt(err_mu ** 2 + sig_int ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Cosmology prior / physical-bounds helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_Omega_m(H0: float, omch2: float, ombh2: float = 0.0224) -> float:
    """Ωm = (Ωch² + Ωbh²)/h²."""
    h = H0 / 100.0
    return (omch2 + ombh2) / h ** 2


def check_physical_prior(H0: float, omch2: float, ombh2: float = 0.0224) -> bool:
    """Enforce physical flat ΛCDM: 0 < Ωm < 1 (so that ΩΛ ≥ 0)."""
    Om = compute_Omega_m(H0, omch2, ombh2)
    return 0.0 < Om < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Exception counter
# ─────────────────────────────────────────────────────────────────────────────
class ExceptionCounter:
    """Lightweight, thread-unsafe counter for CAMB/optimizer silent failures.

    Use as a context decorator:

        camb_err = ExceptionCounter("CAMB background")
        ...
        try:
            ... = camb.get_background(...)
        except Exception as e:
            camb_err.record(e)
            return 1e10
        ...
        print(camb_err.summary())
    """
    def __init__(self, label: str):
        self.label = label
        self.count = 0
        self.last_msg: str | None = None

    def record(self, exc: BaseException | str) -> None:
        self.count += 1
        self.last_msg = str(exc)[:200]

    def summary(self) -> str:
        if self.count == 0:
            return f"   ✓ {self.label}: 0 failures"
        tail = f" (last: {self.last_msg})" if self.last_msg else ""
        return f"   ⚠ {self.label}: {self.count} silent failures{tail}"

    def __int__(self) -> int:
        return self.count
