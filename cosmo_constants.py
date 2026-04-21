"""
Cosmological constants and fiducial values used across the γCDM verification
pipeline.

Single source of truth for physical constants, external-data calibrations and
prior bounds. All modules in notebooks/ must import from here rather than
hard-coding numerical values.

References
----------
- Planck 2018 shift parameter (R):  Aghanim et al. (2020), A&A 641, A6, Table 2
- SH0ES local H0:                   Riess et al. (2022), ApJ 934 L7
- Pantheon+ calibration residual:   Brout et al. (2022), arXiv:2202.04077
- Lusso+20 QSO Hubble diagram:      Lusso et al. (2020), A&A 642 A150
- Fiducial ombh2:                   Cooke+18 / Planck 2018 baseline
"""

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT_KMS = 299792.458            # Speed of light [km/s]

# ─────────────────────────────────────────────────────────────────────────────
# CMB / PLANCK 2018
# ─────────────────────────────────────────────────────────────────────────────
Z_STAR = 1089.92                    # CMB decoupling redshift (Planck 2018)
R_PLANCK = 1.7502                   # Shift parameter R at z*
SIGMA_R_PLANCK = 0.0046             # 1-sigma uncertainty on R
SIGMA_CORR_CMB = 0.02               # Tolerance for Δμ(z*) penalty [mag]

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL H0 / SH0ES
# ─────────────────────────────────────────────────────────────────────────────
SH0ES_H0 = 73.04                    # Riess et al. (2022) central value
SH0ES_SIG = 1.04                    # 1-sigma uncertainty on local H0
SH0ES_Z_PIVOT = 0.023               # Effective pivot redshift of Cepheid+SN sample
                                    # (set Z_PIVOT to 0.0 to recover the z→0 limit
                                    # H0_local = H0_cosmo · 10^(-A/5))
#Z_PIVOT = SH0ES_Z_PIVOT
Z_PIVOT = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# FIDUCIAL COSMOLOGY
# ─────────────────────────────────────────────────────────────────────────────
OMBH2_FIDUCIAL = 0.0224             # Planck 2018 baryon density (fixed)
PLANCK_H0 = 67.4                    # Planck ΛCDM H0 (for fixed-anchor runs)
PLANCK_OMCH2 = 0.12                 # Planck ΛCDM cold-DM density

# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING-PARAMETER PRIOR BOUNDS (shared across all MLE / MCMC / NS runs)
# ─────────────────────────────────────────────────────────────────────────────
H0_MIN, H0_MAX = 40.0, 100.0
OMCH2_MIN, OMCH2_MAX = 0.01, 0.35
M_MIN, M_MAX = -3.0, 3.0
GAMMA_MIN, GAMMA_MAX = -3.0, 3.0
SIGMA_INT_MIN, SIGMA_INT_MAX = 0.0, 2.0
A_MIN, A_MAX = -3.0, 3.0
ZD_MIN, ZD_MAX = 0.01, 10.0
ZB_MIN, ZB_MAX = 0.01, 10.0         # Local bubble decay scale
ZH_MIN, ZH_MAX = 0.01, 1e10         # Long-range Kerr decay scale

# Intrinsic-scatter fit ranges
SINT_SNE_MIN, SINT_SNE_MAX = 0.001, 0.5
SINT_QSO_MIN, SINT_QSO_MAX = 0.1, 3.0

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION PRIORS (Gaussian on δM, externally justified)
# ─────────────────────────────────────────────────────────────────────────────
M_SNE_PRIOR_SCALE = 0.05            # Pantheon+ residual calibration ~0.02–0.04 mag
M_QSO_PRIOR_SCALE = 0.15            # Lusso+20 L_X-L_UV intercept ~0.1–0.2 mag

# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN PENALTY ON M (only active when --penalty-m is passed)
# ─────────────────────────────────────────────────────────────────────────────
PENALTY_M_SIGMA = 0.1               # σ for M-penalty Gaussian prior

# ─────────────────────────────────────────────────────────────────────────────
# BAO / DESI DR1 (optional, enabled via --bao or --bao-null)
# ─────────────────────────────────────────────────────────────────────────────
# See notebooks/bao_desi_dr1.py for data + χ² computation.
# The modification Δμ(z) propagates to D_M (and hence D_V) as
#     D_M_model(z) = D_M_LCDM(z) · 10^(Δμ(z)/5)
# D_H = c/H(z) stays ΛCDM (the μ-based correction does not prescribe H(z)).
BAO_PROPAGATE_CORRECTION = False     
# False → BAO sees only the standard ΛCDM background
# (interpretation: the γCDM correction affects inferred luminosity distances, 
# but is not assumed to propagate to angular-diameter / ruler-based BAO observables).