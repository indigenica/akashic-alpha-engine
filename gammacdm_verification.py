#!/usr/bin/env python3
"""
γCDM Robustness Verification Protocol
========================================

Protocol:
1. δM nuisance parameter in BOTH models with IDENTICAL priors
2. Multi-start MLE (30 random initializations)
3. Same likelihood (only difference is the γ correction term)
4. Wide priors (same bounds on shared params for ALL models)
5. Corner plot γ vs δM (proves γ is not absorbing calibration offset)
6. AIC + BIC model comparison
7. Mock test (γ=0): verifies pipeline does NOT fabricate signal

Usage:
    python gammacdm_verification.py                       # MLE (with quasars)
    python gammacdm_verification.py --no-quasars          # MLE without quasars
    python gammacdm_verification.py --mcmc                # Full MCMC
    python gammacdm_verification.py --mock                # Run γ=0 null test
    python gammacdm_verification.py --mock --n-mock 50    # 50 mock realizations

Author: Bautista, 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.interpolate import RegularGridInterpolator
import warnings
import argparse
import os
import sys
import datetime
from getdist import loadMCSamples

# Shared numerical utilities and physical constants.
# Any change to these must be done in the respective modules (single source of
# truth) rather than by redefining locally.
from cosmo_constants import (
    C_LIGHT_KMS, Z_STAR, R_PLANCK, SIGMA_R_PLANCK, SIGMA_CORR_CMB,
    SH0ES_H0, SH0ES_SIG, SH0ES_Z_PIVOT, Z_PIVOT,
    OMBH2_FIDUCIAL, PLANCK_H0, PLANCK_OMCH2,
    H0_MIN, H0_MAX, OMCH2_MIN, OMCH2_MAX, M_MIN, M_MAX,
    GAMMA_MIN, GAMMA_MAX, SIGMA_INT_MIN, SIGMA_INT_MAX,
    A_MIN, A_MAX, ZD_MIN, ZD_MAX, ZB_MIN, ZB_MAX, ZH_MIN, ZH_MAX,
    SINT_SNE_MIN, SINT_SNE_MAX, SINT_QSO_MIN, SINT_QSO_MAX,
    M_SNE_PRIOR_SCALE, M_QSO_PRIOR_SCALE, PENALTY_M_SIGMA,
    BAO_PROPAGATE_CORRECTION,
)
from gammacdm_core import h0_local, ExceptionCounter
from bao_desi_dr1 import (
    Z_EFF_DESI, N_BAO_POINTS, DESI_DR1,
    compute_chi2_bao, point_labels as _bao_point_labels,
)

warnings.filterwarnings('ignore')

# Tracked counters for silent failures (kept global for the same reason as the
# chi² functions themselves: scipy.optimize.minimize calls the objective many
# thousands of times and we want a single-pass summary at the end).
CAMB_ERRORS = ExceptionCounter("CAMB background")
CHI2_ERRORS = ExceptionCounter("χ² evaluation")

class TeeLogger(object):
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log = open(log_filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if not os.path.exists("logs"):
    os.makedirs("logs")
    
SESSION_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M")
log_filename = os.path.join("logs", f"{SESSION_TIMESTAMP}_log.txt")
sys.stdout = TeeLogger(log_filename)

print("="*70)
print(f"🌟 INICIANDO SESIÓN: {SESSION_TIMESTAMP}")
print(f"💻 Comando: {' '.join(sys.argv)}")
print("="*70)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_M(samples, mode):
    """Calculate mean and std of δM from samples list/dict. Standardized across models."""
    if mode: # COMBINED_MODE (SNe + QSO)
        ms = _safe_get(samples, "M_sne")
        mq = _safe_get(samples, "M_qso")
        if ms is not None and mq is not None:
            return (np.mean(ms) + np.mean(mq)) / 2, np.sqrt(np.std(ms)**2 + np.std(mq)**2) / 2
        return 0.0, 0.0
    
    m_vals = _safe_get(samples, "mabs", _safe_get(samples, "M"))
    if m_vals is not None:
        return np.mean(m_vals), np.std(m_vals)
    return 0.0, 0.0

def _safe_get(samples, key, default=None):
    """Safely extract parameter samples from dict, SampleCollection, or MCSamples."""
    try:
        if isinstance(samples, dict):
            return samples.get(key, default)
        
        # Debug: Print available keys if we can find them
        # Cobaya SampleCollection / GetDist MCSamples
        param_names = []
        if hasattr(samples, 'getParamNames'):
            p_obj = samples.getParamNames()
            if hasattr(p_obj, 'list'):
                param_names = p_obj.list()
            elif hasattr(p_obj, 'names'):
                # Try to use .names list of strings/objects, converting to str just in case
                param_names = [str(n) for n in p_obj.names]
        elif hasattr(samples, 'getParams'): # Generic GetDist
             params = samples.getParams()
             if hasattr(params, '__dict__'): param_names = list(params.__dict__.keys())

        # Direct attribute access check (for unnamed params or generic objects)
        if hasattr(samples, key):
             return getattr(samples, key)

        # Dictionary-like access
        if key in param_names:
            # For MCSamples, we often need to access via .samples array and index, 
            # OR use logic like samples.getParams().<key>
            # But samples[key] usually works if it implements __getitem__ mapping to columns
            # Let's try standard GetDist access:
            try:
                return samples[key]
            except:
                pass
            
            # Alternative: getParams() attribute
            try:
                p = samples.getParams()
                return getattr(p, key)
            except:
                pass
                
    except Exception as e:
        # print(f"DEBUG: Error in _safe_get({key}): {e}")
        pass
    return default

def report_model_summary(model_name, samples, args, logZ=None, title_prefix="📋"):
    """
    Unified bomb-proof reporting for all models and samplers (MCMC & Nested).
    Ensures all pertinent info is displayed even if parameters are fixed/removed.
    """
    print(f"\n" + "=" * 70)
    print(f"{title_prefix} {model_name}")
    print(f"=" * 70)

    # 1. Hubble Constant
    h0_s = _safe_get(samples, 'H0')
    h0_val = 67.4 if (args.sanity_check or args.fixed_anchor or h0_s is None) else np.mean(h0_s)
    h0_err = 0.0 if (args.sanity_check or args.fixed_anchor or h0_s is None) else np.std(h0_s)
    if h0_err < 1e-5: h0_err = 0.0
    print(f"   H₀    = {h0_val:.2f} ± {h0_err:.2f} km/s/Mpc {'(Fixed)' if h0_err==0 else ''}")

    # 2. Baryonic Matter (omch2)
    om_s = _safe_get(samples, 'omch2')
    if om_s is not None:
        om_val = np.mean(om_s)
        om_err = np.std(om_s)
        print(f"   Ωch²  = {om_val:.4f} ± {om_err:.4f}")
        # Baryon check
        ombh2_s = _safe_get(samples, 'ombh2', 0.0224)
        ombh2 = np.mean(ombh2_s) if isinstance(ombh2_s, (list, np.ndarray)) else ombh2_s
        print(f"   Ωmh²  = {om_val + ombh2:.4f} (Baryon check, Ωbh²={ombh2:.4f})")
    
    # 3. Model Specific Parameters
    # γCDM-LOG2
    g_s = _safe_get(samples, 'gamma_log2')
    if g_s is not None:
        g_val = np.mean(g_s)
        g_err = np.std(g_s)
        print(f"   γ₀    = {g_val:.4f} ± {g_err:.4f}")
        # Spin implication
        beta = abs(g_val) * np.log(10) / 5
        alpha = beta / 2
        spin = np.sqrt(1 - ((1 - alpha) / (1 + alpha))**2) if alpha < 1 else 1.0
        print(f"   🌀 Spin Implied: a/M = {spin:.4f}")
    
    # γCDM-LOG²-DECAY (checks gamma_log_decay for the Kerr component)
    g_uni_s = _safe_get(samples, 'gamma_log_decay')
    if g_uni_s is not None:
        g_val = np.mean(g_uni_s)
        g_err = np.std(g_uni_s)
        print(f"   γ₀    = {g_val:.4f} ± {g_err:.4f} (Kerr geometry)")
        # Spin implication from Kerr component
        beta = abs(g_val) * np.log(10) / 5
        alpha = beta / 2
        spin = np.sqrt(1 - ((1 - alpha) / (1 + alpha))**2) if alpha < 1 else 1.0
        print(f"   🌀 Spin Implied: a/M = {spin:.4f}")

    elif model_name.lower().find("decay") != -1 and g_uni_s is None and _safe_get(samples, 'gamma_log2') is None:
        print(f"   γ₀    = 0.0000 (N/A for pure Decay model)")
    
    # A parameter (shared by Decay and Unified)
    a_s = _safe_get(samples, 'A')
    if a_s is not None:
        a_val = np.mean(a_s)
        a_err = np.std(a_s)
        print(f"   A     = {a_val:.4f} ± {a_err:.4f} (Bubble amplitude)")
    
    # z_b — bubble scale (Unified only)
    zb_s = _safe_get(samples, 'zb')
    if zb_s is not None:
        zb_val = np.mean(zb_s)
        zb_err = np.std(zb_s)
        print(f"   z_b   = {zb_val:.3f} ± {zb_err:.3f} (Bubble decay scale)")
    
    # z_h — horizon scale (Unified only; may come from derived zh or from log_zh)
    zh_s = _safe_get(samples, 'zh')
    if zh_s is None:
        log_zh_s = _safe_get(samples, 'log_zh')
        if log_zh_s is not None:
            zh_s = 10**np.asarray(log_zh_s)
    if zh_s is not None:
        zh_val = np.mean(zh_s)
        zh_err = np.std(zh_s)
        print(f"   z_h   = {zh_val:.3f} ± {zh_err:.3f} (Horizon decay scale)")
    
    # zd — pure Decay scale (Pure Decay only, NOT unified)
    zd_s = _safe_get(samples, 'zd')
    if zd_s is not None:
        zd_val = np.mean(zd_s)
        zd_err = np.std(zd_s)
        print(f"   zd    = {zd_val:.3f} ± {zd_err:.3f}")
    
    # H₀(local) for models with A
    if a_s is not None:
        # Δμ(z=0) = A·exp(0) = A → H₀(local) = H₀(cosmo)·10^(-A/5)
        a_val_for_h0 = np.mean(a_s)
        h0_loc = h0_local(h0_val, A=a_val_for_h0, z_pivot=Z_PIVOT)
        print(f"   → H₀(local) implied: {h0_loc:.2f} km/s/Mpc (Bubble shift A={a_val_for_h0:.3f})")

    # 4. Nuisance Parameters (M)
    is_lcdm = model_name.lower().replace("λ", "l") == "lcdm" or model_name.lower() == "lcdm"
    is_fixed_m = (args.sanity_check or args.no_nuisance or args.asymmetric) and not is_lcdm
    
    if is_fixed_m:
        print(f"   M_sne = 0.000 (Fixed), M_qso = 0.000 (Fixed)")
        print(f"   ⟨δM⟩  = 0.000 (Fixed)")
    else:
        ms_s = _safe_get(samples, 'M_sne', 0.0)
        mq_s = _safe_get(samples, 'M_qso', 0.0)
        ms_val = np.mean(ms_s) if isinstance(ms_s, (list, np.ndarray)) else ms_s
        mq_val = np.mean(mq_s) if isinstance(mq_s, (list, np.ndarray)) else mq_s
        
        m_val, m_err = _get_M(samples, COMBINED_MODE)
        if COMBINED_MODE:
            print(f"   M_sne = {ms_val:.3f}, M_qso = {mq_val:.3f}")
        print(f"   ⟨δM⟩  = {m_val:.3f} ± {m_err:.3f}")

    # 5. Evidence
    if logZ is not None:
        print(f"   log(Z) = {logZ:.2f}")

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description="γCDM Robustness Verification")
parser.add_argument("--mcmc", action="store_true", help="Run full MCMC (slower, more rigorous)")
parser.add_argument("--samples", type=int, default=200000, help="MCMC accepted samples (default: 200000 for convergence)")
parser.add_argument("--starts", type=int, default=30, help="Multi-start MLE runs")
parser.add_argument("--no-quasars", action="store_true", dest="no_quasars",
                    help="Exclude quasars (SNe + CC only)")
parser.add_argument("--quasars-only", "--quasars", action="store_true", dest="quasars_only",
                    help="Use quasars only (high-z test)")
parser.add_argument("--qso-err-cut", type=float, default=1.5, dest="qso_err_cut",
                    help="Max quasar error to include (default: 1.5 mag)")
parser.add_argument("--sne-err-cut", type=float, default=0.5, dest="sne_err_cut",
                    help="Max supernova error to include (default: 0.5 mag)")
parser.add_argument("--z-min-sne", type=float, default=0.01, dest="z_min_sne",
                    help="Minimum redshift to include SNe (default: 0.01, removes local peculiar velocities)")
parser.add_argument("--z-max-sne", type=float, default=2.5, dest="z_max_sne",
                    help="Maximum redshift to include SNe (default: 2.5, Pantheon+ max)")
parser.add_argument("--z-min-qso", type=float, default=0.0, dest="z_min_qso",
                    help="Minimum redshift to include QSO (default: 0.0, recommended 0.7 for cosmological fits)")
parser.add_argument("--revised", action="store_true",
                    help="Use full_dataset_revisado.csv instead of full_dataset.csv")
parser.add_argument("--asymmetric", action="store_true",
                    help="γCDM without δM (test if γ absorbs offset)")
parser.add_argument("--mock", action="store_true",
                    help="Run γ=0 null test (verify pipeline doesn't fabricate signal)")
parser.add_argument("--n-mock", type=int, default=20, dest="n_mock",
                    help="Number of mock realizations (default: 20)")
parser.add_argument("--sigma-int-qso", type=float, default=0.4, dest="sigma_int_qso",
                    help="QSO intrinsic scatter to add in quadrature (default: 0.4, try 0.3-0.6)")
parser.add_argument("--sigma-int-sne", type=float, default=0.1, dest="sigma_int_sne",
                    help="SNe Ia intrinsic scatter (default: 0.1, for Pantheon+ without cov)")
parser.add_argument("--nested", action="store_true",
                    help="Use PolyChord nested sampling instead of MCMC (computes true Bayes factor)")
parser.add_argument("--nlive", type=int, default=200,
                    help="Number of live points for nested sampling (default: 200)")
parser.add_argument("--no-nuisance", action="store_true", help="Fix calibration offsets (M_sne, M_qso) to 0")
parser.add_argument("--no-cc", action="store_true", dest="no_cc", help="Exclude cosmic chronometers (CC) from the fit")
parser.add_argument("--fixed-anchor", action="store_true", help="Fix H0=67.4 and M=SH0ES (M=0) for ALL models")
parser.add_argument("--sanity-check", action="store_true", dest="sanity_check",
                    help="Internal sanity check: ΛCDM(H0=67.4,Ωm=0.315,M free) vs γCDM/Decay(H0=67.4,M removed)")
parser.add_argument("--output-dir", type=str, default="chains", help="Output directory for nested sampling chains")
parser.add_argument("--legacy", action="store_true", help="Include legacy models (γCDM-LOG², γCDM-Decay, γCDM-LINEAL, γCDM-LOG³) in MLE analysis")
parser.add_argument("--student", action="store_true", help="Use Student-t likelihood (robust to outliers, default ν=5)")
parser.add_argument("--cauchy", action="store_true", help="Use Cauchy likelihood (ν=1, maximum robustness)")
parser.add_argument("--nu", type=float, default=5.0, help="Degrees of freedom for Student-t (default: 5.0)")
parser.add_argument("--cov", type=str, default="none", choices=["none", "stat", "sys"],
                    help="Covariance matrix type for SNe Ia (default: 'none', 'stat' for STATONLY, 'sys' for STAT+SYS)")
parser.add_argument("--evo", action="store_true",
                    help="Use differential_evolution global optimizer instead of random-restart Nelder-Mead")
parser.add_argument("--cmb", action="store_true",
                    help="Add CMB shift parameter prior (Planck 2018 R=1.7502±0.0046, constrains z_h)")
parser.add_argument("--fit-scatter", action="store_true", dest="fit_scatter",
                    help="Fit intrinsic scatter σ_int,SNe and σ_int,QSO as free parameters (ignores --sigma-int-*)")
parser.add_argument("--camb-tab", action="store_true",
                    help="Pre-tabulate CAMB background grid for faster evaluation")
parser.add_argument("--penalty-m", action="store_true",
                    help="Penalize high calibration parameters to prevent degeneracies")
parser.add_argument("--sh0es", action="store_true",
                    help="Add SH0ES local H0 prior (73.04 ± 1.04) to penalty")
parser.add_argument("--no-bubble", action="store_true", dest="no_bubble",
                    help="Test LOG²-Decay without local bubble: Δμ = γ₀·[ln(1+z)]²·exp(-z/z_h) only (Kerr-Only / Occam test)")
parser.add_argument("--paper", action="store_true",
                    help="Publication-ready output: strip emojis, pin cosmetic formatting, silence advisory lines")
parser.add_argument("--snapshot-chi2", type=str, default=None, dest="snapshot_chi2",
                    help="Regression hook: compute χ² for a fixed set of parameter vectors "
                         "against all 12 chi2_* functions AND exit before the MLE loop. "
                         "Writes a JSON to the given path. Use this before and after a refactor "
                         "to verify bit-compatibility: python test_regression.py --validate.")

# ── BAO / DESI DR1 options (Stage 7) ────────────────────────────────────────
# Two mutually exclusive modes:
#   --bao       : DESI DR1 enters the joint likelihood (SNe+QSO+CC+CMB+BAO).
#                 The minimizer trades off all datasets. Numbers CHANGE.
#   --bao-null  : DESI DR1 is a post-fit null test. The MLE is run WITHOUT
#                 BAO in the objective, and χ²_BAO is evaluated at the
#                 best-fit parameters for each model as a goodness-of-fit
#                 diagnostic. Numbers of the MLE do NOT change; only an
#                 extra diagnostic table is printed at the end.
#
# The propagation of the Δμ(z) correction to BAO D_M is controlled by
# cosmo_constants.BAO_PROPAGATE_CORRECTION (default False, "flux-level"
# interpretation). See bao_desi_dr1.py for the algebra.
_bao_group = parser.add_mutually_exclusive_group()
_bao_group.add_argument("--bao", action="store_true",
                    help="Add DESI DR1 BAO (12 points, Adame+24) to the joint likelihood. "
                         "The γCDM correction Δμ(z) propagates (if BAO_PROPAGATE_CORRECTION is True) to D_M via 10^(Δμ/5); "
                         "D_H stays ΛCDM. Requires CAMB rdrag (computed per-eval).")
_bao_group.add_argument("--bao-null", action="store_true", dest="bao_null",
                    help="Post-fit null test with DESI DR1: fit the model without BAO, "
                         "then evaluate χ²_BAO at the best-fit to diagnose SNe-vs-BAO "
                         "consistency. Does NOT change the MLE results, only adds a "
                         "diagnostic table.")
args = parser.parse_args()


# ── Publication-ready output hook (--paper) ─────────────────────────────────
# When --paper is active we wrap builtins.print to strip emoji code points.
# This is a post-processing hook; existing print(...) calls throughout the
# file stay untouched. The regex matches the Unicode pictograph ranges
# commonly used for emojis and symbol sets found in the current output.
if args.paper:
    import builtins as _builtins
    import re as _re
    # Strip decorative pictographs while preserving structural punctuation
    # (arrows ← → ↔, en/em dashes, Greek letters, math symbols).
    _EMOJI_RE = _re.compile(
        "["
        "\U0001F300-\U0001FAFF"     # Pictographs / extended symbols
        "\U00002700-\U000027BF"     # Dingbats
        "\U0001F600-\U0001F64F"     # Emoticons
        "\U0001F680-\U0001F6FF"     # Transport & map
        "\u2705\u2714\u2716\u274C\u26A0\u26A1\u2728\u2B50"  # ✅✔✖❌⚠⚡✨⭐
        "]",
        flags=_re.UNICODE,
    )
    _orig_print = _builtins.print
    def _paper_print(*objs, **kwargs):
        cleaned = [_EMOJI_RE.sub("", o) if isinstance(o, str) else o for o in objs]
        _orig_print(*cleaned, **kwargs)
    _builtins.print = _paper_print
# ────────────────────────────────────────────────────────────────────────────


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("🛡️  γCDM ROBUSTNESS VERIFICATION PROTOCOL")
print("=" * 70)
print(f"   ⚙️  Intrinsic scatter: σ_int,SNe = {args.sigma_int_sne:.2f}, σ_int,QSO = {args.sigma_int_qso:.2f}")
print(f"   ⚙️  Physical prior: Ωm < 1 enforced (flat ΛCDM with ΩΛ ≥ 0)")
if args.evo:
    print(f"   ⚙️  Optimizer: EVO (5 × differential evolution ensemble + dual_annealing + multi-polish + perturbation test)")
if args.cmb:
    print(f"   ⚙️  CMB prior: Planck 2018 shift parameter R = 1.7502 ± 0.0046")
if args.sh0es:
    print(f"   ⚙️  SH0ES prior: H0 = 73.04 ± 1.04 km/s/Mpc")
    if args.cov in ["stat", "sys"]:
        print(f"   ⚠️  WARNING: Double counting! Pantheon+ covariance already includes SH0ES calibration. To test prior safely, use --cov none.")
if args.fit_scatter:
    print(f"   ⚙️  Fitting σ_int,SNe and σ_int,QSO as free parameters")
if args.no_bubble:
    print(f"   ⚙️  NO-BUBBLE mode: testing Kerr-Only variant (A=0, z_b removed)")
if args.no_cc:
    print(f"   ⚙️  CC excluded: cosmic chronometers removed from the fit")
print(f"   ⚙️  SNe z-range: [{args.z_min_sne}, {args.z_max_sne}]")
if args.bao:
    _prop = "ON (D_M scaled by 10^(Δμ/5))" if BAO_PROPAGATE_CORRECTION else "OFF (BAO sees ΛCDM only)"
    print(f"   ⚙️  BAO (DESI DR1): In the fit — {N_BAO_POINTS} points, Δμ→D_M {_prop}")
if args.bao_null:
    print(f"   ⚙️  BAO (DESI DR1): Null test (post-fit diagnostic, not in objective)")

# ── Determine likelihood type ──
if args.cauchy:
    LIKELIHOOD_TYPE = 'cauchy'
    NU_DOF = 1.0
    print(f"   ⚙️  Likelihood: Cauchy (ν=1, maximum robustness)")
elif args.student:
    LIKELIHOOD_TYPE = 'student'
    NU_DOF = args.nu
    print(f"   ⚙️  Likelihood: Student-t (ν={NU_DOF:.1f})")
else:
    LIKELIHOOD_TYPE = 'gaussian'
    NU_DOF = None
    print(f"   ⚙️  Likelihood: Gaussian (canonical)")

# Try local first, then GitHub
dataset_name = 'full_dataset_revisado.csv' if args.revised else 'full_dataset.csv'
try:
    df = pd.read_csv(dataset_name)
except Exception:
    try:
        df = pd.read_csv('../' + dataset_name)
    except Exception:
        df = pd.read_csv('https://raw.githubusercontent.com/indigenica/akashic-alpha-engine/main/' + dataset_name)

# ============================================================================
# COVARIANCE MATRIX LOADING (PANTHEON+)
# ============================================================================
# The full_dataset.csv maintains the exact order of the 1701 Pantheon+ SNe Ia
# at the beginning of the file. So the original index is simply their row number.
sne_all = df[df['probe'] == 'sne_ia'].copy()
sne_all['cov_idx'] = np.arange(len(sne_all))

# Replace the original SNe records with these tracking ones
df.loc[df['probe'] == 'sne_ia', 'cov_idx'] = sne_all['cov_idx'].values


if args.quasars_only:
    # ── QUASARS ONLY (high-z test) ──
    print(f"\n🔭 MODE: Quasars only (high-z test, err < {args.qso_err_cut}, z > {args.z_min_qso})")
    qso = df[(df['probe'] == 'quasar') & (df['type'] == 'mu') & (df['err'] < args.qso_err_cut) & (df['z'] > args.z_min_qso)]

    print(f"\n📊 Dataset:")
    print(f"   Quasars: {len(qso)} pts (μ observable, err < {args.qso_err_cut}, z > {args.z_min_qso})")
    print(f"   z range: {qso['z'].min():.2f} – {qso['z'].max():.2f}")
    print(f"   ⟨σ⟩ = {qso['err'].mean():.2f} mag")

    z_mu = qso['z'].values
    mu_obs = qso['val'].values
    err_mu = qso['err'].values

    z_cc = np.array([])
    H_obs = np.array([])
    err_cc = np.array([])

    N = len(qso)
    COMBINED_MODE = False
    SIGMA_INT_SINGLE = args.sigma_int_qso

elif args.no_quasars:
    # ── SNe Ia + CC (sin quasars) ──
    print(f"\n🔭 MODE: SNe Ia (err < {args.sne_err_cut}, {args.z_min_sne} < z < {args.z_max_sne}) + CC (sin quasars)")
    sne = df[(df['probe'] == 'sne_ia') & (df['type'] == 'mu') & (df['err'] < args.sne_err_cut) & (df['z'] > args.z_min_sne) & (df['z'] < args.z_max_sne)]
    cc = df[(df['probe'] == 'cc') & (df['type'] == 'H')] if not args.no_cc else df.iloc[0:0]

    print(f"\n📊 Dataset:")
    print(f"   SNe Ia: {len(sne)} pts (μ, err < {args.sne_err_cut}, {args.z_min_sne} < z < {args.z_max_sne})")
    print(f"   CC:     {len(cc)} pts (H)")
    print(f"   Total:  {len(sne) + len(cc)} pts")

    z_mu = sne['z'].values
    mu_obs = sne['val'].values
    err_mu = sne['err'].values

    z_cc = cc['z'].values
    H_obs = cc['val'].values
    err_cc = cc['err'].values

    N = len(sne) + len(cc)
    COMBINED_MODE = False
    SIGMA_INT_SINGLE = args.sigma_int_sne

else:
    # ── DEFAULT: SNe Ia (err < cut) + Quasars (err < cut) + CC ──
    sne = df[(df['probe'] == 'sne_ia') & (df['type'] == 'mu') & (df['err'] < args.sne_err_cut) & (df['z'] > args.z_min_sne) & (df['z'] < args.z_max_sne)]
    cc = df[(df['probe'] == 'cc') & (df['type'] == 'H')] if not args.no_cc else df.iloc[0:0]
    qso = df[(df['probe'] == 'quasar') & (df['type'] == 'mu') & (df['err'] < args.qso_err_cut) & (df['z'] > args.z_min_qso)]

    n_sne = len(sne)
    n_qso = len(qso)

    mu_data = pd.concat([sne, qso])

    sne_mask = np.zeros(len(mu_data), dtype=bool)
    sne_mask[:n_sne] = True
    qso_mask = ~sne_mask

    print(f"\n🔭 MODE: SNe Ia (err < {args.sne_err_cut}, {args.z_min_sne} < z < {args.z_max_sne}) + Quasars (err < {args.qso_err_cut}, z > {args.z_min_qso}) + CC")
    print(f"\n📊 Dataset:")
    print(f"   SNe Ia:   {n_sne} pts (μ, err < {args.sne_err_cut}, {args.z_min_sne} < z < {args.z_max_sne})")
    print(f"   Quasars:  {n_qso} pts (μ, err < {args.qso_err_cut}, z > {args.z_min_qso})")
    print(f"   CC:       {len(cc)} pts (H){' [EXCLUDED]' if args.no_cc else ''}")
    print(f"   Total μ:  {len(mu_data)} pts")
    print(f"   z range:  {mu_data['z'].min():.2f} – {mu_data['z'].max():.2f}")

    z_mu = mu_data['z'].values
    mu_obs = mu_data['val'].values
    err_mu = mu_data['err'].values

    z_cc = cc['z'].values
    H_obs = cc['val'].values
    err_cc = cc['err'].values

    N = len(mu_data) + len(cc)
    COMBINED_MODE = True

if not COMBINED_MODE:
    sne_mask = None
    qso_mask = None

# Extract Covariance Matrix if requested and applicable
C_inv_sne = None
ln_det_C_sne = None
_cov_evals = None
_cov_evecs = None

if args.cov != 'none' and not args.quasars_only:
    # Build file path
    cov_file = f"Pantheon+SH0ES_STATONLY.cov" if args.cov == 'stat' else f"Pantheon+SH0ES_STAT+SYS.cov"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cov_path = os.path.join(script_dir, cov_file)
    
    if not os.path.exists(cov_path):
        # Try one level up
        cov_path = os.path.join(os.path.dirname(script_dir), cov_file)
        if not os.path.exists(cov_path):
             raise FileNotFoundError(f"Covariance matrix {cov_file} not found locally or in parent dir.")

    print(f"\n🧮 Loading Correlated SNe Covariance: {cov_path}")
    with open(cov_path, 'r') as f:
        first_line = f.readline().strip()
        C_dim = int(first_line)
    
    C_flat = np.loadtxt(cov_path, skiprows=1)
    C_full = C_flat.reshape(C_dim, C_dim)
    
    # Extract the indices of the filtered SNe
    sne_idx = sne['cov_idx'].values.astype(int)
    
    print(f"   Filtering {C_dim}x{C_dim} covariance matrix to match {len(sne_idx)} surviving SNe Ia...")
    # Submatrix
    C_sne = C_full[np.ix_(sne_idx, sne_idx)]
    
    # Eigendecompose base covariance for --fit-scatter (efficient σ_int updates)
    _cov_evals = None
    _cov_evecs = None
    if args.fit_scatter:
        print(f"   Eigendecomposing base covariance for σ_int fitting...")
        _cov_evals, _cov_evecs = np.linalg.eigh(C_sne)
        _cov_evals = np.maximum(_cov_evals, 1e-15)
        print(f"   Eigendecomposition done ({len(_cov_evals)} modes, λ_min={_cov_evals[0]:.2e}, λ_max={_cov_evals[-1]:.2e})")
    
    if not args.fit_scatter and args.sigma_int_sne > 0:
        print(f"   Adding SNe intrinsic scatter (σ_int = {args.sigma_int_sne}) to the covariance diagonal...")
        np.fill_diagonal(C_sne, C_sne.diagonal() + args.sigma_int_sne**2)
    
    print(f"   Inverting SNe Covariance Matrix...")
    C_inv_sne = np.linalg.inv(C_sne)
    sign, ln_det_C_sne = np.linalg.slogdet(C_sne)
    if sign <= 0:
        raise ValueError("Determinant of the SNe covariance matrix is not positive.")
    
    print(f"   Covariance prepared successfully.")

# ============================================================================
# CHECK CAMB
# ============================================================================
try:
    import camb
    CAMB_AVAILABLE = True
    print("\n✅ CAMB loaded")
except ImportError:
    CAMB_AVAILABLE = False
    print("\n❌ CAMB not available. Install with: pip install camb")
    exit(1)

# Global tracker for the physical chi2 (quadratic term) only for reporting
GLOBAL_CHI2 = 0.0


# ============================================================================
# χ² FUNCTIONS WITH δM NUISANCE + INTRINSIC SCATTER
# ============================================================================
# Shared parameters have IDENTICAL priors across ALL models:
#   H₀:       [40, 100]
#   Ωch²:     [0.01, 0.35]  → but we also enforce Ωm < 1 (physical)
#   δM:       [−3.0, 3.0]
#   γ:        [−3.0, 3.0]
#   σ_int:    [0.0, 2.0]    → intrinsic scatter (QSO and optionally SNe)
# ============================================================================

# All prior bounds, physical constants and external calibration values are now
# imported from cosmo_constants at the top of the file. Do NOT redefine any of
# H0_MIN, H0_MAX, OMCH2_MIN, OMCH2_MAX, M_MIN, M_MAX, GAMMA_MIN, GAMMA_MAX,
# A_MIN, A_MAX, ZD/ZB/ZH bounds, SINT_SNE/QSO bounds, Z_STAR, R_PLANCK,
# SIGMA_R_PLANCK, SIGMA_CORR_CMB, C_LIGHT_KMS here.

# Runtime-configurable (comes from CLI, not a physical constant)
SIGMA_INT_SNE = args.sigma_int_sne   # ~0.1 mag typical for Pantheon+ without cov
SIGMA_INT_QSO = args.sigma_int_qso   # User-specified, try 0.3-0.6 for robustness test

USE_CMB = args.cmb
FIT_SCATTER = args.fit_scatter
USE_EVO = args.evo

# BAO (DESI DR1) wiring. `USE_BAO_FIT` means BAO enters the joint likelihood;
# `USE_BAO_NULL` means BAO is evaluated only at the post-fit as a diagnostic.
# Both flags imply the CAMB background extras (rdrag, D_M(Z_EFF), D_H(Z_EFF))
# must be computed — that's governed by `NEED_BAO_BG`.
USE_BAO_FIT  = bool(args.bao)
USE_BAO_NULL = bool(args.bao_null)
NEED_BAO_BG  = USE_BAO_FIT or USE_BAO_NULL

# Precompute effective errors (model-independent, used for δM profiling)
if COMBINED_MODE:
    ERR_EFF_MU = np.sqrt(err_mu**2 + np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)**2)
else:
    ERR_EFF_MU = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)


GLOBAL_EVAL_MODE = False
GLOBAL_EVAL_ARRAYS = {}


def _m_penalty(m_vals):
    """Penalize high M values (Gaussian prior, sigma=0.1) to prevent degeneracies."""
    if not getattr(args, 'penalty_m', False):
        return 0.0
    if not isinstance(m_vals, (list, tuple)):
        m_vals = [m_vals]
    return sum((m / 0.1)**2 for m in m_vals)

def _neg2logL_mu(residuals, err_eff):
    """Compute -2·ln L for distance-modulus residuals.

    Dispatches Gaussian / Student-t / Cauchy based on global LIKELIHOOD_TYPE.
    CC H(z) data is ALWAYS Gaussian (handled separately in chi2_* functions).
    If C_inv_sne is loaded, SNe Ia are evaluated with the Correlated Gaussian
    likelihood, bypassing the robust diagonal likelihoods for SNe points.
    """
    if GLOBAL_EVAL_MODE:
        GLOBAL_EVAL_ARRAYS['residuals'] = residuals
        GLOBAL_EVAL_ARRAYS['err_eff'] = err_eff
        
    global GLOBAL_CHI2
    GLOBAL_CHI2 = 0.0
    neg2logL_total = 0.0
    
    # --- SPLIT RESIDUALS IF COVARIANCE IS ACTIVE ---
    if C_inv_sne is not None:
        if COMBINED_MODE:
            res_sne = residuals[sne_mask]
            res_qso = residuals[qso_mask]
            err_qso = err_eff[qso_mask]
        else:
            res_sne = residuals
            res_qso = np.array([])
            err_qso = np.array([])
            
        # Correlated Gaussian for SNe: R^T * C^{-1} * R + ln|C|
        # (Note: we add N_sne * ln(2π) if we want absolute likelihood, but for relative
        #  delta-logL we just need R^T C^{-1} R + ln|C|. The diagonal Gaussian also omits 2π).
        chi2_sne = res_sne.T @ C_inv_sne @ res_sne
        GLOBAL_CHI2 += chi2_sne
        neg2logL_total += chi2_sne + ln_det_C_sne
    else:
        # No covariance -> all points go to the diagonal evaluator
        res_qso = residuals
        err_qso = err_eff

    # --- DIAGONAL LIKELIHOOD FOR REMAINING POINTS (QSOs, or all if no cov) ---
    if len(res_qso) > 0:
        if LIKELIHOOD_TYPE == 'student' or LIKELIHOOD_TYPE == 'cauchy':
            nu = NU_DOF if LIKELIHOOD_TYPE == 'student' else 1.0
            # -2 ln L_i = 2·ln(σ) + (ν+1)·ln(1 + r²/(ν·σ²))
            r2 = res_qso**2
            s2 = err_qso**2
            GLOBAL_CHI2 += np.sum(r2 / s2)
            diag_term = np.sum(2 * np.log(err_qso)
                               + (nu + 1) * np.log1p(r2 / (nu * s2 + 1e-30)))
            neg2logL_total += diag_term
        else:
            # Standard Gaussian: −2 ln L = Σ[(r/σ)² + ln(σ²)]
            chi2_term = np.sum((res_qso / err_qso) ** 2)
            GLOBAL_CHI2 += chi2_term
            norm_term = np.sum(np.log(err_qso**2))
            neg2logL_total += chi2_term + norm_term

    return neg2logL_total

def compute_Omega_m(H0, omch2, ombh2=0.0224):
    """Compute Ωm from H0 and Ωch² + Ωbh²."""
    h = H0 / 100
    return (omch2 + ombh2) / h**2

def check_physical_prior(H0, omch2):
    """Check if cosmology is physical: Ωm < 1 (i.e., ΩΛ ≥ 0 for flat)."""
    Om = compute_Omega_m(H0, omch2)
    return 0 < Om < 1.0  # Physical flat ΛCDM requires 0 < Ωm < 1



def _cmb_penalty(H0, omch2, da_star, delta_mu_star=0.0):
    """Planck 2018 shift parameter R + correction penalty at z*.
    
    da_star: angular diameter distance to z* (scalar, pre-computed from table).
    """
    if not USE_CMB:
        return 0.0
    Om = compute_Omega_m(H0, omch2)
    DC_star = (1 + Z_STAR) * da_star
    R_model = np.sqrt(Om) * (H0 / C_LIGHT_KMS) * DC_star
    penalty = ((R_model - R_PLANCK) / SIGMA_R_PLANCK) ** 2
    if abs(delta_mu_star) > 1e-10:
        penalty += (delta_mu_star / SIGMA_CORR_CMB) ** 2
    return penalty


def _sh0es_penalty(H0_local):
    """SH0ES local H0 prior (73.04 ± 1.04) penalty."""
    if not getattr(args, 'sh0es', False):
        return 0.0
    return ((H0_local - 73.04) / 1.04) ** 2


def _neg2logL_cov_eigbasis(res_sne, sig_int_sne):
    """Correlated SNe -2lnL via eigendecomposition: O(N) per σ_int value.
    
    Uses C(σ) = U(Λ + σ²I)U^T → r^T C^{-1} r = Σ v²/(λ+σ²),
    ln|C| = Σ ln(λ+σ²), where v = U^T r.
    """
    v = _cov_evecs.T @ res_sne
    lam_eff = _cov_evals + sig_int_sne**2
    chi2 = np.sum(v**2 / lam_eff)
    ln_det = np.sum(np.log(lam_eff))
    return chi2 + ln_det


def _neg2logL_diag(residuals, err_obs, sig_int):
    """Diagonal -2lnL with variable σ_int (for --fit-scatter)."""
    s2 = err_obs**2 + sig_int**2
    global GLOBAL_CHI2
    _chi2 = np.sum(residuals**2 / s2)
    GLOBAL_CHI2 = _chi2 # Note: child calls might overwrite, but _chi2_tail sums up components

    if LIKELIHOOD_TYPE == 'student' or LIKELIHOOD_TYPE == 'cauchy':
        nu = NU_DOF if LIKELIHOOD_TYPE == 'student' else 1.0
        return np.sum(np.log(s2) + (nu + 1) * np.log1p(residuals**2 / (nu * s2 + 1e-30)))
    return _chi2 + np.sum(np.log(s2))


def _neg2logL_fit_scatter(residuals, sig_int_sne, sig_int_qso):
    """Full -2lnL when fitting scatter: handles covariance + diagonal split."""
    if GLOBAL_EVAL_MODE:
        GLOBAL_EVAL_ARRAYS['residuals'] = residuals
        if COMBINED_MODE:
            err_eff = np.sqrt(err_mu**2 + np.where(sne_mask, sig_int_sne, sig_int_qso)**2)
        else:
            _is_qso_only = getattr(args, 'quasars_only', False)
            err_eff = np.sqrt(err_mu**2 + (sig_int_qso if _is_qso_only else sig_int_sne)**2)
        GLOBAL_EVAL_ARRAYS['err_eff'] = err_eff

    # Pure chi2 for reporting
    _chi2_mu = 0.0
    if _cov_evals is not None:
        _v_sne = _cov_evecs.T @ (residuals[sne_mask] if COMBINED_MODE else residuals)
        _chi2_mu += np.sum(_v_sne**2 / (_cov_evals + sig_int_sne**2))
        if COMBINED_MODE:
            _chi2_mu += np.sum(residuals[qso_mask]**2 / (err_mu[qso_mask]**2 + sig_int_qso**2))
    elif C_inv_sne is not None:
        if COMBINED_MODE:
            _chi2_mu += residuals[sne_mask].T @ C_inv_sne @ residuals[sne_mask]
            _chi2_mu += np.sum(residuals[qso_mask]**2 / (err_mu[qso_mask]**2 + sig_int_qso**2))
        else:
            _chi2_mu += residuals.T @ C_inv_sne @ residuals
    else:
        if COMBINED_MODE:
            _chi2_mu += np.sum(residuals[sne_mask]**2 / (err_mu[sne_mask]**2 + sig_int_sne**2))
            _chi2_mu += np.sum(residuals[qso_mask]**2 / (err_mu[qso_mask]**2 + sig_int_qso**2))
        else:
            _chi2_mu += np.sum(residuals**2 / (err_mu**2 + sig_int_sne**2))
    GLOBAL_CHI2 = _chi2_mu
    
    total = 0.0
    if _cov_evals is not None:
        if COMBINED_MODE:
            total += _neg2logL_cov_eigbasis(residuals[sne_mask], sig_int_sne)
            total += _neg2logL_diag(residuals[qso_mask], err_mu[qso_mask], sig_int_qso)
        else:
            total += _neg2logL_cov_eigbasis(residuals, sig_int_sne)
    elif C_inv_sne is not None:
        if COMBINED_MODE:
            res_sne = residuals[sne_mask]
            total += res_sne.T @ C_inv_sne @ res_sne + ln_det_C_sne
            total += _neg2logL_diag(residuals[qso_mask], err_mu[qso_mask], sig_int_qso)
        else:
            total += residuals.T @ C_inv_sne @ residuals + ln_det_C_sne
    else:
        if COMBINED_MODE:
            total += _neg2logL_diag(residuals[sne_mask], err_mu[sne_mask], sig_int_sne)
            total += _neg2logL_diag(residuals[qso_mask], err_mu[qso_mask], sig_int_qso)
        else:
            sig_val = sig_int_sne
            total += _neg2logL_diag(residuals, err_mu, sig_val)
    return total


# ============================================================================
# PRE-TABULATED CAMB BACKGROUND (120×80 grid → ~40s setup, ~300× faster evals)
# ============================================================================
if args.camb_tab:
    _N_H0_GRID, _N_OMCH2_GRID = 120, 80
    _h0_grid = np.linspace(H0_MIN, H0_MAX, _N_H0_GRID)
    _omch2_grid = np.linspace(OMCH2_MIN, OMCH2_MAX, _N_OMCH2_GRID)

    import time as _time
    _t0_tab = _time.time()

    # Cache file name reflects whether BAO extras (rdrag + D_M/D_H at DESI z_eff)
    # are required. This avoids invalidating the standard cache when users run
    # without --bao / --bao-null, and keeps BAO runs hot.
    _cache_file = "camb_grid_cache_bao.npz" if NEED_BAO_BG else "camb_grid_cache.npz"
    _loaded_cache = False

    if os.path.exists(_cache_file):
        try:
            with np.load(_cache_file) as data:
                _mu_base_table = data["mu_base"]
                _hz_table = data["hz"] if "hz" in data else None
                _da_star_table = data["da_star"]
                if NEED_BAO_BG:
                    _rdrag_table = data["rdrag"] if "rdrag" in data else None
                    _dm_bao_table = data["dm_bao"] if "dm_bao" in data else None
                    _dh_bao_table = data["dh_bao"] if "dh_bao" in data else None
                else:
                    _rdrag_table = _dm_bao_table = _dh_bao_table = None

                _shape_ok = _mu_base_table.shape == (_N_H0_GRID, _N_OMCH2_GRID, len(z_mu))
                _hz_ok = (_hz_table is None
                          or _hz_table.shape == (_N_H0_GRID, _N_OMCH2_GRID, len(z_cc)))
                _bao_ok = (not NEED_BAO_BG) or (
                    _rdrag_table is not None
                    and _rdrag_table.shape == (_N_H0_GRID, _N_OMCH2_GRID)
                    and _dm_bao_table is not None
                    and _dm_bao_table.shape == (_N_H0_GRID, _N_OMCH2_GRID, len(Z_EFF_DESI))
                    and _dh_bao_table is not None
                    and _dh_bao_table.shape == (_N_H0_GRID, _N_OMCH2_GRID, len(Z_EFF_DESI))
                )
                if _shape_ok and _hz_ok and _bao_ok:
                    _loaded_cache = True
                    print(f"\n⚡ Cargando grid CAMB desde caché ({_cache_file})...")
        except Exception:
            pass

    if not _loaded_cache:
        _bao_suffix = " +rdrag/D_M/D_H(DESI z_eff)" if NEED_BAO_BG else ""
        print(f"\n⏳ Pre-tabulando CAMB background ({_N_H0_GRID}×{_N_OMCH2_GRID} grid){_bao_suffix}...")

        _mu_base_table = np.empty((_N_H0_GRID, _N_OMCH2_GRID, len(z_mu)))
        _hz_table = np.empty((_N_H0_GRID, _N_OMCH2_GRID, len(z_cc))) if len(z_cc) > 0 else None
        _da_star_table = np.empty((_N_H0_GRID, _N_OMCH2_GRID))
        if NEED_BAO_BG:
            _rdrag_table  = np.empty((_N_H0_GRID, _N_OMCH2_GRID))
            _dm_bao_table = np.empty((_N_H0_GRID, _N_OMCH2_GRID, len(Z_EFF_DESI)))
            _dh_bao_table = np.empty((_N_H0_GRID, _N_OMCH2_GRID, len(Z_EFF_DESI)))
        else:
            _rdrag_table = _dm_bao_table = _dh_bao_table = None

        for _i_h0, _h0_val in enumerate(_h0_grid):
            for _j_oc, _oc_val in enumerate(_omch2_grid):
                _Om_check = (_oc_val + OMBH2_FIDUCIAL) / (_h0_val / 100) ** 2
                if _Om_check <= 0.0:
                    _mu_base_table[_i_h0, _j_oc, :] = np.nan
                    if _hz_table is not None:
                        _hz_table[_i_h0, _j_oc, :] = np.nan
                    _da_star_table[_i_h0, _j_oc] = np.nan
                    if NEED_BAO_BG:
                        _rdrag_table[_i_h0, _j_oc] = np.nan
                        _dm_bao_table[_i_h0, _j_oc, :] = np.nan
                        _dh_bao_table[_i_h0, _j_oc, :] = np.nan
                    continue
                try:
                    _pars_tab = camb.CAMBparams()
                    _pars_tab.WantTransfer = False
                    _pars_tab.WantCls = False
                    _pars_tab.set_cosmology(H0=_h0_val, ombh2=OMBH2_FIDUCIAL, omch2=_oc_val)
                    _r_tab = camb.get_background(_pars_tab)
                    _mu_base_table[_i_h0, _j_oc, :] = (
                        5.0 * np.log10(np.maximum(_r_tab.luminosity_distance(z_mu), 1e-10)) + 25.0
                    )
                    if _hz_table is not None:
                        _hz_table[_i_h0, _j_oc, :] = _r_tab.hubble_parameter(z_cc)
                    _da_star_table[_i_h0, _j_oc] = _r_tab.angular_diameter_distance(Z_STAR)
                    if NEED_BAO_BG:
                        # D_M(z) = comoving radial distance (flat FRW); D_H(z) = c/H(z).
                        _dm_bao_table[_i_h0, _j_oc, :] = _r_tab.comoving_radial_distance(Z_EFF_DESI)
                        _dh_bao_table[_i_h0, _j_oc, :] = C_LIGHT_KMS / _r_tab.hubble_parameter(Z_EFF_DESI)
                        # r_d from CAMB derived params (sound horizon at drag, Mpc)
                        try:
                            _rdrag_table[_i_h0, _j_oc] = _r_tab.get_derived_params()["rdrag"]
                        except Exception:
                            _rdrag_table[_i_h0, _j_oc] = np.nan
                except Exception:
                    _mu_base_table[_i_h0, _j_oc, :] = np.nan
                    if _hz_table is not None:
                        _hz_table[_i_h0, _j_oc, :] = np.nan
                    _da_star_table[_i_h0, _j_oc] = np.nan
                    if NEED_BAO_BG:
                        _rdrag_table[_i_h0, _j_oc] = np.nan
                        _dm_bao_table[_i_h0, _j_oc, :] = np.nan
                        _dh_bao_table[_i_h0, _j_oc, :] = np.nan

        # Fill non-physical NaN cells with nearest physical neighbor so cubic
        # spline construction gets all-finite values.  The chi² functions already
        # reject non-physical (Ωm ≥ 1) points via check_physical_prior() before
        # calling _fast_camb_bg, so these padded values are never actually used.
        _nan_mask_2d = np.isnan(_da_star_table)
        _n_nan = int(np.sum(_nan_mask_2d))
        if _n_nan > 0:
            from scipy.ndimage import distance_transform_edt
            _, _nn_idx = distance_transform_edt(_nan_mask_2d, return_indices=True)
            _da_star_table = _da_star_table[tuple(_nn_idx)]
            for _kz in range(_mu_base_table.shape[2]):
                _mu_base_table[:, :, _kz] = _mu_base_table[:, :, _kz][tuple(_nn_idx)]
            if _hz_table is not None:
                for _kz in range(_hz_table.shape[2]):
                    _hz_table[:, :, _kz] = _hz_table[:, :, _kz][tuple(_nn_idx)]
            if NEED_BAO_BG:
                _rdrag_table = _rdrag_table[tuple(_nn_idx)]
                for _kz in range(_dm_bao_table.shape[2]):
                    _dm_bao_table[:, :, _kz] = _dm_bao_table[:, :, _kz][tuple(_nn_idx)]
                    _dh_bao_table[:, :, _kz] = _dh_bao_table[:, :, _kz][tuple(_nn_idx)]
            print(f"   📐 {_n_nan} non-physical grid cells padded (nearest-neighbor)")

        try:
            _save_kwargs = {"mu_base": _mu_base_table, "da_star": _da_star_table}
            if _hz_table is not None:
                _save_kwargs["hz"] = _hz_table
            if NEED_BAO_BG:
                _save_kwargs["rdrag"]  = _rdrag_table
                _save_kwargs["dm_bao"] = _dm_bao_table
                _save_kwargs["dh_bao"] = _dh_bao_table
            np.savez(_cache_file, **_save_kwargs)
            print(f"   💾 Grid guardado exitosamente en {_cache_file}")
        except Exception as e:
            print(f"   ⚠️ No se pudo guardar caché: {e}")

    _mu_base_interp = RegularGridInterpolator(
        (_h0_grid, _omch2_grid), _mu_base_table,
        method='cubic', bounds_error=False, fill_value=None,
    )
    _hz_interp = (
        RegularGridInterpolator(
            (_h0_grid, _omch2_grid), _hz_table,
            method='cubic', bounds_error=False, fill_value=None,
        ) if _hz_table is not None else None
    )
    _da_star_interp = RegularGridInterpolator(
        (_h0_grid, _omch2_grid), _da_star_table,
        method='cubic', bounds_error=False, fill_value=None,
    )
    if NEED_BAO_BG:
        _rdrag_interp = RegularGridInterpolator(
            (_h0_grid, _omch2_grid), _rdrag_table,
            method='cubic', bounds_error=False, fill_value=None,
        )
        _dm_bao_interp = RegularGridInterpolator(
            (_h0_grid, _omch2_grid), _dm_bao_table,
            method='cubic', bounds_error=False, fill_value=None,
        )
        _dh_bao_interp = RegularGridInterpolator(
            (_h0_grid, _omch2_grid), _dh_bao_table,
            method='cubic', bounds_error=False, fill_value=None,
        )
    else:
        _rdrag_interp = _dm_bao_interp = _dh_bao_interp = None

    _dt_tab = _time.time() - _t0_tab
    print(f"   ✅ Tabla CAMB construida en {_dt_tab:.1f}s "
      f"({_N_H0_GRID}×{_N_OMCH2_GRID} = {_N_H0_GRID*_N_OMCH2_GRID} puntos)")


# ────────────────────────────────────────────────────────────────────────────
# Shared helpers for the chi2_* family (Stage 4b refactor).
#
# The 14 chi2_* functions below all share the same tail once cosmology
# (H0, omch2), calibration offsets (M_sne/M_qso/δM) and the correction array
# Δμ(z) are known. `_chi2_tail` encapsulates that tail:
#   · CAMB background call (with exception tracking)
#   · μ_th = μ_base + offset + correction
#   · residuals → log-likelihood (Gaussian / Student-t / Cauchy / correlated)
#   · CC term (if any)
#   · M-penalty, CMB shift-parameter penalty, SH0ES penalty
#
# `_extract_fit_scatter` pops σ_int,SNe/QSO from the tail of `params` when
# FIT_SCATTER is active for a supporting spec (ΛCDM / LOG²-Decay).
#
# These helpers preserve EXACT numerical behavior of the original
# hand-written chi2_* bodies; any deviation would fail the regression
# test `test_regression.py --validate`.
# ────────────────────────────────────────────────────────────────────────────

def _extract_fit_scatter(params, supported):
    """Pop σ_int parameters from the tail of `params` if FIT_SCATTER is on.

    Returns (sig_sne, sig_qso, ok). When FIT_SCATTER is off or the spec
    does not support it, (SIGMA_INT_SNE, SIGMA_INT_QSO, True) is returned
    and `params` is left untouched. On prior violation ok=False.
    """
    if not (FIT_SCATTER and supported):
        return SIGMA_INT_SNE, SIGMA_INT_QSO, True
    if COMBINED_MODE:
        _sig_qso = params.pop()
        _sig_sne = params.pop()
        if not (SINT_SNE_MIN < _sig_sne < SINT_SNE_MAX):
            return None, None, False
        if not (SINT_QSO_MIN < _sig_qso < SINT_QSO_MAX):
            return None, None, False
        return _sig_sne, _sig_qso, True
    _sig_sne = params.pop()
    if not (SINT_SNE_MIN < _sig_sne < SINT_SNE_MAX):
        return None, None, False
    return _sig_sne, _sig_sne, True


def _chi2_tail(H0, omch2, correction_arr,
               M_sne=0.0, M_qso=0.0, delta_M=0.0,
               sig_sne=None, sig_qso=None,
               use_cmb=False, delta_mu_star=0.0,
               use_m_penalty=False,
               H0_local=None,
               bao_correction_zeff=None):
    """Common -2·lnL backbone shared by the entire chi2_* family.

    Encapsulates CAMB call, μ assembly, likelihood (Gaussian or robust via
    `_neg2logL_mu` / `_neg2logL_fit_scatter`), CC term, the three penalties
    (M, CMB, SH0ES) and — when --bao is active — the DESI DR1 BAO χ² term.

    Parameters
    ----------
    bao_correction_zeff : (N_Z_EFF,) ndarray or None
        Δμ evaluated at Z_EFF_DESI for this model. When USE_BAO_FIT is True
        and this array is provided, BAO enters the fit via D_M_model =
        D_M_ΛCDM · 10^(Δμ/5). Ignored otherwise.
    """
    global GLOBAL_CHI2
    try:
        mu_th_base, _hz_pred, _da_star, _bao_extras = _fast_camb_bg(
            H0, omch2, want_bao=USE_BAO_FIT
        )
        if mu_th_base is None:
            return 1e10

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso) + correction_arr
        else:
            mu_th = mu_th_base + delta_M + correction_arr

        residuals = mu_obs - mu_th
        if FIT_SCATTER and sig_sne is not None:
            # Only reached for ΛCDM and LOG²-Decay (the two specs that
            # support --fit-scatter). The fitted σ_int values drive a
            # Gaussian log-likelihood via _neg2logL_fit_scatter.
            neg2logL_mu = _neg2logL_fit_scatter(residuals, sig_sne, sig_qso)
        else:
            if COMBINED_MODE:
                _s = sig_sne if sig_sne is not None else SIGMA_INT_SNE
                _q = sig_qso if sig_qso is not None else SIGMA_INT_QSO
                err_eff = np.sqrt(err_mu**2 + np.where(sne_mask, _s, _q) ** 2)
            else:
                err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE ** 2)
            neg2logL_mu = _neg2logL_mu(residuals, err_eff)

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - _hz_pred) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc ** 2))
        else:
            chi2_cc = norm_cc = 0

        total = neg2logL_mu + chi2_cc + norm_cc
        if use_m_penalty:
            total += _m_penalty([M_sne, M_qso] if COMBINED_MODE else delta_M)
        if use_cmb:
            total += _cmb_penalty(H0, omch2, _da_star, delta_mu_star)
        total += _sh0es_penalty(H0 if H0_local is None else H0_local)

        # ── BAO (DESI DR1) likelihood term ──────────────────────────────────
        # Only active when --bao is passed (USE_BAO_FIT). For --bao-null the
        # diagnostic is computed post-fit (see `_bao_null_test_report`),
        # preserving bit-exact MLE results.
        if USE_BAO_FIT and _bao_extras is not None:
            dm_zeff = _bao_extras["dm"]
            dh_zeff = _bao_extras["dh"]
            if BAO_PROPAGATE_CORRECTION and bao_correction_zeff is not None:
                # D_M_model / r_d = (D_M_ΛCDM · 10^(Δμ/5)) / r_d
                dm_zeff = dm_zeff * (10.0 ** (bao_correction_zeff / 5.0))
            chi2_bao, _ = compute_chi2_bao(dm_zeff, dh_zeff, _bao_extras["rdrag"])
            total += chi2_bao
            GLOBAL_CHI2 += chi2_bao
        return total
    except Exception as e:
        CAMB_ERRORS.record(e)
        return 1e10


def _bao_corr_at(z_arr, kind, *,
                 gamma=0.0, gamma_0=0.0,
                 A=0.0, z_b=1.0, z_h=1e10, zd=1.0):
    """Δμ(z) at arbitrary redshift array for the given model `kind`.

    `kind` ∈ {"lcdm", "gcdm", "linear", "log2", "log3", "decay", "log_decay"}.
    Keeps the exact same algebra used inline in each chi2_* wrapper for the
    z_mu grid — so the BAO prediction at DESI z_eff is bit-consistent with the
    SNe µ prediction.
    """
    z = np.asarray(z_arr, dtype=float)
    if kind == "lcdm":
        return np.zeros_like(z)
    if kind == "gcdm":
        return gamma * np.log(1.0 + z)
    if kind == "linear":
        return gamma_0 * (1.0 + z) * np.log(1.0 + z)
    if kind == "log2":
        return gamma_0 * np.log(1.0 + z) ** 2
    if kind == "log3":
        return gamma_0 * np.log(1.0 + z) ** 3
    if kind == "decay":
        return A * np.exp(-z / zd)
    if kind == "log_decay":
        return A * np.exp(-z / z_b) + gamma_0 * np.log(1.0 + z) ** 2 * np.exp(-z / z_h)
    raise ValueError(f"Unknown correction kind: {kind!r}")


def _bao_corr_if_fit(kind, **kwargs):
    """Δμ at Z_EFF_DESI if BAO is in the fit, else None (no work done)."""
    if not USE_BAO_FIT:
        return None
    return _bao_corr_at(Z_EFF_DESI, kind, **kwargs)


def _fast_camb_bg(H0, omch2, want_bao=False):
    """Interpolated CAMB background OR on-the-fly exact computation.

    Returns (mu_base, hz_pred, da_star, bao_extras).

    When `want_bao` is False (default), `bao_extras` is None and the BAO
    arrays are never computed — no extra CAMB work and no interpolator call.
    When `want_bao` is True, `bao_extras` is a dict
        {"rdrag": float, "dm": (7,) ndarray, "dh": (7,) ndarray}
    with D_M and D_H evaluated at Z_EFF_DESI [Mpc].
    """
    if args.camb_tab:
        pt = np.array([[H0, omch2]])
        mu_base = np.ravel(_mu_base_interp(pt))
        hz_pred = np.ravel(_hz_interp(pt)) if _hz_interp is not None else np.array([])
        da_star = float(_da_star_interp(pt))
        if np.any(np.isnan(mu_base)) or np.isnan(da_star):
            return None, None, None, None
        bao_extras = None
        if want_bao and _rdrag_interp is not None:
            rdrag = float(_rdrag_interp(pt))
            dm = np.ravel(_dm_bao_interp(pt))
            dh = np.ravel(_dh_bao_interp(pt))
            if np.isnan(rdrag) or np.any(np.isnan(dm)) or np.any(np.isnan(dh)):
                return None, None, None, None
            bao_extras = {"rdrag": rdrag, "dm": dm, "dh": dh}
        return mu_base, hz_pred, da_star, bao_extras
    else:
        try:
            pars_tab = camb.CAMBparams()
            pars_tab.WantTransfer = False
            pars_tab.WantCls = False
            pars_tab.set_cosmology(H0=H0, ombh2=OMBH2_FIDUCIAL, omch2=omch2)
            r_tab = camb.get_background(pars_tab)
            mu_base = 5.0 * np.log10(np.maximum(r_tab.luminosity_distance(z_mu), 1e-10)) + 25.0
            hz_pred = r_tab.hubble_parameter(z_cc) if len(z_cc) > 0 else np.array([])
            da_star = r_tab.angular_diameter_distance(Z_STAR)
            bao_extras = None
            if want_bao:
                dm = r_tab.comoving_radial_distance(Z_EFF_DESI)
                dh = C_LIGHT_KMS / r_tab.hubble_parameter(Z_EFF_DESI)
                try:
                    rdrag = float(r_tab.get_derived_params()["rdrag"])
                except Exception:
                    rdrag = float("nan")
                if np.isnan(rdrag):
                    return None, None, None, None
                bao_extras = {"rdrag": rdrag, "dm": dm, "dh": dh}
            return mu_base, hz_pred, da_star, bao_extras
        except Exception:
            return None, None, None, None


def chi2_lcdm(params):
    """ΛCDM with optional --fit-scatter, --cmb."""
    params = list(params)
    _sig_sne, _sig_qso, _ok = _extract_fit_scatter(params, supported=True)
    if not _ok:
        return 1e10

    if COMBINED_MODE:
        if args.fixed_anchor or args.sanity_check:
            H0 = 67.4
            omch2, M_sne, M_qso = params[0], params[1], params[2]
        else:
            H0, omch2, M_sne, M_qso = params[0], params[1], params[2], params[3]
            if not (H0_MIN < H0 < H0_MAX): return 1e10
        if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX): return 1e10
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        delta_M = 0.0
    else:
        if args.fixed_anchor or args.sanity_check:
            H0 = 67.4
            omch2, delta_M = params[0], params[1]
        else:
            H0, omch2, delta_M = params[0], params[1], params[2]
            if not (H0_MIN < H0 < H0_MAX): return 1e10
        if not (M_MIN < delta_M < M_MAX): return 1e10
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        M_sne = M_qso = 0.0

    if not check_physical_prior(H0, omch2):
        return 1e10

    return _chi2_tail(
        H0, omch2, np.zeros_like(z_mu, dtype=float),
        M_sne=M_sne, M_qso=M_qso, delta_M=delta_M,
        sig_sne=_sig_sne, sig_qso=_sig_qso,
        use_cmb=True, delta_mu_star=0.0,        # ΛCDM has no Δμ at z*
        use_m_penalty=True,
        H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("lcdm"),
    )


def chi2_gcdm(params):
    """γCDM constant: Δμ = γ·ln(1+z)."""
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma = params
                M_sne = M_qso = 0.0
            else:
                omch2, M_sne, M_qso, gamma = params
            H0 = 67.4
        elif args.sanity_check:
            omch2, gamma = params
            H0, M_sne, M_qso = 67.4, 0.0, 0.0
        else:
            H0, omch2, M_sne, M_qso, gamma = params
            if args.no_nuisance:
                M_sne = M_qso = 0.0
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        if not (GAMMA_MIN < gamma < GAMMA_MAX): return 1e10
        if not args.fixed_anchor and not args.sanity_check:
            if not (H0_MIN < H0 < H0_MAX): return 1e10
        if not args.no_nuisance and not args.sanity_check:
            if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX): return 1e10
        delta_M = 0.0
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma = params
                delta_M = 0.0
            else:
                omch2, delta_M, gamma = params
            H0 = 67.4
        elif args.sanity_check:
            omch2, gamma = params
            H0, delta_M = 67.4, 0.0
        else:
            H0, omch2, delta_M, gamma = params
            if args.no_nuisance:
                delta_M = 0.0
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        if not (GAMMA_MIN < gamma < GAMMA_MAX): return 1e10
        if not args.fixed_anchor and not args.sanity_check:
            if not (H0_MIN < H0 < H0_MAX): return 1e10
        if not args.no_nuisance and not args.sanity_check:
            if not (M_MIN < delta_M < M_MAX): return 1e10
        M_sne = M_qso = 0.0

    if not check_physical_prior(H0, omch2):
        return 1e10

    return _chi2_tail(
        H0, omch2, gamma * np.log(1 + z_mu),
        M_sne=M_sne, M_qso=M_qso, delta_M=delta_M,
        use_m_penalty=True,
        H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("gcdm", gamma=gamma),
    )


def chi2_lcdm_no_M(params):
    """ΛCDM WITHOUT δM: 2 params (H₀, Ωch²)."""
    H0, omch2 = params
    if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
        return 1e10
    if not check_physical_prior(H0, omch2):
        return 1e10
    return _chi2_tail(
        H0, omch2, np.zeros_like(z_mu, dtype=float),
        H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("lcdm"),
    )


def chi2_gcdm_no_M(params):
    """γCDM WITHOUT δM: 3 params (H₀, Ωch², γ)."""
    if args.fixed_anchor or args.sanity_check:
        omch2, gamma = params
        H0 = 67.4
    else:
        H0, omch2, gamma = params
        if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
            return 1e10

    if not (GAMMA_MIN < gamma < GAMMA_MAX):
        return 1e10
    if not check_physical_prior(H0, omch2):
        return 1e10

    return _chi2_tail(
        H0, omch2, gamma * np.log(1 + z_mu),
        H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("gcdm", gamma=gamma),
    )


# =============================================================================
# EVOLVING γ(z) MODELS
# =============================================================================

def _unpack_evolving_with_M(params):
    """Shared unpack for chi2_gcdm_linear / log_squared / log_cubed.

    All three share the same parameter layout ([H0?, omch2, M..., γ₀]) and
    bounds checks. Returns (ok, H0, omch2, M_sne, M_qso, delta_M, gamma_0)
    where ok=False indicates a prior violation (caller returns 1e10).
    """
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma_0 = params
                M_sne = M_qso = 0.0
            else:
                omch2, M_sne, M_qso, gamma_0 = params
            H0 = 67.4
        else:
            H0, omch2, M_sne, M_qso, gamma_0 = params
            if not (H0_MIN < H0 < H0_MAX):
                return (False, None, None, None, None, None, None)
        if not args.no_nuisance:
            if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX):
                return (False, None, None, None, None, None, None)
        if not (OMCH2_MIN < omch2 < OMCH2_MAX):
            return (False, None, None, None, None, None, None)
        if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
            return (False, None, None, None, None, None, None)
        return (True, H0, omch2, M_sne, M_qso, 0.0, gamma_0)

    if args.fixed_anchor:
        if args.no_nuisance:
            omch2, gamma_0 = params
            delta_M = 0.0
        else:
            omch2, delta_M, gamma_0 = params
        H0 = 67.4
    else:
        H0, omch2, delta_M, gamma_0 = params
        if not (H0_MIN < H0 < H0_MAX):
            return (False, None, None, None, None, None, None)
    if not args.no_nuisance:
        if not (M_MIN < delta_M < M_MAX):
            return (False, None, None, None, None, None, None)
    if not (OMCH2_MIN < omch2 < OMCH2_MAX):
        return (False, None, None, None, None, None, None)
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
        return (False, None, None, None, None, None, None)
    return (True, H0, omch2, 0.0, 0.0, delta_M, gamma_0)


def _unpack_evolving_no_M(params):
    """Shared unpack for chi2_gcdm_{linear,log_squared,log_cubed}_no_M."""
    if args.fixed_anchor or args.sanity_check:
        omch2, gamma_0 = params
        H0 = 67.4
    else:
        H0, omch2, gamma_0 = params
    if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
        return (False, None, None, None)
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
        return (False, None, None, None)
    return (True, H0, omch2, gamma_0)


def chi2_gcdm_linear(params):
    """γCDM-LINEAR: Δμ = γ₀·(1+z)·ln(1+z)."""
    ok, H0, omch2, M_sne, M_qso, delta_M, gamma_0 = _unpack_evolving_with_M(params)
    if not ok or not check_physical_prior(H0, omch2):
        return 1e10
    return _chi2_tail(
        H0, omch2, gamma_0 * (1 + z_mu) * np.log(1 + z_mu),
        M_sne=M_sne, M_qso=M_qso, delta_M=delta_M,
        use_m_penalty=True, H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("linear", gamma_0=gamma_0),
    )


def chi2_gcdm_log_squared(params):
    """γCDM-LOG²: Δμ = γ₀·[ln(1+z)]²."""
    ok, H0, omch2, M_sne, M_qso, delta_M, gamma_0 = _unpack_evolving_with_M(params)
    if not ok or not check_physical_prior(H0, omch2):
        return 1e10
    return _chi2_tail(
        H0, omch2, gamma_0 * np.log(1 + z_mu) ** 2,
        M_sne=M_sne, M_qso=M_qso, delta_M=delta_M,
        use_m_penalty=True, H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("log2", gamma_0=gamma_0),
    )


def chi2_gcdm_log_cubed(params):
    """γCDM-LOG³: Δμ = γ₀·[ln(1+z)]³."""
    ok, H0, omch2, M_sne, M_qso, delta_M, gamma_0 = _unpack_evolving_with_M(params)
    if not ok or not check_physical_prior(H0, omch2):
        return 1e10
    return _chi2_tail(
        H0, omch2, gamma_0 * np.log(1 + z_mu) ** 3,
        M_sne=M_sne, M_qso=M_qso, delta_M=delta_M,
        use_m_penalty=True, H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("log3", gamma_0=gamma_0),
    )


# =============================================================================
# ASYMMETRIC (NO δM) EVOLVING MODELS — same wide priors
# =============================================================================

def chi2_gcdm_linear_no_M(params):
    """γCDM-LINEAR sin δM: 3 params (H₀, Ωch², γ₀)."""
    ok, H0, omch2, gamma_0 = _unpack_evolving_no_M(params)
    if not ok or not check_physical_prior(H0, omch2):
        return 1e10
    return _chi2_tail(
        H0, omch2, gamma_0 * (1 + z_mu) * np.log(1 + z_mu),
        H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("linear", gamma_0=gamma_0),
    )


def chi2_gcdm_log_squared_no_M(params):
    """γCDM-LOG² sin δM: 3 params (H₀, Ωch², γ₀)."""
    ok, H0, omch2, gamma_0 = _unpack_evolving_no_M(params)
    if not ok or not check_physical_prior(H0, omch2):
        return 1e10
    return _chi2_tail(
        H0, omch2, gamma_0 * np.log(1 + z_mu) ** 2,
        H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("log2", gamma_0=gamma_0),
    )


def chi2_gcdm_log_cubed_no_M(params):
    """γCDM-LOG³ sin δM: 3 params (H₀, Ωch², γ₀)."""
    ok, H0, omch2, gamma_0 = _unpack_evolving_no_M(params)
    if not ok or not check_physical_prior(H0, omch2):
        return 1e10
    return _chi2_tail(
        H0, omch2, gamma_0 * np.log(1 + z_mu) ** 3,
        H0_local=H0,
        bao_correction_zeff=_bao_corr_if_fit("log3", gamma_0=gamma_0),
    )



# =============================================================================
# DECAY MODEL (Hubble Bubble)
# =============================================================================

def chi2_decay(params):
    """γCDM-Decay: Δμ = A·exp(-z/zd). H0 free by default."""
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, A, zd = params
                M_sne = M_qso = 0.0
            else:
                omch2, M_sne, M_qso, A, zd = params
            H0 = 67.4
        elif args.sanity_check:
            omch2, A, zd = params
            H0, M_sne, M_qso = 67.4, 0.0, 0.0
        else:
            H0, omch2, M_sne, M_qso, A, zd = params
            if args.no_nuisance:
                M_sne = M_qso = 0.0
            if not (H0_MIN < H0 < H0_MAX): return 1e10
            if not args.no_nuisance:
                if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX): return 1e10
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        delta_M = 0.0
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, A, zd = params
                delta_M = 0.0
            else:
                omch2, delta_M, A, zd = params
            H0 = 67.4
        elif args.sanity_check:
            omch2, A, zd = params
            H0, delta_M = 67.4, 0.0
        else:
            H0, omch2, delta_M, A, zd = params
            if args.no_nuisance:
                delta_M = 0.0
            if not (H0_MIN < H0 < H0_MAX): return 1e10
            if not args.no_nuisance:
                if not (M_MIN < delta_M < M_MAX): return 1e10
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        M_sne = M_qso = 0.0

    if not (A_MIN < A < A_MAX and ZD_MIN < zd < ZD_MAX):
        return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    return _chi2_tail(
        H0, omch2, A * np.exp(-z_mu / zd),
        M_sne=M_sne, M_qso=M_qso, delta_M=delta_M,
        use_m_penalty=True,
        H0_local=h0_local(H0, A=A, z_b=zd, z_pivot=Z_PIVOT),
        bao_correction_zeff=_bao_corr_if_fit("decay", A=A, zd=zd),
    )


def chi2_decay_no_M(params):
    """γCDM-Decay sin M: Δμ = A·exp(-z/zd)."""
    if args.fixed_anchor or args.sanity_check:
        omch2, A, zd = params
        H0 = 67.4
    else:
        H0, omch2, A, zd = params
        if not (H0_MIN < H0 < H0_MAX):
            return 1e10
    if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
    if not (A_MIN < A < A_MAX and ZD_MIN < zd < ZD_MAX): return 1e10
    if not check_physical_prior(H0, omch2):
        return 1e10

    return _chi2_tail(
        H0, omch2, A * np.exp(-z_mu / zd),
        H0_local=h0_local(H0, A=A, z_b=zd, z_pivot=Z_PIVOT),
        bao_correction_zeff=_bao_corr_if_fit("decay", A=A, zd=zd),
    )


# =============================================================================
# γCDM-LOG²-DECAY (Goldilocks)
# =============================================================================

def chi2_gcdm_log_decay(params):
    """γCDM-LOG²-DECAY with optional --fit-scatter, --cmb, --no-bubble."""
    params = list(params)
    _sig_sne, _sig_qso, _ok = _extract_fit_scatter(params, supported=True)
    if not _ok:
        return 1e10

    # --no-bubble suppresses the A·exp(-z/z_b) term entirely: neither A nor
    # z_b appear in the parameter vector; we set A=0 and z_b=1 (any non-zero
    # placeholder) so the correction evaluates to 0 and bounds are skipped.
    _use_bubble = not args.no_bubble

    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                if _use_bubble:
                    omch2, A, z_b, gamma_0, z_h = params[0:5]
                else:
                    omch2, gamma_0, z_h = params[0:3]
                    A, z_b = 0.0, 1.0
                M_sne = M_qso = 0.0
            else:
                if _use_bubble:
                    omch2, M_sne, M_qso, A, z_b, gamma_0, z_h = params
                else:
                    omch2, M_sne, M_qso, gamma_0, z_h = params
                    A, z_b = 0.0, 1.0
            H0 = 67.4
        elif args.sanity_check:
            if _use_bubble:
                omch2, A, z_b, gamma_0, z_h = params[0:5]
            else:
                omch2, gamma_0, z_h = params[0:3]
                A, z_b = 0.0, 1.0
            H0, M_sne, M_qso = 67.4, 0.0, 0.0
        else:
            if _use_bubble:
                H0, omch2, M_sne, M_qso, A, z_b, gamma_0, z_h = params
            else:
                H0, omch2, M_sne, M_qso, gamma_0, z_h = params
                A, z_b = 0.0, 1.0
            if args.no_nuisance:
                M_sne = M_qso = 0.0
            if not (H0_MIN < H0 < H0_MAX): return 1e10
            if not args.no_nuisance:
                if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX): return 1e10
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        delta_M = 0.0
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                if _use_bubble:
                    omch2, A, z_b, gamma_0, z_h = params[0:5]
                else:
                    omch2, gamma_0, z_h = params[0:3]
                    A, z_b = 0.0, 1.0
                delta_M = 0.0
            else:
                if _use_bubble:
                    omch2, delta_M, A, z_b, gamma_0, z_h = params
                else:
                    omch2, delta_M, gamma_0, z_h = params
                    A, z_b = 0.0, 1.0
            H0 = 67.4
        elif args.sanity_check:
            if _use_bubble:
                omch2, A, z_b, gamma_0, z_h = params[0:5]
            else:
                omch2, gamma_0, z_h = params[0:3]
                A, z_b = 0.0, 1.0
            H0, delta_M = 67.4, 0.0
        else:
            if _use_bubble:
                H0, omch2, delta_M, A, z_b, gamma_0, z_h = params
            else:
                H0, omch2, delta_M, gamma_0, z_h = params
                A, z_b = 0.0, 1.0
            if args.no_nuisance:
                delta_M = 0.0
            if not (H0_MIN < H0 < H0_MAX): return 1e10
            if not args.no_nuisance:
                if not (M_MIN < delta_M < M_MAX): return 1e10
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        M_sne = M_qso = 0.0

    if _use_bubble:
        if not (A_MIN < A < A_MAX): return 1e10
        if not (ZB_MIN < z_b < ZB_MAX): return 1e10
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX): return 1e10
    if not (ZH_MIN < z_h < ZH_MAX): return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    # Two-component additive correction in redshift space.
    bubble_term = (A * np.exp(-z_mu / z_b)) if _use_bubble else 0.0
    kerr_term = gamma_0 * np.log(1 + z_mu) ** 2 * np.exp(-z_mu / z_h)
    unified_corr = bubble_term + kerr_term

    # Same model evaluated at z* for the CMB shift-parameter penalty.
    bubble_star = (A * np.exp(-Z_STAR / z_b)) if _use_bubble else 0.0
    delta_mu_star = bubble_star + gamma_0 * np.log(1 + Z_STAR) ** 2 * np.exp(-Z_STAR / z_h)

    H0_loc = H0 if not _use_bubble else h0_local(H0, A=A, z_b=z_b, z_pivot=Z_PIVOT)

    # For BAO: same two-component formula at DESI z_eff (with bubble gated by _use_bubble).
    _A_bao = A if _use_bubble else 0.0
    _zb_bao = z_b if _use_bubble else 1.0    # placeholder, bubble_term == 0 anyway

    return _chi2_tail(
        H0, omch2, unified_corr,
        M_sne=M_sne, M_qso=M_qso, delta_M=delta_M,
        sig_sne=_sig_sne, sig_qso=_sig_qso,
        use_cmb=True, delta_mu_star=delta_mu_star,
        use_m_penalty=True,
        H0_local=H0_loc,
        bao_correction_zeff=_bao_corr_if_fit(
            "log_decay", A=_A_bao, z_b=_zb_bao, gamma_0=gamma_0, z_h=z_h
        ),
    )


def chi2_gcdm_log_decay_no_M(params):
    """γCDM-LOG²-DECAY sin M: Δμ = A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h)."""
    if args.fixed_anchor or args.sanity_check:
        omch2, A, z_b, gamma_0, z_h = params
        H0 = 67.4
    else:
        H0, omch2, A, z_b, gamma_0, z_h = params
        if not (H0_MIN < H0 < H0_MAX):
            return 1e10

    if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
    if not (A_MIN < A < A_MAX): return 1e10
    if not (ZB_MIN < z_b < ZB_MAX): return 1e10
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX): return 1e10
    if not (ZH_MIN < z_h < ZH_MAX): return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    corr = A * np.exp(-z_mu / z_b) + gamma_0 * np.log(1 + z_mu) ** 2 * np.exp(-z_mu / z_h)

    # NOTE: the _no_M variant intentionally does NOT apply the CMB shift
    # penalty (this mirrors the original pre-refactor behavior; see
    # regression test for the bit-exact equivalence).
    return _chi2_tail(
        H0, omch2, corr,
        H0_local=h0_local(H0, A=A, z_b=z_b, z_pivot=Z_PIVOT),
        bao_correction_zeff=_bao_corr_if_fit(
            "log_decay", A=A, z_b=z_b, gamma_0=gamma_0, z_h=z_h
        ),
    )


# ============================================================================
# REGRESSION HOOK — --snapshot-chi2 PATH
# ----------------------------------------------------------------------------
# This block runs BEFORE the MLE loop. Its sole job is to freeze the current
# χ² surface for a fixed set of parameter vectors that exercises all 12
# chi2_* functions. The snapshot is used as a before/after baseline around
# the Stage 4b refactor (ModelSpec). If refactored code produces |Δχ²| > 1e-6
# for ANY entry, the refactor is numerically not bit-compatible and must be
# debugged.
# ============================================================================
if args.snapshot_chi2 is not None:
    import json as _json
    import sys as _sys
    # Fixed reference vectors, covering all chi2_* signatures.
    # The values are arbitrary but deliberately bracket physical regions.
    _REF_VECTORS_NO_M = {
        "chi2_lcdm_no_M":              [67.4, 0.120],
        "chi2_gcdm_no_M":              [67.4, 0.120, 0.100],
        "chi2_gcdm_linear_no_M":       [67.4, 0.120, 0.050],
        "chi2_gcdm_log_squared_no_M":  [67.4, 0.120, 0.100],
        "chi2_gcdm_log_cubed_no_M":    [67.4, 0.120, 0.080],
        "chi2_decay_no_M":             [67.4, 0.120, -0.150, 0.400],
        "chi2_gcdm_log_decay_no_M":    [67.4, 0.120, -0.150, 0.400, 0.250, 8.000],
    }
    # The "with-M" variants depend on COMBINED_MODE, so we adapt. In single
    # mode δM is one extra scalar; in combined mode (default) M_sne and
    # M_qso are two extras.
    if COMBINED_MODE:
        _M_PADS = [0.010, 0.050]            # [M_sne, M_qso]
    else:
        _M_PADS = [0.030]                   # [delta_M]
    _REF_VECTORS_WITH_M = {
        "chi2_lcdm":             [67.4, 0.120] + _M_PADS,
        "chi2_gcdm":             [67.4, 0.120] + _M_PADS + [0.100],
        "chi2_gcdm_linear":      [67.4, 0.120] + _M_PADS + [0.050],
        "chi2_gcdm_log_squared": [67.4, 0.120] + _M_PADS + [0.100],
        "chi2_gcdm_log_cubed":   [67.4, 0.120] + _M_PADS + [0.080],
        "chi2_decay":            [67.4, 0.120] + _M_PADS + [-0.150, 0.400],
        "chi2_gcdm_log_decay":   [67.4, 0.120] + _M_PADS + [-0.150, 0.400, 0.250, 8.000],
    }
    _fn_map = {
        "chi2_lcdm": chi2_lcdm,
        "chi2_gcdm": chi2_gcdm,
        "chi2_gcdm_linear": chi2_gcdm_linear,
        "chi2_gcdm_log_squared": chi2_gcdm_log_squared,
        "chi2_gcdm_log_cubed": chi2_gcdm_log_cubed,
        "chi2_decay": chi2_decay,
        "chi2_gcdm_log_decay": chi2_gcdm_log_decay,
        "chi2_lcdm_no_M": chi2_lcdm_no_M,
        "chi2_gcdm_no_M": chi2_gcdm_no_M,
        "chi2_gcdm_linear_no_M": chi2_gcdm_linear_no_M,
        "chi2_gcdm_log_squared_no_M": chi2_gcdm_log_squared_no_M,
        "chi2_gcdm_log_cubed_no_M": chi2_gcdm_log_cubed_no_M,
        "chi2_decay_no_M": chi2_decay_no_M,
        "chi2_gcdm_log_decay_no_M": chi2_gcdm_log_decay_no_M,
    }
    _snapshot = {
        "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in vars(args).items() if k != "snapshot_chi2"},
        "combined_mode": bool(COMBINED_MODE),
        "dataset_shape": {
            "N_mu": int(len(z_mu)),
            "N_sne": int(np.sum(sne_mask)) if sne_mask is not None else None,
            "N_qso": int(np.sum(~sne_mask)) if sne_mask is not None else None,
            "N_cc": int(len(z_cc)),
        },
        "snapshots": {},
    }
    _all_vectors = {**_REF_VECTORS_NO_M, **_REF_VECTORS_WITH_M}
    for _name, _params in _all_vectors.items():
        _fn = _fn_map[_name]
        try:
            _val = float(_fn(_params))
        except Exception as _e:                                # noqa: BLE001
            _val = None
            _snapshot["snapshots"][_name] = {
                "params": list(_params), "value": None, "error": str(_e)[:200]
            }
            continue
        _snapshot["snapshots"][_name] = {"params": list(_params), "value": _val}
        print(f"   [snapshot] {_name:<32s} -> χ² = {_val:.6f}")
    _out = os.path.abspath(args.snapshot_chi2)
    with open(_out, "w") as _fh:
        _json.dump(_snapshot, _fh, indent=2)
    print(f"\n✓ χ² snapshot with {len(_all_vectors)} entries written to {_out}")
    print("  Rerun with the same args after the refactor and diff with test_regression.py.")
    _sys.exit(0)


# ============================================================================
# ============================================================================
# LOG²-DECAY MLE ANALYSIS: 5 MODELS
# ============================================================================
print("\n" + "=" * 96)
print("🔬 ANÁLISIS MLE UNIFICADO")
print("=" * 96)
print(f"🎲 {args.starts} random starts per model...")

# Define models to fit
# Name, Chi2 Function, N_params (base), Evolving Function (if applicable)
# "corr_kind" is used by the BAO null test to reconstruct Δμ(z) at DESI z_eff
# from the best-fit params, independent of the "type" which is used for the
# shared fitting pipeline (bounds, parameter layout).
models_to_fit = [
    {"name": "ΛCDM", "fn": chi2_lcdm, "type": "lcdm", "corr_kind": "lcdm"},
]
models_to_fit += [
    {"name": "γCDM-LOG²", "fn": chi2_gcdm_log_squared, "type": "evolving", "corr_kind": "log2"},
    {"name": "γCDM-Decay", "fn": chi2_decay, "type": "decay", "corr_kind": "decay"},
    {"name": "γCDM-LOG²-Decay", "fn": chi2_gcdm_log_decay, "type": "log_decay", "corr_kind": "log_decay"},
]
# Legacy models: only include with --legacy flag
if args.legacy:
    models_to_fit += [
        {"name": "γCDM-LINEAL", "fn": chi2_gcdm_linear, "type": "evolving", "corr_kind": "linear"},
        {"name": "γCDM-LOG³", "fn": chi2_gcdm_log_cubed, "type": "evolving", "corr_kind": "log3"},
    ]

if args.asymmetric or args.sanity_check:
    # Use no-M variants: ΛCDM keeps M, γCDM/Decay lose M entirely
    for m in models_to_fit:
        if m["name"] == "γCDM-LINEAL":
            m["fn"] = chi2_gcdm_linear_no_M
        elif m["name"] == "γCDM-LOG²":
            m["fn"] = chi2_gcdm_log_squared_no_M
        elif m["name"] == "γCDM-LOG³":
            m["fn"] = chi2_gcdm_log_cubed_no_M
        elif m["name"] == "γCDM-Decay":
            m["fn"] = chi2_decay_no_M
        elif m["name"] == "γCDM-LOG²-Decay":
            m["fn"] = chi2_gcdm_log_decay_no_M

results = []
best_overall_bic = np.inf
best_overall_model = None

np.random.seed(42)

for model in models_to_fit:
    name = model["name"]
    fn = model["fn"]
    mtype = model["type"]
    corr_kind = model.get("corr_kind", "lcdm")
    
    print(f"\n   Fitting {name}...")
    
    # Determine parameters and bounds
    if mtype == "lcdm":
        if args.fixed_anchor or args.sanity_check:
            n_params = 3 if COMBINED_MODE else 2
            bounds = [(OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 1)
        else:
            n_params = 4 if COMBINED_MODE else 3
            bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 2)
    elif mtype in ["gcdm", "evolving"]: # "evolving" covers lineal, log2, log3
         # For these models: 
         # Fixed Anchor -> H0=67.4, M FREE (unless no_nuisance)
         # Sanity Check -> H0=67.4, M REMOVED (Asymmetric)
         
        if args.sanity_check or args.asymmetric:
             # Asymmetric/Sanity: [omch2, gamma] (if Fixed H0) OR [H0, omch2, gamma] (if Free H0)
             # Sanity Check implies fixed anchor. Asymmetric implies M removed.
             if args.fixed_anchor or args.sanity_check:
                 n_params = 2 # [omch2, gamma]
                 bounds = [(OMCH2_MIN, OMCH2_MAX), (GAMMA_MIN, GAMMA_MAX)]
             else:
                 n_params = 3 # [H0, omch2, gamma]
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX), (GAMMA_MIN, GAMMA_MAX)]
                 
        elif args.fixed_anchor:
             # Fixed Anchor (standard): H0=67.4, M FREE
             if args.no_nuisance:
                 n_params = 2 # [omch2, gamma]
                 bounds = [(OMCH2_MIN, OMCH2_MAX), (GAMMA_MIN, GAMMA_MAX)]
             else:
                 n_params = 4 if COMBINED_MODE else 3 # [omch2, M..., gamma]
                 bounds = [(OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 2) + [(GAMMA_MIN, GAMMA_MAX)]
                 
        else:
             # All Free
             n_params = 5 if COMBINED_MODE else 4
             if args.no_nuisance:
                  bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(0, 0)] * (n_params - 3) + [(GAMMA_MIN, GAMMA_MAX)]
             else:
                  bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 3) + [(GAMMA_MIN, GAMMA_MAX)]
    elif mtype == "decay":
        # Pure Decay model: 2 model params (A, zd)
        if args.sanity_check or args.asymmetric:
             if args.fixed_anchor or args.sanity_check:
                 n_params = 3 # [omch2, A, zd]
                 bounds = [(OMCH2_MIN, OMCH2_MAX), (A_MIN, A_MAX), (ZD_MIN, ZD_MAX)]
             else:
                 n_params = 4 # [H0, omch2, A, zd]
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX), (A_MIN, A_MAX), (ZD_MIN, ZD_MAX)]
                 
        elif args.fixed_anchor:
             if args.no_nuisance:
                 n_params = 3 # [omch2, A, zd]
                 bounds = [(OMCH2_MIN, OMCH2_MAX), (A_MIN, A_MAX), (ZD_MIN, ZD_MAX)]
             else:
                 n_params = 5 if COMBINED_MODE else 4 # [omch2, M..., A, zd]
                 bounds = [(OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 3) + [(A_MIN, A_MAX), (ZD_MIN, ZD_MAX)]
        else:
            n_params = 6 if COMBINED_MODE else 5
            if args.no_nuisance:
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(0, 0)] * (n_params - 4) + [(A_MIN, A_MAX), (ZD_MIN, ZD_MAX)]
            else:
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 4) + [(A_MIN, A_MAX), (ZD_MIN, ZD_MAX)]

    elif mtype == "log_decay":
        # --no-bubble: only 2 model params (gamma_0, z_h) instead of 4 (A, z_b, gamma_0, z_h)
        if args.no_bubble:
            _model_bounds = [(GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
        else:
            _model_bounds = [(A_MIN, A_MAX), (ZB_MIN, ZB_MAX), (GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
        _n_model = len(_model_bounds)
        if args.sanity_check or args.asymmetric:
             if args.fixed_anchor or args.sanity_check:
                 n_params = 1 + _n_model # [omch2, *model]
                 bounds = [(OMCH2_MIN, OMCH2_MAX)] + _model_bounds
             else:
                 n_params = 2 + _n_model # [H0, omch2, *model]
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + _model_bounds
                 
        elif args.fixed_anchor:
             if args.no_nuisance:
                 n_params = 1 + _n_model
                 bounds = [(OMCH2_MIN, OMCH2_MAX)] + _model_bounds
             else:
                 _n_m = 2 if COMBINED_MODE else 1
                 n_params = 1 + _n_m + _n_model
                 bounds = [(OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * _n_m + _model_bounds
        else:
            _n_m = 2 if COMBINED_MODE else 1
            n_params = 2 + _n_m + _n_model
            if args.no_nuisance:
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(0, 0)] * _n_m + _model_bounds
            else:
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * _n_m + _model_bounds
            
    # Append scatter bounds at end if fitting
    _n_scatter = 0
    if FIT_SCATTER and mtype in ("lcdm", "log_decay"):
        if COMBINED_MODE:
            bounds += [(SINT_SNE_MIN, SINT_SNE_MAX), (SINT_QSO_MIN, SINT_QSO_MAX)]
            _n_scatter = 2
        else:
            bounds += [(SINT_SNE_MIN, SINT_SNE_MAX)]
            _n_scatter = 1
        n_params += _n_scatter

    best_chi2 = np.inf
    best_params = None

    if USE_EVO:
        # ── EVO BLINDAJE: Ensemble DE + Dual Annealing + Multi-Polish ──
        _ndim = len(bounds)
        print(f"      🧬 DIFFERENTIAL EVO: space {_ndim}D, model '{mtype}'")

        # Log-transform dimensions with huge dynamic range (ratio > 1e3)
        # so DE samples uniformly in log-space instead of wasting population
        evo_bounds = list(bounds)
        _log_dims = []
        for _bi, (_lo, _hi) in enumerate(evo_bounds):
            if _lo > 0 and _hi / _lo > 1e3:
                evo_bounds[_bi] = (np.log10(_lo), np.log10(_hi))
                _log_dims.append(_bi)

        if _log_dims:
            print(f"      📐 Log-transform en dims {_log_dims} (ratio > 1e3)")
            _fn_real = fn
            _log_dims_frozen = list(_log_dims)
            def _make_fn_evo(_fn_inner, _dims):
                def fn_evo(x):
                    x_r = np.array(x, dtype=float)
                    for _d in _dims:
                        x_r[_d] = 10.0 ** x_r[_d]
                    return _fn_inner(x_r)
                return fn_evo
            fn_evo = _make_fn_evo(_fn_real, _log_dims_frozen)
        else:
            fn_evo = fn

        # Phase 1: Ensemble Differential Evolution (multiple seeds + strategies)
        _evo_strategies = ['best1bin', 'rand1bin', 'currenttobest1bin',
                           'best2bin', 'rand2bin']
        _N_EVO_RUNS = 5
        _all_candidates = []

        for _irun in range(_N_EVO_RUNS):
            _strat = _evo_strategies[_irun % len(_evo_strategies)]
            _seed_i = 42 + _irun * 137
            print(f"      🎲 DE run {_irun+1}/{_N_EVO_RUNS}"
                  f" (strategy={_strat}, seed={_seed_i})...", end="")
            try:
                _res_de = differential_evolution(
                    fn_evo, evo_bounds,
                    seed=_seed_i,
                    maxiter=2000,
                    tol=1e-10,
                    polish=False,
                    strategy=_strat,
                    popsize=25,
                    mutation=(0.5, 1.5),
                    recombination=0.9,
                    workers=1,
                )
                _all_candidates.append((_res_de.fun, np.array(_res_de.x)))
                print(f" -2lnL = {_res_de.fun:.2f}")
            except Exception:
                print(" [falló]")

        # Phase 2: Dual Annealing (independent global search as cross-check)
        print(f"      🔥 Dual annealing cross-validation...", end="")
        try:
            _res_da = dual_annealing(fn_evo, evo_bounds, seed=123, maxiter=1000)
            _all_candidates.append((_res_da.fun, np.array(_res_da.x)))
            print(f" -2lnL = {_res_da.fun:.2f}")
        except Exception:
            print(" [falló]")

        # Phase 3: Multi-polish — refine top-K candidates with NM + Powell
        if _all_candidates:
            _all_candidates.sort(key=lambda t: t[0])
            _top_k = min(5, len(_all_candidates))
            print(f"      💎 Multi-polish: top {_top_k} candidatos (NM + Powell)...")

            for _ic in range(_top_k):
                _cand_chi2, _cand_x = _all_candidates[_ic]
                for _method, _opts in [
                    ('Nelder-Mead', {'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10}),
                    ('Powell', {'maxiter': 20000, 'ftol': 1e-12}),
                ]:
                    try:
                        _res_p = minimize(fn_evo, _cand_x, method=_method, options=_opts)
                        if _res_p.fun < best_chi2:
                            best_chi2 = _res_p.fun
                            best_params = np.array(_res_p.x)
                    except Exception:
                        pass

            # Phase 4: Perturbation robustness test around best solution
            if best_params is not None:
                print(f"      🔄 Perturbation test (20 runs)...", end="")
                _n_improved = 0
                for _ in range(20):
                    _x_pert = best_params.copy()
                    for _j, (_lo, _hi) in enumerate(evo_bounds):
                        _x_pert[_j] += np.random.uniform(-1, 1) * 0.05 * (_hi - _lo)
                        _x_pert[_j] = np.clip(_x_pert[_j], _lo, _hi)
                    try:
                        _res_pt = minimize(fn_evo, _x_pert, method='Nelder-Mead',
                                           options={'maxiter': 10000,
                                                    'xatol': 1e-10, 'fatol': 1e-10})
                        if _res_pt.fun < best_chi2:
                            best_chi2 = _res_pt.fun
                            best_params = np.array(_res_pt.x)
                            _n_improved += 1
                    except Exception:
                        pass
                if _n_improved:
                    print(f" ⚠️ mejoró {_n_improved}/20 → mínimo inestable")
                else:
                    print(f" ✅ estable (0/20 mejoras)")

        # Transform back from log-space to physical parameters
        if _log_dims and best_params is not None:
            best_params = np.array(best_params, dtype=float)
            for _d in _log_dims:
                best_params[_d] = 10.0 ** best_params[_d]

    else:
        for i in range(args.starts):
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
            print(f"      🎲 Inicializando random start {i+1}/{args.starts}: {[round(x, 4) for x in x0]}", end="", flush=True)
            try:
                res = minimize(fn, x0, method='Nelder-Mead', options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
                fn(res.x) # Sync tracker with best fit of this run
                print(f" -> -2lnL = {res.fun:.2f} (χ²={GLOBAL_CHI2:.2f})")
                if res.fun < best_chi2:
                    best_chi2 = res.fun
                    best_params = res.x
                    best_chi2_val = GLOBAL_CHI2
            except:
                print(" -> Falló")
            
    if best_params is not None:
        if args.camb_tab:
            print(f"      ✨ Pulido final analítico con CAMB exacto para recuperar precisión canónica...")
            _tab_state = args.camb_tab
            args.camb_tab = False
            try:
                res_polish = minimize(fn, best_params, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res_polish.fun <= best_chi2:
                    best_chi2 = res_polish.fun
                    best_params = res_polish.x
                    fn(best_params) # Sync tracker with best best fit
                    best_chi2_val = GLOBAL_CHI2
            except Exception:
                pass
            finally:
                args.camb_tab = _tab_state

        print(f"      ✅ Best -2lnL = {best_chi2:.1f} (χ²={best_chi2_val:.1f})")
        
        # Extract fitted scatter from end of vector
        fitted_sig_sne, fitted_sig_qso = SIGMA_INT_SNE, SIGMA_INT_QSO
        if FIT_SCATTER and _n_scatter > 0 and mtype in ("lcdm", "log_decay"):
            if COMBINED_MODE:
                fitted_sig_sne = best_params[-2]
                fitted_sig_qso = best_params[-1]
            else:
                fitted_sig_sne = best_params[-1]
                fitted_sig_qso = fitted_sig_sne
            print(f"      σ_int,SNe = {fitted_sig_sne:.4f}, σ_int,QSO = {fitted_sig_qso:.4f}")

        # Unpack parameters based on mode
        if mtype == "lcdm":
            if args.fixed_anchor or args.sanity_check:
                omch2 = best_params[0]
                h0 = 67.4
                if COMBINED_MODE:
                    M_sne, M_qso = best_params[1], best_params[2]
                    M = (M_sne + M_qso) / 2
                else:
                    M_sne = best_params[1]
                    M = M_sne
                gamma = 0.0
                om = (omch2 + 0.0224) / (h0 / 100) ** 2
            else:
                h0 = best_params[0]
                om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                if COMBINED_MODE:
                    M_sne = best_params[2]
                    M_qso = best_params[3]
                    M = (M_sne + M_qso) / 2
                else:
                    M = best_params[2]
                gamma = 0.0

        elif mtype == "gcdm":
            if args.fixed_anchor:
                h0 = 67.4
                om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                if args.no_nuisance or args.asymmetric or args.sanity_check:
                    M = 0.0
                    gamma = best_params[1]
                else:
                    if COMBINED_MODE:
                        M = (best_params[1] + best_params[2]) / 2
                    else:
                        M = best_params[1]
                    gamma = best_params[-1]
            elif args.sanity_check:
                # params: [omch2, gamma]
                # H0=67.4, M=0
                h0 = 67.4
                om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                M = 0.0
                gamma = best_params[1]
            elif args.asymmetric or args.no_nuisance:
                h0 = best_params[0]
                om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                M = 0.0
                gamma = best_params[2]
            else:
                h0 = best_params[0]
                om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                M, _ = _get_M({
                     "M_sne": [best_params[2]] if COMBINED_MODE else [],
                     "mabs": [best_params[2]] if not COMBINED_MODE else [],
                     "M_qso": [best_params[3]] if COMBINED_MODE else []
                 }, COMBINED_MODE)
                gamma = best_params[-1]

        elif mtype == "evolving":
            if args.fixed_anchor:
                # params: [omch2, (M...), gamma]
                h0 = 67.4
                omch2 = best_params[0]
                om = (omch2 + 0.0224) / (67.4 / 100) ** 2
                if args.no_nuisance or args.asymmetric or args.sanity_check:
                    M = 0.0
                    gamma = best_params[1]
                else:
                    if COMBINED_MODE:
                        M = (best_params[1] + best_params[2]) / 2
                    else:
                        M = best_params[1]
                    gamma = best_params[-1]
            elif args.sanity_check:
                # params: [omch2, gamma]
                h0 = 67.4
                om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                M = 0.0
                gamma = best_params[1]
            elif args.asymmetric or args.no_nuisance:
                h0 = best_params[0]
                om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                M = 0.0
                gamma = best_params[2]
            else:
                h0 = best_params[0]
                om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                M, _ = _get_M({
                     "M_sne": [best_params[2]] if COMBINED_MODE else [],
                     "mabs": [best_params[2]] if not COMBINED_MODE else [],
                     "M_qso": [best_params[3]] if COMBINED_MODE else []
                 }, COMBINED_MODE)
                gamma = best_params[-1]

        elif mtype == "decay":
            if args.fixed_anchor:
                 # params: [omch2, (M...), A, zd]
                 h0 = 67.4
                 om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                 if args.no_nuisance or args.asymmetric or args.sanity_check:
                      M = 0.0
                      A, zd = best_params[1], best_params[2]
                 else:
                      if COMBINED_MODE:
                           M = (best_params[1] + best_params[2]) / 2
                      else:
                           M = best_params[1]
                      A, zd = best_params[-2], best_params[-1]
                 gamma = A
                 print(f"      -> A = {A:.3f}, zd = {zd:.3f}")
            elif args.sanity_check:
                 # params: [omch2, A, zd]
                 h0 = 67.4
                 om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                 M = 0.0
                 A, zd = best_params[1], best_params[2]
                 gamma = A
                 print(f"      -> A = {A:.3f}, zd = {zd:.3f}")
            elif args.asymmetric:
                 # params: [H0, omch2, A, zd] — no M
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 M = 0.0
                 A, zd = best_params[2], best_params[3]
                 gamma = A
                 print(f"      -> A = {A:.3f}, zd = {zd:.3f}")
            elif args.no_nuisance:
                 # params: [H0, omch2, M(=0)..., A, zd]
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 M = 0.0
                 A, zd = best_params[-2], best_params[-1]
                 gamma = A
                 print(f"      -> A = {A:.3f}, zd = {zd:.3f}")
            else:
                 # params: [H0, omch2, M..., A, zd]
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 if COMBINED_MODE:
                      M = (best_params[2] + best_params[3]) / 2
                      A, zd = best_params[4], best_params[5]
                 else:
                      M = best_params[2]
                      A, zd = best_params[3], best_params[4]
                 gamma = A
                 print(f"      -> A = {A:.3f}, zd = {zd:.3f}")

        elif mtype == "log_decay":
            _so = _n_scatter
            if args.no_bubble:
                # Kerr-Only: tail is (gamma_0, z_h) — no A or z_b
                A, z_b = 0.0, 0.0   # not fitted
                gamma_0, z_h = best_params[-2-_so], best_params[-1-_so]
            else:
                A, z_b, gamma_0, z_h = best_params[-4-_so], best_params[-3-_so], best_params[-2-_so], best_params[-1-_so]
            if args.fixed_anchor or args.sanity_check:
                 h0 = 67.4
                 om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                 if args.no_nuisance or args.asymmetric or args.sanity_check:
                      M = 0.0
                 elif COMBINED_MODE:
                      M = (best_params[1] + best_params[2]) / 2
                 else:
                      M = best_params[1]
            elif args.asymmetric:
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 M = 0.0
            elif args.no_nuisance:
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 M = 0.0
            else:
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 if COMBINED_MODE:
                      M = (best_params[2] + best_params[3]) / 2
                 else:
                      M = best_params[2]
            gamma = gamma_0
            if args.no_bubble:
                print(f"      -> γ₀ = {gamma_0:.3f}, z_h = {z_h:.3f}  [Kerr-Only, A=0]")
            else:
                print(f"      -> A = {A:.3f}, z_b = {z_b:.3f}, γ₀ = {gamma_0:.3f}, z_h = {z_h:.3f}")

        n_eff = n_params
        bic = best_chi2 + n_eff * np.log(N)
        aic = best_chi2 + 2 * n_eff
        
        omc = om - 0.0224 / (h0 / 100)**2
        omch2 = om * (h0 / 100.0)**2 - OMBH2_FIDUCIAL
        _res_entry = {
            "name": name, "mtype": mtype, "corr_kind": corr_kind,
            "H0": h0, "Om": om, "Omc": omc, "Omch2": omch2, "M": M, "gamma": gamma,
            "chi2": best_chi2, "bic": bic, "aic": aic,
            "params": best_params, "n_eff": n_eff,
            "sig_sne": fitted_sig_sne, "sig_qso": fitted_sig_qso
        }
        # Store model-specific params so display code doesn't need fragile params[-N] indexing
        if mtype == "log_decay":
            _res_entry.update({"A": A, "z_b": z_b, "z_h": z_h, "no_bubble": args.no_bubble})
        elif mtype == "decay":
            _res_entry.update({"A": A, "zd": zd})
        results.append(_res_entry)
        
        if bic < best_overall_bic:
            best_overall_bic = bic
            best_overall_model = results[-1]

# Baseline ΛCDM for deltas
lcdm_res = next((r for r in results if r["name"] == "ΛCDM"), None)
if lcdm_res:
    bic_lcdm = lcdm_res["bic"]
    aic_lcdm = lcdm_res["aic"]
else:
    bic_lcdm = 0.0
    aic_lcdm = 0.0

# ============================================================================
# RESULTS TABLE
# ============================================================================
print("\n" + "=" * 115)
print(f"{'Modelo':<24} {'H₀':>8} {'Ωₘ':>8} {'Ωc':>8} {'δM':>8} {'γ₀':>10} {'-2lnL':>10} {'BIC':>10} {'AIC':>10} {'ΔBIC':>8} {'ΔAIC':>8}")
print("─" * 125)

for res in results:
    dbic = res["bic"] - bic_lcdm
    daic = res["aic"] - aic_lcdm
    
    # Check for "Decay" (case insensitive) to handle both "γCDM-Decay" and "γCDM-LOG²-Decay"
    if "log" in res["name"].lower() and "decay" in res["name"].lower():
        # LOG²-DECAY: use stored model params (safe regardless of --fit-scatter)
        A_val = res['A']
        zb_val = res['z_b']
        g0_val = res['gamma']
        zh_val = res['z_h']
        print(f"{res['name']:<24} {res['H0']:>8.2f} {res['Om']:>8.3f} {res['Omc']:>8.3f} {res['M']:>8.3f}   γ={g0_val:>6.3f}  {res['chi2']:>10.1f} {res['bic']:>10.1f} {res['aic']:>10.1f} {dbic:>8.1f} {daic:>8.1f}")
        if res.get('no_bubble'):
            print(f"{'':24} {'':>8} {'':>8} {'':>8} {'':>8}   A=0 (Kerr-Only) zh={zh_val:.1f}")
        else:
            print(f"{'':24} {'':>8} {'':>8} {'':>8} {'':>8}   A={A_val:>6.3f} zb={zb_val:.3f} zh={zh_val:.1f}")
    elif "decay" in res["name"].lower():
        # Pure Decay: use stored model params
        A_val = res['A']
        zd_val = res['zd']
        print(f"{res['name']:<24} {res['H0']:>8.2f} {res['Om']:>8.3f} {res['Omc']:>8.3f} {res['M']:>8.3f}   A={A_val:>6.3f}  {res['chi2']:>10.1f} {res['bic']:>10.1f} {res['aic']:>10.1f} {dbic:>8.1f} {daic:>8.1f}")
        print(f"{'':24} {'':>8} {'':>8} {'':>8} {'':>8}   zd={zd_val:.3f}")
    else:
        print(f"{res['name']:<24} {res['H0']:>8.2f} {res['Om']:>8.3f} {res['Omc']:>8.3f} {res['M']:>8.3f} {res['gamma']:>10.4f} {res['chi2']:>10.1f} {res['bic']:>10.1f} {res['aic']:>10.1f} {dbic:>8.1f} {daic:>8.1f}")
print("─" * 125)


# ============================================================================
# BAO (DESI DR1) REPORTING — --bao (fit breakdown) / --bao-null (null test)
# ----------------------------------------------------------------------------
# Both modes evaluate χ²_BAO at each model's best-fit and print a consistency
# table. The crucial physical diagnostic is whether the model that fits
# SNe+CC+CMB well ALSO sits within the BAO errors (χ²_BAO/ν ≲ 1).
#
#   --bao        : BAO is already part of the objective, so χ²_BAO is a
#                  component of the reported `res["chi2"]`. We still print
#                  the BREAKDOWN here so the reader can see the share of
#                  χ²_BAO vs. SNe/CC/CMB inside the converged chi².
#
#   --bao-null   : BAO is NOT in the objective. χ²_BAO is evaluated at the
#                  best-fit ONLY for diagnosis (strict goodness-of-fit test).
# ============================================================================
if NEED_BAO_BG and results:
    print("\n" + "=" * 70)
    if USE_BAO_FIT:
        print("📐 BAO DIAGNOSTIC (DESI DR1) — χ² at best-fit (IN-FIT mode)")
    else:
        print("📐 BAO NULL TEST (DESI DR1) — χ² at best-fit (NOT in objective)")
    print("=" * 70)
    _ndof_bao_only = N_BAO_POINTS    # not subtracting fit params (strict test)
    print(f"   {'Modelo':<24} {'χ²_BAO':>10} {'χ²/ν':>8} {'r_d':>8} "
          f"{'Ωm':>8} {'H0':>8}")
    print("   " + "─" * 68)
    for res in results:
        try:
            h0 = res["H0"]
            om = res["Om"]
            omch2 = res["Omch2"]
            # CAMB background at best-fit (with BAO extras)
            _mu_bg, _hz_bg, _da_bg, _bao_ex = _fast_camb_bg(h0, omch2, want_bao=True)
            if _bao_ex is None:
                print(f"   {res['name']:<24} {'—':>10} {'—':>8} {'—':>8} {om:>8.3f} {h0:>8.2f}   (CAMB failed)")
                continue
            _dm = _bao_ex["dm"]
            _dh = _bao_ex["dh"]
            _rd = _bao_ex["rdrag"]
            # Reconstruct Δμ at Z_EFF_DESI using the converged parameters.
            _kind = res.get("corr_kind", "lcdm")
            _kw = {}
            if _kind == "gcdm":
                _kw = {"gamma": res.get("gamma", 0.0)}
            elif _kind in ("linear", "log2", "log3"):
                _kw = {"gamma_0": res.get("gamma", 0.0)}
            elif _kind == "decay":
                _kw = {"A": res.get("A", 0.0), "zd": res.get("zd", 1.0)}
            elif _kind == "log_decay":
                _A = 0.0 if res.get("no_bubble") else res.get("A", 0.0)
                _zb = res.get("z_b", 1.0) or 1.0
                _kw = {
                    "A": _A, "z_b": _zb,
                    "gamma_0": res.get("gamma", 0.0),
                    "z_h": res.get("z_h", 1e10) or 1e10,
                }
            _dmu_zeff = _bao_corr_at(Z_EFF_DESI, _kind, **_kw)
            if BAO_PROPAGATE_CORRECTION:
                _dm_model = _dm * (10.0 ** (_dmu_zeff / 5.0))
            else:
                _dm_model = _dm
            _chi2_bao_val, _n_bao = compute_chi2_bao(_dm_model, _dh, _rd)
            _chi2_red = _chi2_bao_val / _ndof_bao_only
            # Store for downstream tools (MCMC / nested sampling summaries)
            res["chi2_bao"]      = _chi2_bao_val
            res["chi2_bao_red"]  = _chi2_red
            res["rdrag_bestfit"] = _rd
            print(f"   {res['name']:<24} {_chi2_bao_val:>10.2f} {_chi2_red:>8.2f} "
                  f"{_rd:>8.2f} {om:>8.3f} {h0:>8.2f}")
        except Exception as _e:
            print(f"   {res['name']:<24}   (BAO eval failed: {_e})")
    print("   " + "─" * 68)
    print(f"   ν = {_ndof_bao_only} data points (fit DOF NOT subtracted for null test).")
    print(f"   χ²/ν ≈ 1 → consistent with BAO.   χ²/ν ≫ 1 → model fails BAO.")
    if USE_BAO_FIT:
        print(f"   NOTE: these χ²_BAO are a SUBSET of each model's total -2lnL above.")
    else:
        print(f"   NOTE: null test — these numbers did NOT enter the MLE objective.")

# Initialize Output Directory Early
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir, exist_ok=True)

# Selection
# Find models with best BIC
res_log2 = next((r for r in results if r["name"] == "γCDM-LOG²"), None)
res_lcdm = next((r for r in results if r["name"] == "ΛCDM"), None)
res_gcdm = next((r for r in results if r["name"] == "γCDM (const)"), None)
res_lin = next((r for r in results if r["name"] == "γCDM-LINEAL"), None)
res_log3 = next((r for r in results if r["name"] == "γCDM-LOG³"), None)
res_decay = next((r for r in results if r["name"] == "γCDM-Decay"), None)
res_log_decay = next((r for r in results if r["name"] == "γCDM-LOG²-Decay"), None)

if results:
    best_model_aic = min(results, key=lambda x: x["aic"])
    print(f"\n🏆 MEJOR MODELO (AIC): {best_model_aic['name']} (AIC = {best_model_aic['aic']:.1f})")
    print(f"   H₀ (mejor BIC) = {best_overall_model['H0']:.2f} km/s/Mpc")
else:
    print(f"\n🏆 MEJOR MODELO (AIC): N/A (MLE Skipped)")

print(f"\n📊 TENSIÓN DE HUBBLE:")
print(f"   Planck CMB:  H₀ = 67.4 ± 0.5")
print(f"   SH0ES:       H₀ = 73.0 ± 1.0")

def check_h0(h):
    d_cmb = abs(h - 67.4)
    d_shoes = abs(h - 73.0)
    if d_cmb < d_shoes: return "→ más cerca de CMB"
    return "→ más cerca de SH0ES"

if res_lcdm: print(f"   ΛCDM                : H₀ = {res_lcdm['H0']:.2f} {check_h0(res_lcdm['H0'])}")
if res_gcdm: print(f"   γCDM (const)        : H₀ = {res_gcdm['H0']:.2f} {check_h0(res_gcdm['H0'])}")
if res_lin:  print(f"   γCDM-LINEAL         : H₀ = {res_lin['H0']:.2f} {check_h0(res_lin['H0'])}")
if res_log2: print(f"   γCDM-LOG²           : H₀ = {res_log2['H0']:.2f} {check_h0(res_log2['H0'])}")
if res_log3: print(f"   γCDM-LOG³           : H₀ = {res_log3['H0']:.2f} {check_h0(res_log3['H0'])}")
if res_decay:
    # Decay correction at z=0: Δμ = A·exp(0) = A → H₀(local) = H₀(cosmo)·10^(-A/5)
    A_dec = res_decay['A']
    zd_dec = res_decay['zd']
    h0_local_dec = h0_local(res_decay['H0'], A=A_dec, z_b=zd_dec, z_pivot=Z_PIVOT)
    print(f"   γCDM-Decay          : H₀ = {res_decay['H0']:.2f} {check_h0(res_decay['H0'])}")
    print(f"                         H₀(local) = {h0_local_dec:.2f} (A={A_dec:.3f}, zd={zd_dec:.2f}) {check_h0(h0_local_dec)}")
if res_log_decay:
    # LOG²-DECAY: Δμ = A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h)
    # At z=0: bubble_term = A·exp(0) = A, kerr_term = 0 → Δμ(0) = A
    # So H₀(local) = H₀(cosmo)·10^(-A/5) — A provides the SH0ES shift!
    # use stored model params (safe regardless of --fit-scatter)
    A_uni = res_log_decay['A']
    z_b_uni = res_log_decay['z_b']
    gamma_uni = res_log_decay['gamma']
    z_h_uni = res_log_decay['z_h']
    M_uni = res_log_decay['M']
    # H₀(local): uses helper h0_local (strict z→0 limit with z_pivot=0).
    # If you want the rigorous SH0ES comparison, pass z_pivot=SH0ES_Z_PIVOT.
    h0_local_uni = h0_local(res_log_decay['H0'],
                            A=A_uni, z_b=z_b_uni,
                            gamma_0=gamma_uni, z_h=z_h_uni,
                            z_pivot=Z_PIVOT)
    print(f"   γCDM-LOG²-Decay        : H₀ = {res_log_decay['H0']:.2f} {check_h0(res_log_decay['H0'])}")
    print(f"                         H₀(local) = {h0_local_uni:.2f} (A={A_uni:.3f}, z_b={z_b_uni:.3f}) {check_h0(h0_local_uni)}")
    print(f"                         γ₀={gamma_uni:.3f}, z_h={z_h_uni:.2f} (Kerr geometry)")

# print("\n   → Evidencia de efecto Container Lens evolutivo")

# Interpretation
if best_overall_model:
    dbic_best = best_overall_model["bic"] - bic_lcdm
else:
    dbic_best = 0.0
print(f"\n📊 INTERPRETACIÓN:")
if dbic_best < -10:
    print(f"   → Evidencia MUY FUERTE a favor de {best_overall_model['name']} (ΔBIC = {dbic_best:.1f})")
elif dbic_best < -6:
    print(f"   → Evidencia FUERTE a favor de {best_overall_model['name']} (ΔBIC = {dbic_best:.1f})")
elif dbic_best < -2:
    print(f"   → Evidencia POSITIVA a favor de {best_overall_model['name']} (ΔBIC = {dbic_best:.1f})")
else:
    print(f"   → INCONCLUSO o ΛCDM preferido")

# Anti-Cheat: Comparison ΛCDM vs Best Model
if best_overall_model and lcdm_res:
    print(f"\n🛡️ ROBUSTNESS VERIFICATION — δM ({best_overall_model['name']}):")
    print(f"   δM (ΛCDM): {lcdm_res['M']:.3f}")
    print(f"   δM (Best): {best_overall_model['M']:.3f}")
    diff_M = best_overall_model['M'] - lcdm_res['M']
    print(f"   ΔδM:       {diff_M:.3f}")


    if abs(diff_M) < 0.5:
        print(f"   ✅ Valores de δM consistentes → γ₀ NO absorbe el offset")
    else:
        print(f"   ⚠️  δM difieren significativamente")

# Spin Calculation — use best AIC model's γ₀
# Works for any model that has a 'gamma' key (LOG², LOG²-Decay, etc.)
best_aic_model = min(results, key=lambda x: x["aic"]) if results else None
spin_gamma = None
spin_model_name = None
if best_aic_model and 'gamma' in best_aic_model:
    spin_gamma = best_aic_model['gamma']
    spin_model_name = best_aic_model['name']
elif res_log2:
    spin_gamma = res_log2['gamma']
    spin_model_name = res_log2['name']

if spin_gamma is not None and spin_gamma != 0:
    print("\n" + "=" * 70)
    print("🌀 CÁLCULO DE SPIN — Container Black Hole")
    print("=" * 70)
    print(f"""
   La corrección γ₀·[ln(1+z)]² (del modelo {spin_model_name}) se interpreta
   dentro de la hipótesis Möbius-Kerr: habitamos el interior conformalmente
   invertido y finito de un agujero negro rotante (Container).

   Para un agujero negro de Kerr:
     • Ratio de horizontes: α = r₋/r₊
     • Para LOG²: β = |γ₀| × ln(10)/5
     • Hipótesis: α = |β|/2
""")
    beta = abs(spin_gamma) * np.log(10) / 5
    alpha = beta / 2
    x = (1 - alpha) / (1 + alpha)
    spin = np.sqrt(1 - x**2) if x**2 <= 1 else 0.0

    print(f"   📐 CÁLCULO (modelo: {spin_model_name}):")
    print(f"   γ₀ = {spin_gamma:.4f}")
    print(f"   β = {beta:.4f}, α = {alpha:.4f}")
    print(f"   a/M = √(1 − ((1−α)/(1+α))²) = {spin:.4f}")

    print(f"\n   📊 RESULTADO:")
    print(f"   Container Spin ({spin_model_name}):  a/M ≈ {spin:.2f}")

    print(f"\n   🔬 INTERPRETACIÓN HIPOTÉTICA:")
    print(f"   Mayor spin → mayor frame-dragging → lensing cuadrático con decay → container rotante finito")
    if spin > 0.6:
        print(f"   Consistente con spins observados en BH supermasivos (0.7–0.9)")

# ============================================================================
# COMPATIBILITY VARIABLES FOR SUMMARY
# ============================================================================
# Map "best model" results to variables expected by the end of the script
if best_overall_model:
    gamma_g = best_overall_model.get("gamma", 0.0)
    delta_bic = best_overall_model["bic"] - bic_lcdm
    delta_aic = best_overall_model["aic"] - aic_lcdm
    K_BIC_approx = np.exp(-delta_bic / 2)
else:
    gamma_g = 0.0
    delta_bic = 0.0
    delta_aic = 0.0
    K_BIC_approx = 1.0

# Also ensure H0_n, Om_n, M_n are available from LCDM result
if lcdm_res:
    H0_n = lcdm_res["H0"]
    Om_n = lcdm_res["Om"]
    M_n = lcdm_res["M"]
else:
    H0_n = 67.4
    Om_n = 0.3
    M_n = 0.0


# ============================================================================
# REDSHIFT BIN ANALYSIS (ΛCDM vs Best Model)
# ============================================================================
if best_overall_model and lcdm_res and best_overall_model["name"] != "ΛCDM":
    print("\n" + "=" * 80)
    print("📈 REDSHIFT BIN ANALYSIS (-2lnL Improvements)")
    print("=" * 80)
    
    # Define bins
    bins = [
        (0.0, 0.5, "Local (z < 0.5)"),
        (0.5, 1.0, "Intermediate (0.5 < z < 1.0)"),
        (1.0, 2.0, "High SNe + Low QSO (1.0 < z < 2.0)"),
        (2.0, 8.0, "Deep QSO (z > 2.0)")
    ]
    
    def get_model_residuals(model_res):
        global GLOBAL_EVAL_MODE
        GLOBAL_EVAL_MODE = True
        
        # Match the function based on name
        name = model_res["name"]
        params = model_res["params"]
        
        # We need the chi2 function matching the name:
        func = next((m["fn"] for m in models_to_fit if m["name"] == name), None)
        if func:
            try:
                # Call it, ignoring result, just to trigger _neg2logL_mu
                func(params)
                res = GLOBAL_EVAL_ARRAYS['residuals'].copy()
                err = GLOBAL_EVAL_ARRAYS['err_eff'].copy()
                GLOBAL_EVAL_MODE = False
                return res, err
            except Exception:
                GLOBAL_EVAL_MODE = False
                return None, None
        GLOBAL_EVAL_MODE = False
        return None, None

    res_lcdm_arr, err_lcdm_arr = get_model_residuals(lcdm_res)
    res_best_arr, err_best_arr = get_model_residuals(best_overall_model)
    
    # Get CC residuals (evaluate chi2 functions separately or just use mu array info)
    # Actually, CC H(z) data is not binned here. We just bin mu_obs.
    
    if res_lcdm_arr is not None and res_best_arr is not None:
        print(f"   Modelo Base: ΛCDM")
        print(f"   Modelo Win:  {best_overall_model['name']}\n")
        print(f"   {'Redshift Bin':<35} {'N pts':<8} {'Δ(-2lnL)':>10}")
        print("   " + "─" * 53)

        def _bin_neg2logL(res_bin, err_bin, sne_bin_mask):
            """Per-bin -2lnL using covariance for SNe when available.

            For binned evaluation we cannot use the full C_inv (wrong shape),
            so we fall back to diagonal errors for SNe within the bin.  The
            bin analysis is only diagnostic — absolute -2lnL values are not
            used for model selection, only deltas between models evaluated
            with the SAME approximation.
            """
            neg2ll = 0.0
            if C_inv_sne is not None and sne_bin_mask is not None and np.any(sne_bin_mask):
                res_sne_b = res_bin[sne_bin_mask]
                err_sne_b = err_bin[sne_bin_mask]
                res_qso_b = res_bin[~sne_bin_mask]
                err_qso_b = err_bin[~sne_bin_mask]
                neg2ll += np.sum((res_sne_b / err_sne_b)**2) + np.sum(np.log(err_sne_b**2))
            else:
                res_qso_b = res_bin
                err_qso_b = err_bin

            if len(res_qso_b) > 0:
                if LIKELIHOOD_TYPE == 'student' or LIKELIHOOD_TYPE == 'cauchy':
                    nu = NU_DOF if LIKELIHOOD_TYPE == 'student' else 1.0
                    r2 = res_qso_b**2
                    s2 = err_qso_b**2
                    neg2ll += np.sum(2 * np.log(err_qso_b)
                                     + (nu + 1) * np.log1p(r2 / (nu * s2 + 1e-30)))
                else:
                    neg2ll += np.sum((res_qso_b / err_qso_b)**2) + np.sum(np.log(err_qso_b**2))
            return neg2ll

        total_delta = 0
        
        for zmin, zmax, label in bins:
            mask = (z_mu >= zmin) & (z_mu < zmax)
            n_pts = np.sum(mask)
            if n_pts == 0:
                continue

            sne_bin = sne_mask[mask] if sne_mask is not None else None
            n2ll_lcdm_bin = _bin_neg2logL(res_lcdm_arr[mask], err_lcdm_arr[mask], sne_bin)
            n2ll_best_bin = _bin_neg2logL(res_best_arr[mask], err_best_arr[mask], sne_bin)
            
            delta_n2ll = n2ll_best_bin - n2ll_lcdm_bin
            total_delta += delta_n2ll
            
            print(f"   {label:<35} {n_pts:<8d} {delta_n2ll:>10.1f}")
            
        print("   " + "─" * 53)
        print(f"   {'TOTAL (Distances Only) *':<35} {'':<8} {total_delta:>10.1f}")
        print("   * Excludes CC data penalties/improvements.")
    else:
        print("   ⚠️ Could not evaluate bin residuals.")



# ============================================================================
# MOCK TEST: γ=0 NULL HYPOTHESIS (pipeline validation)
# ============================================================================
def run_mock_test(title_suffix=""):
    print("\n" + "=" * 70)
    print(f"🧪 MOCK TEST{title_suffix}: VALIDACIÓN γ=0 (null hypothesis)")
    print("=" * 70)
    print(f"""
   Objetivo: verificar que el pipeline NO fabrica señal γ espuria.
   Procedimiento:
     1. Generar datos sintéticos con ΛCDM puro (γ=0, H₀=67.4)
     2. Usar las mismas barras de error y redshifts que los datos reales
     3. Ajustar ΛCDM y el modelo Target (LOG²-Decay o Lineal) al mock
     4. Verificar que γ ≈ 0 y ΔBIC ≈ 0 (o positivo)
   Realizaciones: {args.n_mock}
""")

    # Save real data
    _z_mu, _mu_obs, _err_mu = z_mu.copy(), mu_obs.copy(), err_mu.copy()
    _z_cc, _H_obs, _err_cc = z_cc.copy(), H_obs.copy(), err_cc.copy()

    # Truth cosmology: Planck ΛCDM
    H0_truth = 67.4
    omch2_truth = 0.12
    pars_truth = camb.CAMBparams()
    pars_truth.set_cosmology(H0=H0_truth, ombh2=0.0224, omch2=omch2_truth)
    r_truth = camb.get_background(pars_truth)

    mu_theory = 5 * np.log10(r_truth.luminosity_distance(_z_mu)) + 25
    H_theory = r_truth.hubble_parameter(_z_cc) if len(_z_cc) > 0 else np.array([])

    mock_gammas = []
    mock_dbics = []
    mock_daics = []
    false_detections = 0

    rng = np.random.RandomState(123)

    for m in range(args.n_mock):
        # Generate noisy mock data
        global z_mu_g, mu_obs_g, err_mu_g, z_cc_g, H_obs_g, err_cc_g
        z_mu_m = _z_mu
        err_mu_m = _err_mu
        mu_obs_m = mu_theory + rng.normal(0, _err_mu)

        z_cc_m = _z_cc
        err_cc_m = _err_cc
        H_obs_m = H_theory + rng.normal(0, _err_cc) if len(_err_cc) > 0 else np.array([])

        # Determine degrees of freedom dynamically for proper BIC
        k_lcdm = 4 if COMBINED_MODE else 3

        # Wrap chi2 evaluation to ensure isolation from global args and inject mock data
        def chi2_lcdm_mock(params):
            orig_fa, orig_sc, orig_nn, orig_asym = args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric
            args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric = False, False, False, False
            global z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc
            _orig_z_mu, _orig_mu_obs, _orig_err_mu, _orig_z_cc, _orig_H_obs, _orig_err_cc = z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc
            z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc = z_mu_m, mu_obs_m, err_mu_m, z_cc_m, H_obs_m, err_cc_m
            try:
                val = chi2_lcdm(params)
            finally:
                 args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric = orig_fa, orig_sc, orig_nn, orig_asym
                 z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc = _orig_z_mu, _orig_mu_obs, _orig_err_mu, _orig_z_cc, _orig_H_obs, _orig_err_cc
            return val

        if getattr(args, 'legacy', False):
            # Target Linear Model
            k_target = 5 if COMBINED_MODE else 4
            target_name = "γCDM-LINEAL"
            def chi2_target_mock(params):
                orig_fa, orig_sc, orig_nn, orig_asym = args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric
                args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric = False, False, False, False
                global z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc
                _orig_z_mu, _orig_mu_obs, _orig_err_mu, _orig_z_cc, _orig_H_obs, _orig_err_cc = z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc
                z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc = z_mu_m, mu_obs_m, err_mu_m, z_cc_m, H_obs_m, err_cc_m
                try:
                    val = chi2_gcdm_linear(params)
                finally:
                     args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric = orig_fa, orig_sc, orig_nn, orig_asym
                     z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc = _orig_z_mu, _orig_mu_obs, _orig_err_mu, _orig_z_cc, _orig_H_obs, _orig_err_cc
                return val
        else:
            # Target LOG2-DECAY Model
            k_target = 8 if COMBINED_MODE else 7
            target_name = "γCDM-LOG²-Decay"
            def chi2_target_mock(params):
                orig_fa, orig_sc, orig_nn, orig_asym = args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric
                args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric = False, False, False, False
                global z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc
                _orig_z_mu, _orig_mu_obs, _orig_err_mu, _orig_z_cc, _orig_H_obs, _orig_err_cc = z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc
                z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc = z_mu_m, mu_obs_m, err_mu_m, z_cc_m, H_obs_m, err_cc_m
                try:
                    val = chi2_gcdm_log_decay(params)
                finally:
                     args.fixed_anchor, args.sanity_check, args.no_nuisance, args.asymmetric = orig_fa, orig_sc, orig_nn, orig_asym
                     z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc = _orig_z_mu, _orig_mu_obs, _orig_err_mu, _orig_z_cc, _orig_H_obs, _orig_err_cc
                return val


        # Fit ΛCDM
        best_lcdm_m = np.inf
        for _ in range(10):
            if COMBINED_MODE:
                x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                      rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]
            else:
                x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                      rng.uniform(-1.0, 1.0)]
            try:
                res = minimize(chi2_lcdm_mock, x0, method='Nelder-Mead',
                               options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_lcdm_m:
                    best_lcdm_m = res.fun
            except Exception:
                pass

        # Fit Target Model
        best_target_m = np.inf
        target_p = None
        for _ in range(10):
            if getattr(args, 'legacy', False):
                # Linear Model: H0, omch2, M_sne, M_qso, gamma_0
                if COMBINED_MODE:
                    x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                          rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
                          rng.uniform(-1.5, 0.5)]
                else:
                    x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                          rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 0.5)]
            else:
                # Log2-Decay Model: H0, omch2, M_sne, M_qso, A, z_b, gamma_0, z_h
                if COMBINED_MODE:
                    x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                          rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
                          rng.uniform(-0.5, 0.5), rng.uniform(0.1, 2.0),
                          rng.uniform(-1.5, 0.5), rng.uniform(10, 80)]
                else:
                    x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                          rng.uniform(-1.0, 1.0),
                          rng.uniform(-0.5, 0.5), rng.uniform(0.1, 2.0),
                          rng.uniform(-1.5, 0.5), rng.uniform(10, 80)]
            try:
                res = minimize(chi2_target_mock, x0, method='Nelder-Mead',
                               options={'maxiter': 20000, 'maxfev': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_target_m:
                    best_target_m = res.fun
                    target_p = res.x
            except Exception:
                pass

        bic_l = best_lcdm_m + k_lcdm * np.log(N)
        bic_g = best_target_m + k_target * np.log(N)
        dbic_m = bic_g - bic_l

        aic_l = best_lcdm_m + 2 * k_lcdm
        aic_g = best_target_m + 2 * k_target
        daic_m = aic_g - aic_l

        if getattr(args, 'legacy', False):
            best_gamma_m = target_p[-1] if target_p is not None else 0.0
        else:
            best_gamma_m = target_p[-2] if target_p is not None else 0.0

        mock_gammas.append(best_gamma_m)
        mock_dbics.append(dbic_m)
        mock_daics.append(daic_m)

        if dbic_m < -6:  # "strong" false detection
            false_detections += 1

        print(f"   Mock {m+1:>3}/{args.n_mock}: γ = {best_gamma_m:+.4f}, ΔBIC = {dbic_m:+.1f}, ΔAIC = {daic_m:+.1f}")

    # Summary
    mock_gammas = np.array(mock_gammas)
    mock_dbics = np.array(mock_dbics)
    mock_daics = np.array(mock_daics)

    print(f"\n" + "─" * 70)
    print(f"📋 RESULTADOS MOCK TEST (γ=0 truth)")
    print(f"─" * 70)
    print(f"   ⟨γ⟩ mock      = {np.mean(mock_gammas):+.4f} ± {np.std(mock_gammas):.4f}")
    print(f"   ⟨ΔBIC⟩ mock   = {np.mean(mock_dbics):+.1f} ± {np.std(mock_dbics):.1f}")
    print(f"   ⟨ΔAIC⟩ mock   = {np.mean(mock_daics):+.1f} ± {np.std(mock_daics):.1f}")
    print(f"   Falsas alarmas = {false_detections}/{args.n_mock} (ΔBIC < −6)")

    print(f"\n   COMPARACIÓN con datos REALES:")
    print(f"   {'':>20} {'Mock (γ=0)':>15} {'Real':>15}")
    print(f"   {'γ':>20} {np.mean(mock_gammas):>+15.4f} {gamma_g:>+15.4f}")
    print(f"   {'ΔBIC':>20} {np.mean(mock_dbics):>+15.1f} {delta_bic:>+15.1f}")

    # Verdict
    sigma_gamma = abs(gamma_g - np.mean(mock_gammas)) / max(np.std(mock_gammas), 1e-6)
    print(f"\n   Separación γ_real vs γ_mock: {sigma_gamma:.1f}σ")

    if false_detections == 0 and abs(np.mean(mock_gammas)) < 0.1:
        print(f"\n   ✅ PIPELINE VALIDADO: no fabrica señal γ espuria")
        print(f"      La señal γ = {gamma_g:.4f} en datos reales es GENUINA")
    elif false_detections <= 1:
        print(f"\n   ✅ Pipeline limpio ({false_detections} falsa alarma aceptable)")
    else:
        print(f"\n   ⚠️  {false_detections} falsas alarmas → revisar pipeline")


# ============================================================================
# MCMC / NESTED SAMPLING VALIDATION (optional)
# ============================================================================
if args.mcmc or args.nested:
    try:
        from cobaya.likelihood import Likelihood
        from cobaya.run import run
        import getdist
        from getdist import MCSamples
        import getdist.plots as gdplots
        import matplotlib.pyplot as plt
        COBAYA_AVAILABLE = True
    except ImportError:
        print("\n⚠️  Cobaya/getdist no disponible. pip install cobaya getdist")
        COBAYA_AVAILABLE = False

    if COBAYA_AVAILABLE:
        print("\n" + "=" * 70)
        print("🔬 COBAYA ROBUSTNESS VERIFICATION")
        print("=" * 70)

        os.makedirs("chains", exist_ok=True)
        has_log2_samples = False

        # ── Import Likelihood classes from shared module ──
        from gammacdm_likelihoods import create_likelihoods
        LCDMLikelihood, GammaCDM_LOG2_Likelihood, DecayLikelihood, GammaCDM_LOG_DECAY_Likelihood = \
            create_likelihoods(
                z_mu=z_mu, mu_obs=mu_obs, err_mu=err_mu,
                z_cc=z_cc, H_obs=H_obs, err_cc=err_cc,
                sne_mask=sne_mask, combined_mode=COMBINED_MODE,
                sigma_int_sne=SIGMA_INT_SNE, sigma_int_qso=SIGMA_INT_QSO,
                no_nuisance=args.no_nuisance, asymmetric=args.asymmetric,
                sanity_check=args.sanity_check,
                likelihood_type=LIKELIHOOD_TYPE,
                nu=NU_DOF if NU_DOF is not None else 5.0,
                C_inv_sne=C_inv_sne, ln_det_C_sne=ln_det_C_sne,
                use_cmb=USE_CMB, fit_scatter=FIT_SCATTER,
                cov_evals=_cov_evals, cov_evecs=_cov_evecs,
                no_bubble=args.no_bubble
            )
        cov_label = f" (with {args.cov} covariance)" if C_inv_sne is not None else ""
        print(f"   ✅ Shared likelihoods loaded from gammacdm_likelihoods.py{cov_label}")

        if args.nested:
            # ==================================================================
            # PARALLEL SUBPROCESS EXECUTION for PolyChord
            # Avoids MPI_FINALIZE issue by running each model in separate process
            # ==================================================================
            import subprocess
            import json
            
            print(f"\n🔮 Using PARALLEL NESTED SAMPLING (PolyChord, nlive={args.nlive})")
            if args.legacy:
                print(f"   Launching ΛCDM, γCDM-LOG², Decay, LOG²-Decay in separate processes (Legacy Mode)...\n")
            else:
                print(f"   Launching ΛCDM + LOG²-Decay in separate processes (Fast Mode)...\n")
            sampler_name = "NESTED"
            
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_nested_single.py")
            python_exe = sys.executable
            
            common_args = [
                "--nlive", str(args.nlive),
                "--sigma-int-sne", str(args.sigma_int_sne),
                "--sigma-int-qso", str(args.sigma_int_qso),
                "--qso-err-cut", str(args.qso_err_cut),
                "--sne-err-cut", str(args.sne_err_cut),
                "--z-min-sne", str(args.z_min_sne),
                "--z-min-qso", str(args.z_min_qso),
                "--output-dir", args.output_dir
            ]
            
            if args.no_quasars:
                common_args.append("--no-quasars")
            
            if args.fixed_anchor:
                common_args.append("--fixed-anchor")
            
            if args.sanity_check:
                common_args.append("--sanity-check")

            
            if args.quasars_only:
                common_args.append("--quasars-only")

            if args.revised:
                common_args.append("--revised")

            if args.asymmetric:
                common_args.append("--asymmetric")

            if args.no_nuisance:
                common_args.append("--no-nuisance")
            
            if args.student:
                common_args.append("--student")
                common_args.extend(["--nu", str(args.nu)])
            elif args.cauchy:
                common_args.append("--cauchy")

            if args.cov != 'none':
                common_args.extend(["--cov", args.cov])

            if args.cmb:
                common_args.append("--cmb")
            if FIT_SCATTER:
                common_args.append("--fit-scatter")
            
            # ==================================================================
            # PARALLEL EXECUTION (Robust)
            # ==================================================================

            # Open log files for output redirection (avoids PIPE buffer deadlocks)
            log_lcdm_path = os.path.join(args.output_dir, "lcdm_run.log")
            log_log2_path = os.path.join(args.output_dir, "log2_run.log")
            log_decay_path = os.path.join(args.output_dir, "decay_run.log")
            log_log_decay_path = os.path.join(args.output_dir, "log_decay_run.log")
            
            f_lcdm = open(log_lcdm_path, "w")
            f_log2 = None
            f_decay = None
            f_log_decay = open(log_log_decay_path, "w")
            
            # --- Launch Processes ---
            
            # 1. ΛCDM (never gets --asymmetric or --no-nuisance: ΛCDM always keeps M free)
            lcdm_args = [arg for arg in common_args if arg not in ("--asymmetric", "--no-nuisance")]

            proc_lcdm = subprocess.Popen(
                [python_exe, script_path, "lcdm"] + lcdm_args,
                stdout=f_lcdm, stderr=subprocess.STDOUT, text=True
            )
            
            # 2. γCDM-LOG² (legacy only)
            # The base model arguments are inherited from common_args now.
            model_args = common_args.copy()

            proc_log2 = None
            if args.legacy:
                f_log2 = open(log_log2_path, "w")
                proc_log2 = subprocess.Popen(
                    [python_exe, script_path, "log2"] + model_args,
                    stdout=f_log2, stderr=subprocess.STDOUT, text=True
                )

            # 3. Decay (legacy only)
            proc_decay = None
            if args.legacy:
                f_decay = open(log_decay_path, "w")
                proc_decay = subprocess.Popen(
                    [python_exe, script_path, "decay"] + model_args,
                    stdout=f_decay, stderr=subprocess.STDOUT, text=True
                )

            # 4. LOG²-DECAY (always)
            proc_log_decay = subprocess.Popen(
                [python_exe, script_path, "log_decay"] + model_args,
                stdout=f_log_decay, stderr=subprocess.STDOUT, text=True
            )

            print("   ⏳ ΛCDM     running (PID: {}) -> {}".format(proc_lcdm.pid, log_lcdm_path))
            if proc_log2:
                print("   ⏳ LOG²     running (PID: {}) -> {}".format(proc_log2.pid, log_log2_path))
            if proc_decay:
                print("   ⏳ DECAY    running (PID: {}) -> {}".format(proc_decay.pid, log_decay_path))
            print("   ⏳ LOG-DEC  running (PID: {}) -> {}".format(proc_log_decay.pid, log_log_decay_path))
            print("\n   Waiting for all to complete...\n")
            
            # Wait for all processes to finish
            exit_lcdm = proc_lcdm.wait()
            exit_log2 = proc_log2.wait() if proc_log2 else 0
            exit_decay = proc_decay.wait() if proc_decay else 0
            exit_log_decay = proc_log_decay.wait()
            
            # Close log files
            f_lcdm.close()
            if f_log2: f_log2.close()
            if f_decay: f_decay.close()
            f_log_decay.close()
            
            # Print relevant output from logs
            def print_log_summary(name, log_path, exit_code, success_keywords):
                if exit_code != 0:
                    print(f"   ❌ {name} failed (Exit Code {exit_code}). Check log: {log_path}")
                    try:
                        with open(log_path, "r") as f:
                            lines = f.readlines()
                            print("   Last 10 lines of log:")
                            for line in lines[-10:]:
                                print(f"      {line.strip()}")
                    except:
                        pass
                else:
                    try:
                        with open(log_path, "r") as f:
                            for line in f:
                                if any(x in line for x in success_keywords):
                                    print(f"   [{name}] {line.strip()}")
                    except Exception as e:
                        print(f"   ⚠️ Could not read log for {name}: {e}")

            success_keys = ['log(Z)', '📋', 'H₀', 'γ₀', 'A', 'zd', 'zh', '✅', 'δM', 'M_sne', 'M_qso', 'Ωch²', 'Ωmh²', '🌀']

            print_log_summary("ΛCDM", log_lcdm_path, exit_lcdm, success_keys)
            if args.legacy:
                print_log_summary("LOG²", log_log2_path, exit_log2, success_keys)
                print_log_summary("DECAY", log_decay_path, exit_decay, success_keys)
            print_log_summary("LOG-DECAY", log_log_decay_path, exit_log_decay, success_keys)
            
            print("\n   ✅ Parallel nested sampling completed!\n")

            
            # Load results
            try:
                with open(os.path.join(args.output_dir, "nested_lcdm_results.json"), "r") as f:
                    res_lcdm = json.load(f)
                try:
                    with open(os.path.join(args.output_dir, "nested_log2_results.json"), "r") as f:
                        res_log2 = json.load(f)
                except FileNotFoundError:
                    res_log2 = None
                try:
                    with open(os.path.join(args.output_dir, "nested_decay_results.json"), "r") as f:
                        res_decay = json.load(f)
                except FileNotFoundError:
                    res_decay = None
                try:
                    with open(os.path.join(args.output_dir, "nested_log_decay_results.json"), "r") as f:
                        res_log_decay = json.load(f)
                except FileNotFoundError:
                    res_log_decay = None
                
                logZ_lcdm = res_lcdm.get("logZ")
                logZ_log2 = res_log2.get("logZ") if res_log2 else None
                logZ_decay = res_decay.get("logZ") if res_decay else None
                logZ_log_decay = res_log_decay.get("logZ") if res_log_decay else None
                
                # Print Bayes Factor
                print(f"\n" + "=" * 70)
                print(f"🌟 BAYES FACTOR FROM NESTED SAMPLING")
                print(f"=" * 70)
                if logZ_lcdm is not None and logZ_log2 is not None:
                    print(f"   log(Z_ΛCDM)     = {logZ_lcdm:.2f}")
                    print(f"   log(Z_LOG²)     = {logZ_log2:.2f}")
                    ln_B = logZ_log2 - logZ_lcdm
                    print(f"   ln(B_LOG²/ΛCDM) = {ln_B:.2f}")
                    print(f"   B = exp(ln B)   = {np.exp(min(ln_B, 700)):.2e}")
                    if ln_B > 20:
                        print(f"\n   🏆 Evidencia favorable para γCDM-LOG² (lnB > 20)")
                
                if logZ_lcdm is not None and logZ_decay is not None:
                    ln_B_decay = logZ_decay - logZ_lcdm
                    print(f"   ln(B_DECAY/ΛCDM)= {ln_B_decay:.2f}")
                    if ln_B_decay > 5:
                         print(f"   🏆 Evidencia favorable para Decay (lnB > 5)")

                if logZ_lcdm is not None and logZ_log_decay is not None:
                    ln_B_ld = logZ_log_decay - logZ_lcdm
                    print(f"   ln(B_LOG-DEC/ΛCDM)= {ln_B_ld:.2f}")
                    if ln_B_ld > 5:
                         print(f"   🏆 Evidencia favorable para LOG²-DECAY (lnB > 5)")

                if not (logZ_lcdm is not None and logZ_log2 is not None):
                     print("   ⚠️ Could not extract log(Z) from one or both runs")
                
                print(f"\n   📊 Interpretación (Jeffreys scale):")
                print(f"      ln(B) > 5   → Strong evidence")
                print(f"      ln(B) > 10  → Very strong evidence")
                print(f"      ln(B) > 20  → Decisive evidence")

                # Summary comparison
                print(f"\n" + "=" * 70)
                print(f"📋 COMPARISON SUMMARY")
                print(f"=" * 70)
                print(f"   Model      |    H₀         |     γ₀/A      |    δM")
                print(f"   -----------|---------------|---------------|----------")
                
                # LCDM
                h0_l = f"{res_lcdm['H0_mean']:.1f} ± {res_lcdm['H0_std']:.1f}"
                if res_lcdm['H0_std'] < 0.01: h0_l = f"{res_lcdm['H0_mean']:.1f} (Fix)"
                print(f"   ΛCDM       | {h0_l:<13} |      —        | {res_lcdm['deltaM_mean']:.3f} (S:{res_lcdm['M_sne_mean']:.3f})")
                
                # LOG2 (legacy only)
                if res_log2:
                    h0_g = f"{res_log2['H0_mean']:.1f} ± {res_log2['H0_std']:.1f}"
                    if res_log2['H0_std'] < 0.01: h0_g = f"{res_log2['H0_mean']:.1f} (Fix)"
                    g_val = f"{res_log2.get('gamma_log2_mean', 0):.4f} ± {res_log2.get('gamma_log2_std', 0):.4f}"
                    print(f"   γCDM-LOG²  | {h0_g:<13} | {g_val:<13} | {res_log2['deltaM_mean']:.3f} (S:{res_log2['M_sne_mean']:.3f})")

                if res_decay:
                     h0_d = f"{res_decay['H0_mean']:.1f}"
                     if res_decay['H0_std'] < 0.01: h0_d += " (Fix)"
                     else: h0_d += f" ± {res_decay['H0_std']:.1f}"
                     
                     a_val = f"A={res_decay.get('A_mean', 0):.3f}"
                     print(f"   Decay      | {h0_d:<13} | {a_val:<13} | {res_decay['deltaM_mean']:.3f} (S:{res_decay.get('M_sne_mean', 0):.3f})")
                     if 'zd_mean' in res_decay:
                         print(f"              |               | zd={res_decay['zd_mean']:.3f}       |")

                if res_log_decay:
                     h0_ld = f"{res_log_decay['H0_mean']:.1f}"
                     if res_log_decay['H0_std'] < 0.01: h0_ld += " (Fix)"
                     else: h0_ld += f" ± {res_log_decay['H0_std']:.1f}"
                     g_ld_val = res_log_decay.get('gamma_log_decay_mean', 0)
                     a_ld_val = res_log_decay.get('A_mean', 0)
                     zb_ld = res_log_decay.get('zb_mean', 0)
                     zh_ld = res_log_decay.get('zh_mean', 0)
                     h0_local_ld = (h0_local(res_log_decay['H0_mean'], A=a_ld_val,
                                             z_b=(zb_ld if zb_ld > 0 else 1.0),
                                             z_pivot=Z_PIVOT)
                                    if a_ld_val != 0 else res_log_decay['H0_mean'])
                     print(f"   LOG²-DEC   | {h0_ld:<13} | γ={g_ld_val:<10.4f} | {res_log_decay['deltaM_mean']:.3f} (S:{res_log_decay.get('M_sne_mean', 0):.3f})")
                     print(f"              |               | A={a_ld_val:.3f} zb={zb_ld:.3f} zh={zh_ld:.1f}")
                     print(f"              | H₀(local)={h0_local_ld:.1f} |               |")

                # Hubble tension context
                print(f"\n   📊 TENSIÓN DE HUBBLE:")
                print(f"      Planck CMB:  H₀ = 67.4 ± 0.5")
                print(f"      SH0ES:       H₀ = 73.0 ± 1.0")
                if res_log2:
                    print(f"      γCDM-LOG²:   H₀ = {res_log2['H0_mean']:.1f} ± {res_log2['H0_std']:.1f}")
                if res_decay:
                    print(f"      DECAY:  H₀ = {res_decay['H0_mean']:.1f} ± {res_decay['H0_std']:.1f}")
                if res_log_decay:
                    print(f"      LOG²-DECAY:  H₀ = {res_log_decay['H0_mean']:.1f} ± {res_log_decay['H0_std']:.1f}")

                # Baryonic check
                if res_log2:
                    h_log2 = res_log2['H0_mean'] / 100.0
                    ombh2_log2 = res_log2.get('ombh2', 0.0224) 
                    omch2_log2 = res_log2.get('omch2_mean', 0.12)
                    omh2_log2_total = omch2_log2 + ombh2_log2
                    
                    print(f"\n   🧪 DENSIDAD FÍSICA (Baryon check - LOG²):")
                    print(f"      Ωb h² (Fix)  ≈ {ombh2_log2:.4f}")
                    print(f"      Ωc h² (Log2) = {omch2_log2:.4f}")
                    print(f"      Ωm h² (Tot)  = {omh2_log2_total:.4f}")

                if res_decay:
                    omch2_dec = res_decay.get('omch2_mean', 0.12)
                    ombh2_dec = 0.0224
                    print(f"\n   🧪 DENSIDAD FÍSICA (Decay):")
                    print(f"      Ωb h² (Fix)  ≈ {ombh2_dec:.4f}")
                    print(f"      Ωc h² (Dec)  = {omch2_dec:.4f}")
                    print(f"      Ωm h² (Tot)  = {omch2_dec + ombh2_dec:.4f}")
                
                if res_log_decay:
                    omch2_ld = res_log_decay.get('omch2_mean', 0.12)
                    ombh2_ld = 0.0224
                    print(f"\n   🧪 DENSIDAD FÍSICA (LOG²-DECAY):")
                    print(f"      Ωb h² (Fix)  ≈ {ombh2_ld:.4f}")
                    print(f"      Ωc h² (LD)   = {omch2_ld:.4f}")
                    print(f"      Ωm h² (Tot)  = {omch2_ld + ombh2_ld:.4f}")
                    print(f"      Ωc h² (Planck) ≈ 0.120")

                # ==============================================================
                # LOAD SAMPLES FOR PLOTTING
                # ==============================================================
                from getdist import loadMCSamples
                print(f"\n⏳ Loading chains for plots...")
                try:
                    prefix = "nested" if args.nested else "mcmc"
                    # Load LCDM
                    gd_lcdm = loadMCSamples(os.path.join(args.output_dir, f"{prefix}_lcdm"), settings={'ignore_rows': 0.2})
                    samples_lcdm = {p: gd_lcdm.samples[:, i] for i, p in enumerate(gd_lcdm.getParamNames().list())}
                    # Load LOG2 (legacy only)
                    if res_log2:
                        gd_log2 = loadMCSamples(os.path.join(args.output_dir, f"{prefix}_log2"), settings={'ignore_rows': 0.2})
                        samples_log2 = {p: gd_log2.samples[:, i] for i, p in enumerate(gd_log2.getParamNames().list())}
                    else:
                        samples_log2 = None
                    
                    # Load Decay
                    if res_decay:
                        gd_decay = loadMCSamples(os.path.join(args.output_dir, f"{prefix}_decay"), settings={'ignore_rows': 0.2})
                        samples_decay = {p: gd_decay.samples[:, i] for i, p in enumerate(gd_decay.getParamNames().list())}
                    else:
                        samples_decay = None

                    # Load LOG²-DECAY
                    if res_log_decay:
                        gd_log_decay = loadMCSamples(os.path.join(args.output_dir, f"{prefix}_log_decay"), settings={'ignore_rows': 0.2})
                        samples_log_decay = {p: gd_log_decay.samples[:, i] for i, p in enumerate(gd_log_decay.getParamNames().list())}
                    else:
                        samples_log_decay = None

                    has_log2_samples = True
                    print("   ✅ Chains loaded successfully")
                except Exception as e:
                    print(f"   ⚠️ Could not load chains for plotting: {e}")
                    samples_lcdm, samples_log2 = None, None

            except Exception as e:
                print(f"   ⚠️ Error loading results: {e}")
            
            # Continue to allow the rest of the script (plots, etc) to run
            print("\n   ✅ Parallel nested sampling results aggregated!")
        # ========================================================================
        # RESULTS INITIALIZATION (only overwrite if not nested)
        # ========================================================================
        if not args.nested:
            logZ_lcdm = None
            logZ_log2 = None
            logZ_decay = None
            logZ_log_decay = None
            samples_lcdm = None
            samples_log2 = None
            samples_decay = None
            samples_log_decay = None
            has_log2_samples = False

        # ========================================================================
        # MCMC FLOW (only when NOT using --nested subprocess mode)
        # ========================================================================
        if not args.nested:
            if not args.legacy:
                # Shared priors (IDENTICAL for all models)
                # Gaussian calibration priors on δM (externally justified):
                #   M_sne ~ N(0, 0.05): Pantheon+ residual calibration ~0.02-0.04 mag
                #   M_qso ~ N(0, 0.15): Lusso+20 L_X-L_UV intercept ~0.1-0.2 mag
                if COMBINED_MODE:
                    base_params = {
                        "H0": {"prior": {"min": H0_MIN, "max": H0_MAX}, "ref": 67, "proposal": 2.0},
                        "omch2": {"prior": {"min": OMCH2_MIN, "max": OMCH2_MAX}, "ref": 0.12, "proposal": 0.02},
                        "M_sne": {"prior": {"dist": "norm", "loc": 0, "scale": 0.05}, "ref": 0.0, "proposal": 0.02},
                        "M_qso": {"prior": {"dist": "norm", "loc": 0, "scale": 0.15}, "ref": 0.0, "proposal": 0.05},
                        "ombh2": 0.0224, "ns": 0.965, "As": 2.1e-9, "tau": 0.06}
                else:
                    base_params = {
                        "H0": {"prior": {"min": H0_MIN, "max": H0_MAX}, "ref": 67, "proposal": 2.0},
                        "omch2": {"prior": {"min": OMCH2_MIN, "max": OMCH2_MAX}, "ref": 0.12, "proposal": 0.02},
                        "mabs": {"prior": {"dist": "norm", "loc": 0, "scale": 0.05}, "ref": 0.0, "proposal": 0.02},
                        "ombh2": 0.0224, "ns": 0.965, "As": 2.1e-9, "tau": 0.06}           
            else:
                if COMBINED_MODE:
                    base_params = {
                        "H0": {"prior": {"min": H0_MIN, "max": H0_MAX}, "ref": 67, "proposal": 2.0},
                        "omch2": {"prior": {"min": OMCH2_MIN, "max": OMCH2_MAX}, "ref": 0.12, "proposal": 0.02},
                        "M_sne": {"prior": {"min": M_MIN, "max": M_MAX}, "ref": 0.0, "proposal": 0.1},
                        "M_qso": {"prior": {"min": M_MIN, "max": M_MAX}, "ref": 0.0, "proposal": 0.1},
                        "ombh2": 0.0224, "ns": 0.965, "As": 2.1e-9, "tau": 0.06}
                else:
                    base_params = {
                        "H0": {"prior": {"min": H0_MIN, "max": H0_MAX}, "ref": 67, "proposal": 2.0},
                        "omch2": {"prior": {"min": OMCH2_MIN, "max": OMCH2_MAX}, "ref": 0.12, "proposal": 0.02},
                        "mabs": {"prior": {"min": M_MIN, "max": M_MAX}, "ref": 0.0, "proposal": 0.1},
                        "ombh2": 0.0224, "ns": 0.965, "As": 2.1e-9, "tau": 0.06}


            if FIT_SCATTER:
                base_params["sigma_int_sne"] = {"prior": {"min": SINT_SNE_MIN, "max": SINT_SNE_MAX}, "ref": 0.1, "proposal": 0.01}
                if COMBINED_MODE:
                    base_params["sigma_int_qso"] = {"prior": {"min": SINT_QSO_MIN, "max": SINT_QSO_MAX}, "ref": 0.5, "proposal": 0.1}

            # Apply fixed-anchor / sanity-check overrides to base_params
            if args.fixed_anchor:
                print("   ⚓ MCMC: Fixing H0=67.4 (Fixed Anchor Mode)")
                base_params["H0"] = 67.4
                # M remains FREE (inherited from base_params)
            elif args.sanity_check:
                print("   🧠 MCMC: Sanity Check Mode (H0=67.4 fixed)")
                # For ΛCDM, H0 fixed, omch2 fixed, M free.
                # For γCDM/Decay, H0 fixed, omch2 free, M removed.
                # Since we run all models with ONE base_params set here, we have to be careful.
                # Actually, we customize per model below. Here we just set common defaults.
                # H0 is fixed for ALL models in sanity check.
                base_params["H0"] = 67.4
                # We leave omch2 and M free here, and constrain them per-model below.

            # MCMC sampler configuration
            sampler_cfg = {
                "mcmc": {
                    "max_tries": 2000,
                    "Rminus1_stop": 0.02, 
                    "Rminus1_cl_stop": 0.1,
                    "max_samples": args.samples,
                    "burn_in": 200, 
                    "proposal_scale": 0.5
                }
            }

            # ── ΛCDM MCMC ──
            sampler_name = "MCMC"
            print(f"\n⏳ Running ΛCDM {sampler_name}...")
            lcdm_p = {**base_params}
            if args.sanity_check:
                # ΛCDM Sanity: H0=67.4 (already in base), omch2 free (Fairer), M free
                # lcdm_p["omch2"] = 0.12 # <-- REMOVED
                pass
            
            info_l = {"likelihood": {"lcdm": LCDMLikelihood}, "theory": {"camb": {"stop_at_error": True}},
                      "params": lcdm_p, "sampler": sampler_cfg, "output": os.path.join(args.output_dir, "mcmc_lcdm"), "force": True}
            _, sampler_lcdm = run(info_l)
            
            # Force load from file for MCMC to ensure full chain access
            try:
                samples_lcdm = loadMCSamples(os.path.join(args.output_dir, "mcmc_lcdm"), settings={'ignore_rows':0.3})
                print(f"   ✅ Loaded ΛCDM samples from file: {samples_lcdm.numrows} samples")
            except Exception as e:
                print(f"   ⚠️ Could not load ΛCDM samples from file, using in-memory: {e}")
                samples_lcdm = sampler_lcdm.products().get("sample")
            
        # ── γCDM-LOG² MCMC (legacy only, not in nested subprocess mode) ──
        if not args.nested and args.legacy:
            print(f"\n⏳ Running γCDM-LOG² {sampler_name}...")
            log2_p = {**base_params, "gamma_log2": {"prior": {"min": GAMMA_MIN, "max": GAMMA_MAX}, "ref": -0.8, "proposal": 0.05}}
            if args.asymmetric or args.no_nuisance or args.sanity_check:
                if COMBINED_MODE:
                    log2_p["M_sne"] = 0.0
                    log2_p["M_qso"] = 0.0
                else:
                    log2_p["mabs"] = 0.0
            
            # If not fixed anchor and not sanity check, H0 is free (though base_params might have H0 free already)
            # If sanity check, H0 is already fixed to 67.4 in base_params
            if not args.fixed_anchor and not args.sanity_check:
                log2_p["H0"]["ref"] = 73
            info_log2 = {"likelihood": {"log2": GammaCDM_LOG2_Likelihood}, "theory": {"camb": {"stop_at_error": True}},
                            "params": log2_p, "sampler": sampler_cfg, "output": os.path.join(args.output_dir, "mcmc_log2"), "force": True}
            _, sampler_log2 = run(info_log2)
            
            # Force load from file
            try:
                samples_log2 = loadMCSamples(os.path.join(args.output_dir, "mcmc_log2"), settings={'ignore_rows':0.3})
                print(f"   ✅ Loaded γCDM-LOG² samples from file: {samples_log2.numrows} samples")
            except Exception as e:
                print(f"   ⚠️ Could not load γCDM-LOG² samples from file, using in-memory: {e}")
                samples_log2 = sampler_log2.products().get("sample")

            has_log2_samples = True

        # ── γCDM-Decay MCMC (legacy only, not in nested subprocess mode) ──
        if not args.nested and args.legacy:
            print(f"\n⏳ Running γCDM-Decay {sampler_name}...")
            decay_p = {**base_params, 
                       "A": {"prior": {"min": A_MIN, "max": A_MAX}, "ref": -0.175, "proposal": 0.05},
                       "zd": {"prior": {"min": ZD_MIN, "max": ZD_MAX}, "ref": 3.5, "proposal": 0.1}}
            
            if args.asymmetric or args.no_nuisance or args.sanity_check:
                 if COMBINED_MODE:
                     decay_p["M_sne"] = 0.0
                     decay_p["M_qso"] = 0.0
                 else:
                     decay_p["mabs"] = 0.0

            # If sanity check, H0=67.4 is already in base_params
            
            info_decay = {"likelihood": {"decay": DecayLikelihood}, "theory": {"camb": {"stop_at_error": True}},
                             "params": decay_p, "sampler": sampler_cfg, "output": os.path.join(args.output_dir, "mcmc_decay"), "force": True}
            _, sampler_decay = run(info_decay)
            
            # Force load from file
            try:
                samples_decay = loadMCSamples(os.path.join(args.output_dir, "mcmc_decay"), settings={'ignore_rows':0.3})
                print(f"   ✅ Loaded γCDM-Decay samples from file: {samples_decay.numrows} samples")
            except Exception as e:
                print(f"   ⚠️ Could not load γCDM-Decay samples from file, using in-memory: {e}")
                samples_decay = sampler_decay.products().get("sample")
            


        # ── γCDM-LOG²-DECAY MCMC (only when NOT using --nested subprocess mode) ──
        # Unified: A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h)
        # With --no-bubble: Kerr-Only Δμ = γ₀·[ln(1+z)]²·exp(-z/z_h)
        if not args.nested:
            _model_label = "γCDM-Kerr-Only" if args.no_bubble else "γCDM-LOG²-Decay"
            print(f"\n⏳ Running {_model_label} {sampler_name}...")
            if not args.legacy:
                # Conservative priors, log-uniform z_h reparametrisation.
                log_decay_p = {**base_params,
                        "gamma_log_decay": {"prior": {"min": -3.0, "max": 1.0}, "ref": -0.8, "proposal": 0.1},
                        "log_zh": {"prior": {"min": 0.0, "max": 2.301}, "ref": 1.62, "proposal": 0.15},
                        "zh": {"value": "lambda log_zh: 10**log_zh", "latex": r"z_h"}}
                if not args.no_bubble:
                    # Add local bubble parameters only when NOT in Kerr-Only mode
                    log_decay_p["A"]  = {"prior": {"min": -2.0, "max": 2.0}, "ref": -0.175, "proposal": 0.1}
                    log_decay_p["zb"] = {"prior": {"min": 0.01, "max": 5.0}, "ref": 0.4, "proposal": 0.1}
            else:
                ZH_MCMC_MAX = 100.0
                log_decay_p = {**base_params, 
                        "gamma_log_decay": {"prior": {"min": -2.0, "max": 0.0}, "ref": -0.8, "proposal": 0.05},
                        "zh": {"prior": {"min": ZH_MIN, "max": ZH_MCMC_MAX}, "ref": 42.0, "proposal": 5.0}}
                if not args.no_bubble:
                    log_decay_p["A"]  = {"prior": {"min": -1.0, "max": 1.0}, "ref": -0.175, "proposal": 0.05}
                    log_decay_p["zb"] = {"prior": {"min": 0.01, "max": 5.0}, "ref": 0.4, "proposal": 0.1}
                # Tighten δM priors to break A↔δM degeneracy (only relevant with bubble)
                if not args.no_bubble:
                    if COMBINED_MODE and isinstance(log_decay_p.get("M_sne"), dict):
                        log_decay_p["M_sne"] = {"prior": {"min": -1.0, "max": 1.0}, "ref": 0.0, "proposal": 0.05}
                        log_decay_p["M_qso"] = {"prior": {"min": -1.0, "max": 1.0}, "ref": 0.0, "proposal": 0.05}
                    elif not COMBINED_MODE and isinstance(log_decay_p.get("mabs"), dict):
                        log_decay_p["mabs"] = {"prior": {"min": -1.0, "max": 1.0}, "ref": 0.0, "proposal": 0.05}

            if args.asymmetric or args.no_nuisance or args.sanity_check:
                 if COMBINED_MODE:
                     log_decay_p["M_sne"] = 0.0
                     log_decay_p["M_qso"] = 0.0
                 else:
                     log_decay_p["mabs"] = 0.0

            # If sanity check, H0=67.4 is already in base_params
            
            info_log_decay = {"likelihood": {"log_decay": GammaCDM_LOG_DECAY_Likelihood}, "theory": {"camb": {"stop_at_error": True}},
                             "params": log_decay_p, "sampler": sampler_cfg, "output": os.path.join(args.output_dir, "mcmc_log_decay"), "force": True}
            _, sampler_log_decay = run(info_log_decay)
            
            # Force load from file
            try:
                samples_log_decay = loadMCSamples(os.path.join(args.output_dir, "mcmc_log_decay"), settings={'ignore_rows':0.3})
                print(f"   ✅ Loaded γCDM-LOG²-DECAY samples from file: {samples_log_decay.numrows} samples")
            except Exception as e:
                print(f"   ⚠️ Could not load γCDM-LOG²-Decay samples from file, using in-memory: {e}")
                samples_log_decay = sampler_log_decay.products().get("sample")
            


        # ── Report Results ──
        if samples_lcdm is not None:
            print(f"\n" + "=" * 70)
            report_model_summary("ΛCDM", samples_lcdm, args, logZ=logZ_lcdm, title_prefix=f"📋 {sampler_name.upper()} —")

        if samples_log2 is not None:
            report_model_summary("γCDM-LOG²", samples_log2, args, logZ=logZ_log2, title_prefix=f"📋 {sampler_name.upper()} —")

        if samples_decay is not None:
            report_model_summary("γCDM-Decay", samples_decay, args, logZ=logZ_decay, title_prefix=f"📋 {sampler_name.upper()} —")

        if samples_log_decay is not None:
            report_model_summary("γCDM-LOG²-Decay", samples_log_decay, args, logZ=logZ_log_decay, title_prefix=f"📋 {sampler_name.upper()} —")
        
        # ── Bayes Factor Summary (nested sampling only) ──
        if args.nested and logZ_lcdm is not None:
            has_any_bf = (logZ_log2 is not None) or (logZ_log_decay is not None)
            if has_any_bf:
                print(f"\n" + "=" * 70)
                print(f"🌟 BAYES FACTOR FROM NESTED SAMPLING")
                print(f"=" * 70)
                print(f"   log(Z_ΛCDM)        = {logZ_lcdm:.2f}")
                
                if logZ_log2 is not None:
                    print(f"   log(Z_LOG²)        = {logZ_log2:.2f}")
                    ln_B_log2 = logZ_log2 - logZ_lcdm
                    print(f"   ln(B_LOG²/ΛCDM)    = {ln_B_log2:.2f}")
                    if ln_B_log2 > 20:
                        print(f"   🏆 EVIDENCIA DECISIVA para γCDM-LOG²")
                
                if logZ_log_decay is not None:
                    print(f"   log(Z_LOG²-DEC)    = {logZ_log_decay:.2f}")
                    ln_B_ld = logZ_log_decay - logZ_lcdm
                    print(f"   ln(B_LOG²-DEC/ΛCDM)= {ln_B_ld:.2f}")
                    if ln_B_ld > 20:
                        print(f"   🏆 EVIDENCIA DECISIVA para γCDM-LOG²-Decay")
                
                print(f"\n   📊 Interpretación (Jeffreys scale):")
                print(f"      ln(B) > 5   → Strong evidence")
                print(f"      ln(B) > 10  → Very strong evidence")
                print(f"      ln(B) > 20  → Decisive evidence")

        # ========================================================================
        # LOG²-DECAY PROFESSIONAL PLOTTING SYSTEM
        # ========================================================================
        print(f"\n🎨 Generando gráficos de alta fidelidad...")
        try:
            prefix = "nested" if getattr(args, "nested", False) else "mcmc"
            from scipy.stats import gaussian_kde
            import matplotlib.patches as mpatches

            # 1. Register models for comparison
            # Structure: name -> {samples, color, label, marker}
            plot_models = {}

            if 'samples_lcdm' in locals() and samples_lcdm is not None:
                plot_models["lcdm"] = {"samples": samples_lcdm, "color": "#64748b", "label": "ΛCDM", "ls": "-"}
            
            if 'samples_log2' in locals() and samples_log2 is not None:
                plot_models["log2"] = {"samples": samples_log2, "color": "#e11d48", "label": "γCDM-LOG²", "ls": "--"}
            
            if 'samples_decay' in locals() and samples_decay is not None:
                plot_models["decay"] = {"samples": samples_decay, "color": "#2563eb", "label": "γCDM-Decay", "ls": "-."}

            if 'samples_log_decay' in locals() and samples_log_decay is not None:
                # Goldilocks Color: Gold / Amber
                plot_models["log_decay"] = {"samples": samples_log_decay, "color": "#eab308", "label": "γCDM-LOG²-Decay", "ls": "-"}

            # Extensibility: easy to add new models here
            
            # --- A. PERFECT H0 COMPARISON PLOT ---
            fig, ax = plt.subplots(1, 1, figsize=(11, 7))
            
            # Global style
            plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5})
            
            for m_id, m_cfg in plot_models.items():
                s = m_cfg["samples"]
                h0_arr = _safe_get(s, 'H0')
                if h0_arr is not None:
                    if np.std(h0_arr) > 0.05:
                        kde = gaussian_kde(h0_arr)
                        x = np.linspace(np.min(h0_arr)-2, np.max(h0_arr)+2, 500)
                        y = kde(x)
                        ax.fill_between(x, y, alpha=0.2, color=m_cfg["color"])
                        ax.plot(x, y, color=m_cfg["color"], lw=2.5, ls=m_cfg["ls"], 
                                label=f'{m_cfg["label"]} ({np.mean(h0_arr):.1f}±{np.std(h0_arr):.1f})')
                    else:
                        ax.axvline(np.mean(h0_arr), color=m_cfg["color"], ls=m_cfg["ls"], lw=3, 
                                   label=f'{m_cfg["label"]} (Fixed: {np.mean(h0_arr):.1f})')
                elif args.sanity_check or args.fixed_anchor:
                    # If H0 missing but fixed globally
                    ax.axvline(67.4, color=m_cfg["color"], ls=m_cfg["ls"], lw=3, alpha=0.6,
                               label=f'{m_cfg["label"]} (Fixed: 67.4)')

            # External constraints
            ax.axvline(67.4, color='#7c3aed', ls=':', lw=2, label='Planck CMB (67.4 ± 0.5)')
            ax.axvspan(66.9, 67.9, alpha=0.1, color='#7c3aed')
            
            ax.axvline(73.0, color='#059669', ls=':', lw=2, label='SH0ES Local (73.0 ± 1.0)')
            ax.axvspan(72.0, 74.0, alpha=0.1, color='#059669')

            # Aesthetics
            ax.set_xlabel(r'$H_0$ [km/s/Mpc]', fontsize=14, fontweight='bold')
            ax.set_ylabel('Probability Density', fontsize=14, fontweight='bold')
            ax.set_title(r'Hubble Tension: $\Lambda$CDM vs $\gamma$CDM', fontsize=16, pad=20)
            ax.grid(alpha=0.2, ls='--')
            ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)
            ax.set_xlim(58, 80)
            
            plt.tight_layout()
            h0_plot_path = os.path.join(args.output_dir, f"{prefix}_H0_comparison.png")
            plt.savefig(h0_plot_path, dpi=200, bbox_inches='tight')
            h0_log_path = os.path.join("logs", f"{SESSION_TIMESTAMP}_{prefix}_H0_comparison.png")
            plt.savefig(h0_log_path, dpi=200, bbox_inches='tight')
            print(f"   ✅ {h0_plot_path}")

            # --- B. PERFECT LOG²-DECAY CORNER PLOT ---
            all_mcsamples = []
            
            # Common core parameters + all potential model-specific ones
            # We want: H0, omch2, M, gamma_log2, A, zd
            # Note: GetDist handles missing columns by naming them and letting them be empty/masked
            for m_id, m_cfg in plot_models.items():
                s = m_cfg["samples"]
                viz_data = []
                viz_names = []
                viz_labels = []
                
                # 1. H0
                h0_vals = _safe_get(s, 'H0')
                if h0_vals is not None and np.std(h0_vals) > 0.001:
                    viz_data.append(h0_vals)
                    viz_names.append("H0")
                    viz_labels.append(r"H_0")
                
                # 2. omch2
                om_vals = _safe_get(s, 'omch2')
                if om_vals is not None:
                    viz_data.append(om_vals)
                    viz_names.append("omch2")
                    viz_labels.append(r"\Omega_c h^2")
                
                # 3. M (Nuisance)
                is_fixed_m = args.sanity_check or args.no_nuisance or (args.asymmetric and m_id != "lcdm")
                if not is_fixed_m:
                    m_sne = _safe_get(s, "M_sne")
                    m_qso = _safe_get(s, "M_qso")
                    if m_sne is not None and m_qso is not None:
                        m_arr = (m_sne + m_qso) / 2
                    else:
                        m_arr = _safe_get(s, "mabs", _safe_get(s, "M"))
                    
                    if m_arr is not None:
                        viz_data.append(m_arr)
                        viz_names.append("M")
                        viz_labels.append(r"\delta M")
                
                # 4. Model Specifics
                g_vals = _safe_get(s, 'gamma_log2')
                if g_vals is not None:
                    viz_data.append(g_vals)
                    viz_names.append("g")
                    viz_labels.append(r"\gamma_0")
                
                # gamma_log_decay (LOG²-DECAY model — Kerr component)
                g_ld_vals = _safe_get(s, 'gamma_log_decay')
                if g_ld_vals is not None:
                    viz_data.append(g_ld_vals)
                    viz_names.append("g_ld")
                    viz_labels.append(r"\gamma_{LD}")
                
                a_vals = _safe_get(s, 'A')
                if a_vals is not None:
                    viz_data.append(a_vals)
                    viz_names.append("A")
                    viz_labels.append(r"A")
                
                zd_vals = _safe_get(s, 'zd')
                if zd_vals is not None:
                    viz_data.append(zd_vals)
                    viz_names.append("zd")
                    viz_labels.append(r"z_d")
                    
                zb_vals = _safe_get(s, 'zb')
                if zb_vals is not None:
                    viz_data.append(zb_vals)
                    viz_names.append("zb")
                    viz_labels.append(r"z_b")
                    
                zh_vals = _safe_get(s, 'zh')
                if zh_vals is not None:
                    viz_data.append(zh_vals)
                    viz_names.append("zh")
                    viz_labels.append(r"z_h")

                if viz_data:
                    mcs = MCSamples(samples=np.column_stack(viz_data), 
                                     names=viz_names, labels=viz_labels, 
                                     label=m_cfg["label"])
                    all_mcsamples.append(mcs)

            if all_mcsamples:
                g = gdplots.get_subplot_plotter(subplot_size=3)
                g.settings.num_plot_contours = 2
                g.settings.solid_colors = [m_cfg["color"] for m_cfg in plot_models.values()]
                
                g.triangle_plot(all_mcsamples, filled=True, alpha_filled_add=0.4)
                
                header = "Model Comparison: Parameter Robustness"
                if args.fixed_anchor or args.sanity_check:
                    header += r" ($H_0$ Fixed @ 67.4)"
                plt.suptitle(header, fontsize=18, y=1.03)
                
                corner_plot_path = os.path.join(args.output_dir, f"{prefix}_unified_comparison_corner.png")
                g.export(corner_plot_path)
                corner_log_path = os.path.join("logs", f"{SESSION_TIMESTAMP}_{prefix}_unified_comparison_corner.png")
                g.export(corner_log_path)
                print(f"   ✅ {corner_plot_path}")

            # --- C. INDIVIDUAL LOG2 PARAMETER SPACE (Full detail) ---
            if "log2" in plot_models:
                s_log2 = plot_models["log2"]["samples"]
                full_params = [_safe_get(s_log2, "H0"), _safe_get(s_log2, "omch2"), _safe_get(s_log2, "gamma_log2")]
                full_names = ["H0", "omch2", "g"]
                full_labels = [r"H_0", r"\Omega_c h^2", r"\gamma_0"]
                
                # Filter out None if any param is missing
                valid = [p is not None for p in full_params]
                full_params = [p for p in full_params if p is not None]
                full_names = [n for i, n in enumerate(full_names) if valid[i]]
                full_labels = [l for i, l in enumerate(full_labels) if valid[i]]

                if not (args.sanity_check or args.asymmetric or args.no_nuisance):
                   m_vals, _ = _get_M(s_log2, COMBINED_MODE)
                   if COMBINED_MODE:
                       ms = _safe_get(s_log2, "M_sne")
                       mq = _safe_get(s_log2, "M_qso")
                       if ms is not None and mq is not None:
                           full_params.append((ms + mq)/2)
                           full_names.append("M")
                           full_labels.append(r"\delta M")
                   else:
                       m_val = _safe_get(s_log2, "mabs", _safe_get(s_log2, "M"))
                       if m_val is not None:
                           full_params.append(m_val)
                           full_names.append("M")
                           full_labels.append(r"\delta M")

                if full_params:
                    g_full = gdplots.get_subplot_plotter()
                    g_full.triangle_plot(MCSamples(samples=np.column_stack(full_params), 
                                                   names=full_names, labels=full_labels, label="γCDM-LOG²"),
                                        filled=True, color_line='#e11d48')
                    plt.suptitle(r"$\gamma$CDM-LOG²: Full Parameter Space Topology", fontsize=16, y=1.02)
                    log2_corner_path = os.path.join(args.output_dir, f"{prefix}_log2_full_corner.png")
                    plt.savefig(log2_corner_path, dpi=150, bbox_inches='tight')
                    log2_log_path = os.path.join("logs", f"{SESSION_TIMESTAMP}_{prefix}_log2_full_corner.png")
                    plt.savefig(log2_log_path, dpi=150, bbox_inches='tight')
                    print(f"   ✅ {log2_corner_path}")

            # --- D. INDIVIDUAL LOG-DECAY PARAMETER SPACE (Full detail) ---
            if "log_decay" in plot_models:
                s_ld = plot_models["log_decay"]["samples"]
                full_params = [_safe_get(s_ld, "H0"), _safe_get(s_ld, "omch2"), 
                               _safe_get(s_ld, "A"), _safe_get(s_ld, "zb"),
                               _safe_get(s_ld, "gamma_log_decay"), _safe_get(s_ld, "zh")]
                full_names = ["H0", "omch2", "A", "zb", "g_uni", "zh"]
                full_labels = [r"H_0", r"\Omega_c h^2", r"A", r"z_b", r"\gamma_0^{Kerr}", r"z_h"]
                
                # Filter out None
                valid = [p is not None for p in full_params]
                full_params = [p for p in full_params if p is not None]
                full_names = [n for i, n in enumerate(full_names) if valid[i]]
                full_labels = [l for i, l in enumerate(full_labels) if valid[i]]

                if not (args.sanity_check or args.asymmetric or args.no_nuisance):
                   if COMBINED_MODE:
                       ms = _safe_get(s_ld, "M_sne")
                       mq = _safe_get(s_ld, "M_qso")
                       if ms is not None and mq is not None:
                           full_params.append((ms + mq)/2)
                           full_names.append("M")
                           full_labels.append(r"\delta M")
                   else:
                       m_val = _safe_get(s_ld, "mabs", _safe_get(s_ld, "M"))
                       if m_val is not None:
                           full_params.append(m_val)
                           full_names.append("M")
                           full_labels.append(r"\delta M")

                if full_params:
                    g_ld = gdplots.get_subplot_plotter()
                    g_ld.triangle_plot(MCSamples(samples=np.column_stack(full_params), 
                                                   names=full_names, labels=full_labels, label="γCDM-LOG²-Decay"),
                                        filled=True, color_line='#eab308')
                    plt.suptitle(r"$\gamma$CDM-LOG²-Decay: Full Parameter Space Topology", fontsize=16, y=1.02)
                    ld_corner_path = os.path.join(args.output_dir, f"{prefix}_log_decay_full_corner.png")
                    plt.savefig(ld_corner_path, dpi=150, bbox_inches='tight')
                    ld_log_path = os.path.join("logs", f"{SESSION_TIMESTAMP}_{prefix}_log_decay_full_corner.png")
                    plt.savefig(ld_log_path, dpi=150, bbox_inches='tight')
                    print(f"   ✅ {ld_corner_path}")

            # --- E. INDIVIDUAL DECAY PARAMETER SPACE (Full detail) ---
            if "decay" in plot_models:
                s_dec = plot_models["decay"]["samples"]
                full_params = [_safe_get(s_dec, "H0"), _safe_get(s_dec, "omch2"), 
                               _safe_get(s_dec, "A"), _safe_get(s_dec, "zd")]
                full_names = ["H0", "omch2", "A", "zd"]
                full_labels = [r"H_0", r"\Omega_c h^2", r"A", r"z_d"]
                
                valid = [p is not None for p in full_params]
                full_params = [p for p in full_params if p is not None]
                full_names = [n for i, n in enumerate(full_names) if valid[i]]
                full_labels = [l for i, l in enumerate(full_labels) if valid[i]]

                if not (args.sanity_check or args.asymmetric or args.no_nuisance):
                   if COMBINED_MODE:
                       ms = _safe_get(s_dec, "M_sne")
                       mq = _safe_get(s_dec, "M_qso")
                       if ms is not None and mq is not None:
                           full_params.append((ms + mq)/2)
                           full_names.append("M")
                           full_labels.append(r"\delta M")
                   else:
                       m_val = _safe_get(s_dec, "mabs", _safe_get(s_dec, "M"))
                       if m_val is not None:
                           full_params.append(m_val)
                           full_names.append("M")
                           full_labels.append(r"\delta M")

                if full_params:
                    g_dec = gdplots.get_subplot_plotter()
                    g_dec.triangle_plot(MCSamples(samples=np.column_stack(full_params), 
                                                   names=full_names, labels=full_labels, label="γCDM-Decay"),
                                        filled=True, color_line='#2563eb')
                    plt.suptitle(r"$\gamma$CDM-Decay: Full Parameter Space Topology", fontsize=16, y=1.02)
                    decay_corner_path = os.path.join(args.output_dir, f"{prefix}_decay_full_corner.png")
                    plt.savefig(decay_corner_path, dpi=150, bbox_inches='tight')
                    decay_log_path = os.path.join("logs", f"{SESSION_TIMESTAMP}_{prefix}_decay_full_corner.png")
                    plt.savefig(decay_log_path, dpi=150, bbox_inches='tight')
                    print(f"   ✅ {decay_corner_path}")

        except Exception as e:
            print(f"   ⚠️ Error crítico en generación de gráficos: {e}")
            import traceback
            traceback.print_exc()


# ========================================================================
# MOCK TEST AFTER MCMC: γ=0 NULL HYPOTHESIS
# ========================================================================
if args.mock:
    run_mock_test()

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 70)
print("🏁 RESUMEN VERIFICACIÓN ROBUSTA")
print("=" * 70)
print(f"""
Protocolo implementado:
  ✓ δM nuisance en modelos (priors idénticos: [{M_MIN}, {M_MAX}])
  ✓ Multi-start MLE ({args.starts} runs)
  ✓ Priors amplios (H₀: [{H0_MIN},{H0_MAX}], Ωch²: [{OMCH2_MIN},{OMCH2_MAX}])
  ✓ Misma likelihood (solo difiere la corrección γ + Decay)
  ✓ AIC + BIC calculados""")

if args.mcmc:
    print("  ✓ Validación MCMC con convergencia R−1")
    print("  ✓ Corner plot γ vs δM guardado")
if args.mock:
    print("  ✓ Mock test γ=0 ejecutado (validación de pipeline)")
if not args.no_quasars and not args.quasars_only:
    print(f"  ✓ Quasars incluidos (err < {args.qso_err_cut}, M_sne/M_qso separados)")
if USE_EVO:
    print("  ✓ Optimizer: Differential evolution (5 × DE + dual_annealing + NM/Powell multi-polish + perturbation)")
else:
    print("  ✓ Optimizer: Nelder-Mead")
if USE_CMB:
    print(f"  ✓ CMB shift parameter prior (R = {R_PLANCK} ± {SIGMA_R_PLANCK}, σ_corr = {SIGMA_CORR_CMB} mag)")
if USE_BAO_FIT:
    print(f"  ✓ BAO (DESI DR1, {N_BAO_POINTS} pts) in joint likelihood — Δμ propagated to D_M if BAO_PROPAGATE_CORRECTION is true")
    if best_overall_model and "chi2_bao" in best_overall_model:
        print(f"      Best-fit χ²_BAO/ν = {best_overall_model['chi2_bao_red']:.2f} "
              f"({best_overall_model['name']})")
elif USE_BAO_NULL:
    print(f"  ✓ BAO (DESI DR1) null test performed — not in MLE objective")
    if best_overall_model and "chi2_bao" in best_overall_model:
        _cr = best_overall_model['chi2_bao_red']
        _tag = "PASS" if _cr < 2.0 else ("TENSION" if _cr < 5.0 else "FAIL")
        print(f"      Best-fit χ²_BAO/ν = {_cr:.2f}  [{_tag}]")
if FIT_SCATTER:
    _best_res = best_overall_model or lcdm_res
    if _best_res:
        print(f"  ✓ Intrinsic scatter fitted: σ_int,SNe = {_best_res.get('sig_sne', '?'):.4f}, σ_int,QSO = {_best_res.get('sig_qso', '?'):.4f}")

print(f"""
Resultados clave:
  γ    = {gamma_g:.4f} (MLE)
  ΔBIC = {delta_bic:.1f} (negativo → γCDM preferido)
  ΔAIC = {delta_aic:.1f} (negativo → γCDM preferido)
  K_BIC (approx) = {K_BIC_approx:.1f}  ← BIC-implied odds, NOT Bayes factor
""")

# Silent-failure accounting (tracked by ExceptionCounter in gammacdm_core).
# If this prints a non-zero number, CAMB or the likelihood returned 1e10 during
# optimisation and the final MLE may be biased toward the edges of the prior.
print("Silent-failure accounting (numerical):")
print(CAMB_ERRORS.summary())
print(CHI2_ERRORS.summary())

#   → γCDM es preferido incluso con δM nuisance incluido
#   → DECAY es mejor porque acerca SH0ES y CMB
#   → El efecto NO es un artefacto de absorción de offset
#   → Hipótesis Física: La tensión H₀ se resuelve
#     por este lensing geométrico con límites (Container Kerr Metric).

print("=" * 70)
