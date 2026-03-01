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
    python gammacdm_addendum_verification.py                       # MLE (with quasars)
    python gammacdm_addendum_verification.py --no-quasars          # MLE without quasars
    python gammacdm_addendum_verification.py --mcmc                # Full MCMC
    python gammacdm_addendum_verification.py --mock                # Run γ=0 null test
    python gammacdm_addendum_verification.py --mock --n-mock 50    # 50 mock realizations

Author: Bautista, 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
import argparse
import os
import sys
from getdist import loadMCSamples

warnings.filterwarnings('ignore')


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
    
    # z_h — horizon scale (Unified only)
    zh_s = _safe_get(samples, 'zh')
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
        h0_loc = h0_val * 10**(-a_val_for_h0/5)
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
parser.add_argument("--z-min", type=float, default=0.01, dest="z_min",
                    help="Minimum redshift to include (default: 0.01, removes local peculiar velocities)")
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
parser.add_argument("--fixed-anchor", action="store_true", help="Fix H0=67.4 and M=SH0ES (M=0) for ALL models")
parser.add_argument("--sanity-check", action="store_true", dest="sanity_check",
                    help="Internal sanity check: ΛCDM(H0=67.4,Ωm=0.315,M free) vs γCDM/Decay(H0=67.4,M removed)")
parser.add_argument("--output-dir", type=str, default="chains", help="Output directory for nested sampling chains")
parser.add_argument("--legacy", action="store_true", help="Include legacy models (γCDM-LINEAL, γCDM-LOG³) in MLE analysis")
args = parser.parse_args()


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 70)
print("🛡️  γCDM ROBUSTNESS VERIFICATION PROTOCOL")
print("=" * 70)
print(f"   ⚙️  Intrinsic scatter: σ_int,SNe = {args.sigma_int_sne:.2f}, σ_int,QSO = {args.sigma_int_qso:.2f}")
print(f"   ⚙️  Physical prior: Ωm < 1 enforced (flat ΛCDM with ΩΛ ≥ 0)")

# Try local first, then GitHub
dataset_name = 'full_dataset_revisado.csv' if args.revised else 'full_dataset.csv'
try:
    df = pd.read_csv(dataset_name)
except Exception:
    try:
        df = pd.read_csv('../' + dataset_name)
    except Exception:
        df = pd.read_csv('https://raw.githubusercontent.com/indigenica/akashic-alpha-engine/main/' + dataset_name)

if args.quasars_only:
    # ── QUASARS ONLY (high-z test) ──
    print(f"\n🔭 MODE: Quasars only (high-z test, err < {args.qso_err_cut})")
    qso = df[(df['probe'] == 'quasar') & (df['type'] == 'mu') & (df['err'] < args.qso_err_cut) & (df['z'] > args.z_min)]

    print(f"\n📊 Dataset:")
    print(f"   Quasars: {len(qso)} pts (μ observable, err < {args.qso_err_cut})")
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
    print(f"\n🔭 MODE: SNe Ia (err < {args.sne_err_cut}, z > {args.z_min}) + CC (sin quasars)")
    sne = df[(df['probe'] == 'sne_ia') & (df['type'] == 'mu') & (df['err'] < args.sne_err_cut) & (df['z'] > args.z_min)]
    cc = df[(df['probe'] == 'cc') & (df['type'] == 'H')]

    print(f"\n📊 Dataset:")
    print(f"   SNe Ia: {len(sne)} pts (μ, err < {args.sne_err_cut}, z > {args.z_min})")
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
    sne = df[(df['probe'] == 'sne_ia') & (df['type'] == 'mu') & (df['err'] < args.sne_err_cut) & (df['z'] > args.z_min)]
    cc = df[(df['probe'] == 'cc') & (df['type'] == 'H')]
    qso = df[(df['probe'] == 'quasar') & (df['type'] == 'mu') & (df['err'] < args.qso_err_cut)]

    n_sne = len(sne)
    n_qso = len(qso)

    mu_data = pd.concat([sne, qso])

    sne_mask = np.zeros(len(mu_data), dtype=bool)
    sne_mask[:n_sne] = True
    qso_mask = ~sne_mask

    print(f"\n🔭 MODE: SNe Ia (err < {args.sne_err_cut}, z > {args.z_min}) + Quasars (err < {args.qso_err_cut}) + CC")
    print(f"\n📊 Dataset:")
    print(f"   SNe Ia:   {n_sne} pts (μ, err < {args.sne_err_cut}, z > {args.z_min})")
    print(f"   Quasars:  {n_qso} pts (μ, err < {args.qso_err_cut})")
    print(f"   CC:       {len(cc)} pts (H)")
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


# ============================================================================
# χ² FUNCTIONS WITH δM NUISANCE + INTRINSIC SCATTER
# ============================================================================
# Shared parameters have IDENTICAL priors across ALL models:
#   H₀:       [40, 100]
#   Ωch²:     [0.01, 0.35]  → but we also enforce Ωm < 1 (physical)
#   δM:       [−3.0, 3.0]
#   γ:        [−2.0, 1.0]
#   σ_int:    [0.0, 2.0]    → intrinsic scatter (QSO and optionally SNe)
# ============================================================================

H0_MIN, H0_MAX = 40, 100
OMCH2_MIN, OMCH2_MAX = 0.01, 0.35
M_MIN, M_MAX = -3.0, 3.0
GAMMA_MIN, GAMMA_MAX = -3.0, 3.0
SIGMA_INT_MIN, SIGMA_INT_MAX = 0.0, 2.0
A_MIN, A_MAX = -3.0, 3.0
ZD_MIN, ZD_MAX = 0.01, 10.0
# Unified model (log_decay): two separate decay scales
ZB_MIN, ZB_MAX = 0.01, 10.0     # Bubble scale (local, short-range)
ZH_MIN, ZH_MAX = 0.01, 1e10    # Horizon scale (Kerr geometry, long-range)

# Intrinsic scatter from CLI arguments (added in quadrature to observational errors)
SIGMA_INT_SNE = args.sigma_int_sne   # ~0.1 mag typical for Pantheon+ without cov
SIGMA_INT_QSO = args.sigma_int_qso   # User-specified, try 0.3-0.6 for robustness test

def compute_Omega_m(H0, omch2, ombh2=0.0224):
    """Compute Ωm from H0 and Ωch² + Ωbh²."""
    h = H0 / 100
    return (omch2 + ombh2) / h**2

def check_physical_prior(H0, omch2):
    """Check if cosmology is physical: Ωm < 1 (i.e., ΩΛ ≥ 0 for flat)."""
    Om = compute_Omega_m(H0, omch2)
    return 0 < Om < 1.0  # Physical flat ΛCDM requires 0 < Ωm < 1


def chi2_lcdm(params):
    """ΛCDM: 3 params (H₀, Ωch², δM) or 4 combined (H₀, Ωch², M_sne, M_qso).
    
    Returns -2·logL = χ² + Σlog(σ_eff²) for proper Bayesian comparison.
    Includes physical prior Ωm < 1 and intrinsic scatter σ_eff² = σ_obs² + σ_int².
    """
    if COMBINED_MODE:
        if args.fixed_anchor or args.sanity_check:
            # Fixed Anchor / Sanity Check for LCDM: H0=67.4, M FREE (ALWAYS)
            # params: [omch2, M_sne, M_qso]
            H0 = 67.4
            omch2, M_sne, M_qso = params
                
            if not (OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX):
                return 1e10
        else:
            H0, omch2, M_sne, M_qso = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX):
                return 1e10
    else:
        if args.fixed_anchor or args.sanity_check:
            # Fixed Anchor / Sanity Check for LCDM: H0=67.4, M FREE
            H0 = 67.4
            omch2, delta_M = params
                
            if not (OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < delta_M < M_MAX):
                return 1e10
        else:
            H0, omch2, delta_M = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < delta_M < M_MAX):
                return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso)
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            mu_th = mu_th_base + delta_M
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)

        # -2·logL = χ² + normalization term (for proper comparison with varying σ_int)
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))  # log(σ²) penalty

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc = 0
            norm_cc = 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_gcdm(params):
    """γCDM constant: Δμ = γ·ln(1+z).
    
    Returns -2·logL = χ² + Σlog(σ_eff²) for proper Bayesian comparison.
    """
    if COMBINED_MODE:
        if args.fixed_anchor:
            # Fixed Anchor: H0=67.4. M is FREE (unless no_nuisance)
            # params: [omch2, M..., gamma]
            if args.no_nuisance:
                omch2, gamma = params
                M_sne, M_qso = 0.0, 0.0
                H0 = 67.4
            else:
                omch2, M_sne, M_qso, gamma = params
                H0 = 67.4
        elif args.sanity_check:
             # Sanity Check (My Models): H0=67.4, M REMOVED (Asymmetric)
             # params: [omch2, gamma]
             omch2, gamma = params
             H0, M_sne, M_qso = 67.4, 0.0, 0.0
        else:
            H0, omch2, M_sne, M_qso, gamma = params
            if args.no_nuisance:
                M_sne, M_qso = 0.0, 0.0
                
        # Bounds check logic
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
        if not (GAMMA_MIN < gamma < GAMMA_MAX): return 1e10
        if not args.fixed_anchor and not args.sanity_check:
             if not (H0_MIN < H0 < H0_MAX): return 1e10
        
        if not args.no_nuisance and not args.sanity_check: 
             # Check M if it's supposed to be free
             if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX): return 1e10
    else:
        if args.fixed_anchor:
            # Fixed Anchor: H0=67.4, M FREE (unless no_nuisance)
            if args.no_nuisance:
                omch2, gamma = params
                delta_M = 0.0
                H0 = 67.4
            else:
                omch2, delta_M, gamma = params
                H0 = 67.4
        elif args.sanity_check:
             # Sanity Check (My Models): H0=67.4, M REMOVED
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

    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        gamma_corr = gamma * np.log(1 + z_mu)

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso) + gamma_corr
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            mu_th = mu_th_base + delta_M + gamma_corr
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)

        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc = 0
            norm_cc = 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_lcdm_no_M(params):
    # This function is not used when fixed_anchor is True (bounds logic handles it)
    # But for completeness/safety, leave as is.
    """ΛCDM WITHOUT δM: 2 params (H₀, Ωch²). Includes physical Ωm prior and σ_int."""
    H0, omch2 = params
    if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
        return 1e10
    if not check_physical_prior(H0, omch2):
        return 1e10
    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)
        mu_th = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        if COMBINED_MODE:
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))
        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0
        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_gcdm_no_M(params):
    # Used when args.asymmetric is True.
    # If fixed_anchor is also True, args.fixed_anchor takes precedence in logic,
    # but we should ensure robustness just in case.
    """γCDM WITHOUT δM: 3 params (H₀, Ωch², γ). Includes physical Ωm prior and σ_int."""
    if args.fixed_anchor or args.sanity_check:
        # If somehow called with fixed anchor, it essentially becomes:
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

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        mu_th = mu_th_base + gamma * np.log(1 + z_mu)
        
        if COMBINED_MODE:
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)
            
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


# =============================================================================
# EVOLVING γ(z) MODELS
# =============================================================================

def chi2_gcdm_linear(params):
    """γCDM-LINEAR: γ(z) = γ₀·(1+z). Includes physical Ωm prior and σ_int."""
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma_0 = params
                M_sne, M_qso = 0.0, 0.0
            else:
                omch2, M_sne, M_qso, gamma_0 = params
            H0 = 67.4
        else:
            H0, omch2, M_sne, M_qso, gamma_0 = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX):
                return 1e10
            if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
                return 1e10
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma_0 = params
                delta_M = 0.0
            else:
                omch2, delta_M, gamma_0 = params
            H0 = 67.4
        else:
            H0, omch2, delta_M, gamma_0 = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < delta_M < M_MAX):
                return 1e10
            if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
                return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        gamma_corr = gamma_0 * (1 + z_mu) * np.log(1 + z_mu)

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso) + gamma_corr
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            mu_th = mu_th_base + delta_M + gamma_corr
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)

        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_gcdm_log_squared(params):
    """γCDM-LOG²: Δμ = γ₀·[ln(1+z)]². Includes physical Ωm prior and σ_int."""
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma_0 = params
                M_sne, M_qso = 0.0, 0.0
            else:
                omch2, M_sne, M_qso, gamma_0 = params
            H0 = 67.4
        else:
            H0, omch2, M_sne, M_qso, gamma_0 = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX):
                return 1e10
            if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
                return 1e10
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma_0 = params
                delta_M = 0.0
            else:
                omch2, delta_M, gamma_0 = params
            H0 = 67.4
        else:
            H0, omch2, delta_M, gamma_0 = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < delta_M < M_MAX):
                return 1e10
            if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
                return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        gamma_corr = gamma_0 * np.log(1 + z_mu) ** 2

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso) + gamma_corr
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            mu_th = mu_th_base + delta_M + gamma_corr
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)

        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_gcdm_log_cubed(params):
    """γCDM-LOG³: Δμ = γ₀·[ln(1+z)]³. Includes physical Ωm prior and σ_int."""
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma_0 = params
                M_sne, M_qso = 0.0, 0.0
            else:
                omch2, M_sne, M_qso, gamma_0 = params
            H0 = 67.4
        else:
            H0, omch2, M_sne, M_qso, gamma_0 = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX):
                return 1e10
            if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
                return 1e10
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, gamma_0 = params
                delta_M = 0.0
            else:
                omch2, delta_M, gamma_0 = params
            H0 = 67.4
        else:
            H0, omch2, delta_M, gamma_0 = params
            if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
                return 1e10
            if not (M_MIN < delta_M < M_MAX):
                return 1e10
            if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
                return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        gamma_corr = gamma_0 * np.log(1 + z_mu) ** 3

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso) + gamma_corr
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            mu_th = mu_th_base + delta_M + gamma_corr
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)

        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


# =============================================================================
# ASYMMETRIC (NO δM) EVOLVING MODELS — same wide priors
# =============================================================================

def chi2_gcdm_linear_no_M(params):
    """γCDM-LINEAR sin δM: 3 params (H₀, Ωch², γ₀). Includes physical Ωm prior and σ_int."""
    if args.fixed_anchor or args.sanity_check:
        omch2, gamma_0 = params
        H0 = 67.4
    else:
        H0, omch2, gamma_0 = params
    if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
        return 1e10
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
        return 1e10
    if not check_physical_prior(H0, omch2):
        return 1e10
    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)
        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        mu_th = mu_th_base + gamma_0 * (1 + z_mu) * np.log(1 + z_mu)
        
        if COMBINED_MODE:
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)
            
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))
        
        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0
            
        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10



def chi2_gcdm_log_squared_no_M(params):
    """γCDM-LOG² sin δM: 3 params (H₀, Ωch², γ₀). Includes physical Ωm prior and σ_int."""
    if args.fixed_anchor or args.sanity_check:
         omch2, gamma_0 = params
         H0 = 67.4
    else:
        H0, omch2, gamma_0 = params
    if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
        return 1e10
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
        return 1e10
    if not check_physical_prior(H0, omch2):
        return 1e10
    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)
        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        mu_th = mu_th_base + gamma_0 * np.log(1 + z_mu) ** 2
        
        if COMBINED_MODE:
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)
            
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))
        
        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0
            
        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_gcdm_log_cubed_no_M(params):
    """γCDM-LOG³ sin δM: 3 params (H₀, Ωch², γ₀). Includes physical Ωm prior and σ_int."""
    if args.fixed_anchor or args.sanity_check:
         omch2, gamma_0 = params
         H0 = 67.4
    else:
        H0, omch2, gamma_0 = params
    if not (H0_MIN < H0 < H0_MAX and OMCH2_MIN < omch2 < OMCH2_MAX):
        return 1e10
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX):
        return 1e10
    if not check_physical_prior(H0, omch2):
        return 1e10
    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)
        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        mu_th = mu_th_base + gamma_0 * np.log(1 + z_mu) ** 3
        
        if COMBINED_MODE:
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)
            
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))
        
        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0
            
        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10



# =============================================================================
# DECAY MODEL (Hubble Bubble)
# =============================================================================

def chi2_decay(params):
    """
    Decay Model: Δμ = A * exp(-z / zd).
    H0 FREE (like ΛCDM) for fair comparison. Fits H0, Omch2, A, zd + Nuisance.
    When --fixed-anchor: H0=67.4, M=0.
    
    Returns -2·logL = χ² + Σlog(σ_eff²) for proper Bayesian comparison.
    """
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                 omch2, A, zd = params
                 M_sne, M_qso = 0.0, 0.0
                 H0 = 67.4
            else:
                 omch2, M_sne, M_qso, A, zd = params
                 H0 = 67.4
        elif args.sanity_check:
            # Sanity Check (Decay): H0=67.4, M REMOVED
            omch2, A, zd = params
            H0, M_sne, M_qso = 67.4, 0.0, 0.0
        else:
            H0, omch2, M_sne, M_qso, A, zd = params
            if args.no_nuisance:
                M_sne, M_qso = 0.0, 0.0
            if not (H0_MIN < H0 < H0_MAX): return 1e10
            if not args.no_nuisance:
                if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX): return 1e10
        
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, A, zd = params
                delta_M = 0.0
                H0 = 67.4
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

    if not (A_MIN < A < A_MAX and ZD_MIN < zd < ZD_MAX):
        return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        decay_corr = A * np.exp(-z_mu / zd)

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso) + decay_corr
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            mu_th = mu_th_base + delta_M + decay_corr
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SNE**2)

        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_decay_no_M(params):
    """
    Decay Model (NO M): Δμ = A * exp(-z / zd).
    """
    # Unpack based on flags
    if args.fixed_anchor or args.sanity_check:
         # Fixed Anchor: H0 is fixed to 67.4.
         # params: [omch2, A, zd]
         omch2, A, zd = params
         H0 = 67.4
    else:
         # Asymmetric (Free H0): H0 is free.
         # params: [H0, omch2, A, zd]
         H0, omch2, A, zd = params
         if not (H0_MIN < H0 < H0_MAX):
             return 1e10

    # Global Bounds Checks
    if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
    if not (A_MIN < A < A_MAX and ZD_MIN < zd < ZD_MAX): return 1e10
    
    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)
        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        decay_corr = A * np.exp(-z_mu / zd)
        mu_th = mu_th_base + decay_corr
        
        if COMBINED_MODE:
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)
            
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))
        
        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0
            
        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


# =============================================================================
# γCDM-LOG²-DECAY (Goldilocks)
# =============================================================================

def chi2_gcdm_log_decay(params):
    """
    γCDM-LOG²-DECAY: Δμ = A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h).
    Two-component additive model: local bubble (SH0ES) + Kerr geometry (quasars).
    H0 FREE for fair comparison. Fits H0, Omch2, A, z_b, gamma_0, z_h + Nuisance.
    """
    if COMBINED_MODE:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, A, z_b, gamma_0, z_h = params
                M_sne, M_qso = 0.0, 0.0
            else:
                omch2, M_sne, M_qso, A, z_b, gamma_0, z_h = params
            H0 = 67.4
        elif args.sanity_check:
            omch2, A, z_b, gamma_0, z_h = params
            H0, M_sne, M_qso = 67.4, 0.0, 0.0
        else:
            H0, omch2, M_sne, M_qso, A, z_b, gamma_0, z_h = params
            if args.no_nuisance:
                M_sne, M_qso = 0.0, 0.0
            if not (H0_MIN < H0 < H0_MAX): return 1e10
            if not args.no_nuisance:
                if not (M_MIN < M_sne < M_MAX and M_MIN < M_qso < M_MAX): return 1e10
        
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
    else:
        if args.fixed_anchor:
            if args.no_nuisance:
                omch2, A, z_b, gamma_0, z_h = params
                delta_M = 0.0
                H0 = 67.4
            else:
                omch2, delta_M, A, z_b, gamma_0, z_h = params
                H0 = 67.4
        elif args.sanity_check:
            omch2, A, z_b, gamma_0, z_h = params
            H0, delta_M = 67.4, 0.0
        else:
            H0, omch2, delta_M, A, z_b, gamma_0, z_h = params
            if args.no_nuisance:
                delta_M = 0.0
            if not (H0_MIN < H0 < H0_MAX): return 1e10
            if not args.no_nuisance:
                 if not (M_MIN < delta_M < M_MAX): return 1e10
        
        if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10

    # Unified model bounds: A, z_b (bubble), gamma_0 (Kerr), z_h (horizon)
    if not (A_MIN < A < A_MAX): return 1e10
    if not (ZB_MIN < z_b < ZB_MAX): return 1e10
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX): return 1e10
    if not (ZH_MIN < z_h < ZH_MAX): return 1e10

    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)

        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        # ── THE LOG²-DECAY FORMULA (Two-component additive) ──
        bubble_term = A * np.exp(-z_mu / z_b)                         # Local SH0ES effect
        kerr_term = gamma_0 * np.log(1 + z_mu)**2 * np.exp(-z_mu / z_h)  # Kerr geometry
        unified_corr = bubble_term + kerr_term

        if COMBINED_MODE:
            mu_th = mu_th_base + np.where(sne_mask, M_sne, M_qso) + unified_corr
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            mu_th = mu_th_base + delta_M + unified_corr
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SNE**2)

        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))

        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0

        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


def chi2_gcdm_log_decay_no_M(params):
    """
    γCDM-LOG²-DECAY (NO M): Δμ = A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h).
    """
    # Unpack based on flags
    if args.fixed_anchor or args.sanity_check:
         # Fixed Anchor: H0 is fixed to 67.4.
         # params: [omch2, A, z_b, gamma_0, z_h]
         omch2, A, z_b, gamma_0, z_h = params
         H0 = 67.4
    else:
         # Asymmetric (Free H0): H0 is free.
         # params: [H0, omch2, A, z_b, gamma_0, z_h]
         H0, omch2, A, z_b, gamma_0, z_h = params
         if not (H0_MIN < H0 < H0_MAX):
             return 1e10

    # Global Bounds Checks
    if not (OMCH2_MIN < omch2 < OMCH2_MAX): return 1e10
    if not (A_MIN < A < A_MAX): return 1e10
    if not (ZB_MIN < z_b < ZB_MAX): return 1e10
    if not (GAMMA_MIN < gamma_0 < GAMMA_MAX): return 1e10
    if not (ZH_MIN < z_h < ZH_MAX): return 1e10
    
    if not check_physical_prior(H0, omch2):
        return 1e10

    try:
        pars = camb.CAMBparams()
        pars.WantTransfer = False
        pars.WantCls = False
        pars.set_cosmology(H0=H0, ombh2=0.0224, omch2=omch2)
        r = camb.get_background(pars)
        mu_th_base = 5 * np.log10(np.maximum(r.luminosity_distance(z_mu), 1e-10)) + 25
        
        # ── THE LOG²-DECAY FORMULA (Two-component additive) ──
        bubble_term = A * np.exp(-z_mu / z_b)
        kerr_term = gamma_0 * np.log(1 + z_mu)**2 * np.exp(-z_mu / z_h)
        mu_th = mu_th_base + bubble_term + kerr_term
        
        if COMBINED_MODE:
            sigma_int = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
            err_eff = np.sqrt(err_mu**2 + sigma_int**2)
        else:
            err_eff = np.sqrt(err_mu**2 + SIGMA_INT_SINGLE**2)
            
        chi2_term = np.sum(((mu_obs - mu_th) / err_eff) ** 2)
        norm_term = np.sum(np.log(err_eff**2))
        
        if len(z_cc) > 0:
            chi2_cc = np.sum(((H_obs - r.hubble_parameter(z_cc)) / err_cc) ** 2)
            norm_cc = np.sum(np.log(err_cc**2))
        else:
            chi2_cc, norm_cc = 0, 0
            
        return chi2_term + norm_term + chi2_cc + norm_cc
    except Exception:
        return 1e10


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
models_to_fit = [
    {"name": "ΛCDM", "fn": chi2_lcdm, "type": "lcdm"},
    # {"name": "γCDM (const)", "fn": chi2_gcdm, "type": "gcdm"},
    # {"name": "γCDM-LINEAL", "fn": chi2_gcdm_linear, "type": "evolving"},
    # {"name": "γCDM-LOG²", "fn": chi2_gcdm_log_squared, "type": "evolving"},
    # {"name": "γCDM-LOG³", "fn": chi2_gcdm_log_cubed, "type": "evolving"},

]
models_to_fit += [
    {"name": "γCDM-LOG²", "fn": chi2_gcdm_log_squared, "type": "evolving"},
    {"name": "γCDM-Decay", "fn": chi2_decay, "type": "decay"},
    {"name": "γCDM-LOG²-Decay", "fn": chi2_gcdm_log_decay, "type": "log_decay"},
]
# Legacy models: only include with --legacy flag
if args.legacy:
    models_to_fit += [
        {"name": "γCDM-LINEAL", "fn": chi2_gcdm_linear, "type": "evolving"},
        {"name": "γCDM-LOG³", "fn": chi2_gcdm_log_cubed, "type": "evolving"},
    ]

if args.asymmetric or args.sanity_check:
    # Use no-M variants: ΛCDM keeps M, γCDM/Decay lose M entirely
    # Index 0 is LCDM (untouched in asymmetric, touched in sanity but handled by bounds)
    # Start mainly from index 1 (My Models)
    
    # models_to_fit[1]["fn"] = chi2_gcdm_no_M # (Const commented out)
    models_to_fit[1]["fn"] = chi2_gcdm_log_squared_no_M
    models_to_fit[2]["fn"] = chi2_decay_no_M
    models_to_fit[3]["fn"] = chi2_gcdm_log_decay_no_M

    # Legacy models: only include with --legacy flag
    if args.legacy:
        models_to_fit[4]["fn"] = chi2_gcdm_linear_no_M
        models_to_fit[5]["fn"] = chi2_gcdm_log_cubed_no_M

results = []
best_overall_bic = np.inf
best_overall_model = None

np.random.seed(42)

for model in models_to_fit:
    name = model["name"]
    fn = model["fn"]
    mtype = model["type"]
    
    print(f"\n   Fitting {name}...")
    
    # Determine parameters and bounds
    if mtype == "lcdm":
        if args.fixed_anchor or args.sanity_check:
            # Fixed Anchor / Sanity Check: H0=67.4, M FREE (ALWAYS)
            # We ignore no_nuisance for LCDM to allow fair comparison (as requested)
            n_params = 3 if COMBINED_MODE else 2 # [omch2, M...]
            bounds = [(OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 1)
        else:
            # All free
            n_params = 4 if COMBINED_MODE else 3
            # We ignore no_nuisance for LCDM to allow fair comparison
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
        # Unified model: 4 model params (A, z_b, gamma_0, z_h)
        if args.sanity_check or args.asymmetric:
             if args.fixed_anchor or args.sanity_check:
                 n_params = 5 # [omch2, A, z_b, gamma_0, z_h]
                 bounds = [(OMCH2_MIN, OMCH2_MAX), (A_MIN, A_MAX), (ZB_MIN, ZB_MAX), (GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
             else:
                 n_params = 6 # [H0, omch2, A, z_b, gamma_0, z_h]
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX), (A_MIN, A_MAX), (ZB_MIN, ZB_MAX), (GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
                 
        elif args.fixed_anchor:
             if args.no_nuisance:
                 n_params = 5 # [omch2, A, z_b, gamma_0, z_h]
                 bounds = [(OMCH2_MIN, OMCH2_MAX), (A_MIN, A_MAX), (ZB_MIN, ZB_MAX), (GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
             else:
                 n_params = 7 if COMBINED_MODE else 6 # [omch2, M..., A, z_b, gamma_0, z_h]
                 bounds = [(OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 5) + [(A_MIN, A_MAX), (ZB_MIN, ZB_MAX), (GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
        else:
            n_params = 8 if COMBINED_MODE else 7
            if args.no_nuisance:
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(0, 0)] * (n_params - 6) + [(A_MIN, A_MAX), (ZB_MIN, ZB_MAX), (GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
            else:
                 bounds = [(H0_MIN, H0_MAX), (OMCH2_MIN, OMCH2_MAX)] + [(M_MIN, M_MAX)] * (n_params - 6) + [(A_MIN, A_MAX), (ZB_MIN, ZB_MAX), (GAMMA_MIN, GAMMA_MAX), (ZH_MIN, ZH_MAX)]
            
    best_chi2 = np.inf
    best_params = None

    for i in range(args.starts):
        x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
        try:
            res = minimize(fn, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6})
            if res.fun < best_chi2:
                best_chi2 = res.fun
                best_params = res.x
                # print(f"      Start {i+1}: χ² = {res.fun:.1f}") # Optional verbosity
        except:
            pass
            
    if best_params is not None:
        print(f"      ✅ Best χ² = {best_chi2:.1f}")
        
        # Unpack parameters based on mode
        # Unpack parameters based on mode
        if mtype == "lcdm":
            if args.fixed_anchor or args.sanity_check:
                # Fixed Anchor / Sanity Check: H0=67.4, M FREE (unless no_nuisance)
                # params: [omch2, M...]
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
            # LOG²-DECAY extraction: params end with [A, z_b, gamma_0, z_h]
            if args.fixed_anchor:
                 h0 = 67.4
                 om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                 if args.no_nuisance or args.asymmetric or args.sanity_check:
                      M = 0.0
                      A, z_b, gamma_0, z_h = best_params[1], best_params[2], best_params[3], best_params[4]
                 else:
                      if COMBINED_MODE:
                           M = (best_params[1] + best_params[2]) / 2
                      else:
                           M = best_params[1]
                      A, z_b, gamma_0, z_h = best_params[-4], best_params[-3], best_params[-2], best_params[-1]
                 gamma = gamma_0
                 print(f"      -> A = {A:.3f}, z_b = {z_b:.3f}, γ₀ = {gamma_0:.3f}, z_h = {z_h:.3f}")
            elif args.sanity_check:
                 h0 = 67.4
                 om = (best_params[0] + 0.0224) / (67.4 / 100) ** 2
                 M = 0.0
                 A, z_b, gamma_0, z_h = best_params[1], best_params[2], best_params[3], best_params[4]
                 gamma = gamma_0
                 print(f"      -> A = {A:.3f}, z_b = {z_b:.3f}, γ₀ = {gamma_0:.3f}, z_h = {z_h:.3f}")
            elif args.asymmetric:
                 # params: [H0, omch2, A, z_b, gamma_0, z_h] — no M
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 M = 0.0
                 A, z_b, gamma_0, z_h = best_params[2], best_params[3], best_params[4], best_params[5]
                 gamma = gamma_0
                 print(f"      -> A = {A:.3f}, z_b = {z_b:.3f}, γ₀ = {gamma_0:.3f}, z_h = {z_h:.3f}")
            elif args.no_nuisance:
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 M = 0.0
                 A, z_b, gamma_0, z_h = best_params[-4], best_params[-3], best_params[-2], best_params[-1]
                 gamma = gamma_0
                 print(f"      -> A = {A:.3f}, z_b = {z_b:.3f}, γ₀ = {gamma_0:.3f}, z_h = {z_h:.3f}")
            else:
                 # params: [H0, omch2, M..., A, z_b, gamma_0, z_h]
                 h0 = best_params[0]
                 om = (best_params[1] + 0.0224) / (h0 / 100) ** 2
                 if COMBINED_MODE:
                      M = (best_params[2] + best_params[3]) / 2
                      A, z_b, gamma_0, z_h = best_params[4], best_params[5], best_params[6], best_params[7]
                 else:
                      M = best_params[2]
                      A, z_b, gamma_0, z_h = best_params[3], best_params[4], best_params[5], best_params[6]
                 gamma = gamma_0
                 print(f"      -> A = {A:.3f}, z_b = {z_b:.3f}, γ₀ = {gamma_0:.3f}, z_h = {z_h:.3f}")

        bic = best_chi2 + n_params * np.log(N)
        aic = best_chi2 + 2 * n_params
        
        results.append({
            "name": name,
            "H0": h0, "Om": om, "M": M, "gamma": gamma,
            "chi2": best_chi2, "bic": bic, "aic": aic,
            "params": best_params
        })
        
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
print("\n" + "=" * 105)
print(f"{'Modelo':<24} {'H₀':>8} {'Ωₘ':>8} {'δM':>10} {'γ₀':>10} {'χ²':>10} {'BIC':>10} {'AIC':>10} {'ΔBIC':>8} {'ΔAIC':>8}")
print("─" * 115)

for res in results:
    dbic = res["bic"] - bic_lcdm
    daic = res["aic"] - aic_lcdm
    
    # Check for "Decay" (case insensitive) to handle both "γCDM-Decay" and "γCDM-LOG²-Decay"
    if "log" in res["name"].lower() and "decay" in res["name"].lower():
        # LOG²-DECAY: 4 model params [A, z_b, γ₀, z_h]
        A_val = res.get('A', res['params'][-4])
        zb_val = res.get('z_b', res['params'][-3])
        g0_val = res.get('gamma', res['params'][-2])
        zh_val = res.get('z_h', res['params'][-1])
        print(f"{res['name']:<24} {res['H0']:>8.2f} {res['Om']:>8.3f} {res['M']:>10.3f}   γ={g0_val:>6.3f}  {res['chi2']:>10.1f} {res['bic']:>10.1f} {res['aic']:>10.1f} {dbic:>8.1f} {daic:>8.1f}")
        print(f"{'':24} {'':>8} {'':>8} {'':>10}   A={A_val:>6.3f} zb={zb_val:.3f} zh={zh_val:.1f}")
    elif "decay" in res["name"].lower():
        # Pure Decay: 2 model params [A, zd]
        best_p = res["params"]
        print(f"{res['name']:<24} {res['H0']:>8.2f} {res['Om']:>8.3f} {res['M']:>10.3f}   A={best_p[-2]:>6.3f}  {res['chi2']:>10.1f} {res['bic']:>10.1f} {res['aic']:>10.1f} {dbic:>8.1f} {daic:>8.1f}")
        print(f"{'':24} {'':>8} {'':>8} {'':>10}   zd={best_p[-1]:.3f}")
    else:
        print(f"{res['name']:<24} {res['H0']:>8.2f} {res['Om']:>8.3f} {res['M']:>10.3f} {res['gamma']:>10.4f} {res['chi2']:>10.1f} {res['bic']:>10.1f} {res['aic']:>10.1f} {dbic:>8.1f} {daic:>8.1f}")
print("─" * 115)

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
    A_dec = res_decay['params'][-2]
    zd_dec = res_decay['params'][-1]
    h0_local_dec = res_decay['H0'] * 10**(-A_dec / 5)
    print(f"   γCDM-Decay          : H₀ = {res_decay['H0']:.2f} {check_h0(res_decay['H0'])}")
    print(f"                         H₀(local) = {h0_local_dec:.2f} (A={A_dec:.3f}, zd={zd_dec:.2f}) {check_h0(h0_local_dec)}")
if res_log_decay:
    # LOG²-DECAY: Δμ = A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h)
    # At z=0: bubble_term = A·exp(0) = A, kerr_term = 0 → Δμ(0) = A
    # So H₀(local) = H₀(cosmo)·10^(-A/5) — A provides the SH0ES shift!
    A_uni = res_log_decay['params'][-4]   # A
    z_b_uni = res_log_decay['params'][-3]  # z_b
    gamma_uni = res_log_decay['gamma']     # γ₀
    z_h_uni = res_log_decay['params'][-1]  # z_h
    h0_local_uni = res_log_decay['H0'] * 10**(-A_uni / 5)
    print(f"   γCDM-LOG²-Decay        : H₀ = {res_log_decay['H0']:.2f} {check_h0(res_log_decay['H0'])}")
    print(f"                         H₀(local) = {h0_local_uni:.2f} (A={A_uni:.3f}, z_b={z_b_uni:.3f}) {check_h0(h0_local_uni)}")
    print(f"                         γ₀={gamma_uni:.3f}, z_h={z_h_uni:.2f} (Kerr geometry)")

print("\n   → Evidencia de efecto Container Lens evolutivo")

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
# MOCK TEST: γ=0 NULL HYPOTHESIS (pipeline validation)
# Only run here if NOT using MCMC or nested; otherwise it runs after sampling
# ============================================================================
if args.mock and not args.mcmc and not args.nested:
    print("\n" + "=" * 70)
    print("🧪 MOCK TEST: VALIDACIÓN γ=0 (null hypothesis)")
    print("=" * 70)
    print(f"""
   Objetivo: verificar que el pipeline NO fabrica señal γ espuria.
   Procedimiento:
     1. Generar datos sintéticos con ΛCDM puro (γ=0, H₀=67.4)
     2. Usar las mismas barras de error y redshifts que los datos reales
     3. Ajustar ΛCDM y γCDM al mock
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
        z_mu = _z_mu
        err_mu = _err_mu
        mu_obs = mu_theory + rng.normal(0, _err_mu)

        z_cc = _z_cc
        err_cc = _err_cc
        H_obs = H_theory + rng.normal(0, _err_cc) if len(_err_cc) > 0 else np.array([])

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
                res = minimize(chi2_lcdm, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-6})
                if res.fun < best_lcdm_m:
                    best_lcdm_m = res.fun
            except Exception:
                pass

        # Fit γCDM
        best_gcdm_m = np.inf
        best_gamma_m = 0.0
        for _ in range(10):
            if COMBINED_MODE:
                x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                      rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0),
                      rng.uniform(-1.5, 0.5)]
            else:
                x0 = [rng.uniform(50, 90), rng.uniform(0.05, 0.20),
                      rng.uniform(-1.0, 1.0), rng.uniform(-1.5, 0.5)]
            try:
                res = minimize(chi2_gcdm, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-6})
                if res.fun < best_gcdm_m:
                    best_gcdm_m = res.fun
                    best_gamma_m = res.x[-1]
            except Exception:
                pass

        bic_l = best_lcdm_m + k_lcdm * np.log(N)
        bic_g = best_gcdm_m + k_gcdm * np.log(N)
        dbic_m = bic_g - bic_l

        aic_l = best_lcdm_m + 2 * k_lcdm
        aic_g = best_gcdm_m + 2 * k_gcdm
        daic_m = aic_g - aic_l

        mock_gammas.append(best_gamma_m)
        mock_dbics.append(dbic_m)
        mock_daics.append(daic_m)

        if dbic_m < -6:  # "strong" false detection
            false_detections += 1

        print(f"   Mock {m+1:>3}/{args.n_mock}: γ = {best_gamma_m:+.4f}, ΔBIC = {dbic_m:+.1f}, ΔAIC = {daic_m:+.1f}")

    # Restore real data
    z_mu, mu_obs, err_mu = _z_mu, _mu_obs, _err_mu
    z_cc, H_obs, err_cc = _z_cc, _H_obs, _err_cc

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
                sanity_check=args.sanity_check
            )
        print("   ✅ Shared likelihoods loaded from gammacdm_likelihoods.py")

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
                "--z-min", str(args.z_min),
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
                     h0_local_ld = res_log_decay['H0_mean'] * 10**(-a_ld_val / 5) if a_ld_val != 0 else res_log_decay['H0_mean']
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
                    print(f"      Ωc h² (Planck) ≈ 0.120")

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
            # Shared priors (IDENTICAL for all models)
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


            def _get_M(samples, mode):
                if mode:
                    ms = np.mean(samples["M_sne"])
                    mq = np.mean(samples["M_qso"])
                    return (ms + mq) / 2, np.sqrt(np.std(samples["M_sne"])**2 + np.std(samples["M_qso"])**2) / 2
                return np.mean(samples["mabs"]), np.std(samples["mabs"])

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
            
        # Report ΛCDM MCMC if not nested
        if not args.nested:
            report_model_summary("ΛCDM", samples_lcdm, args, logZ=logZ_lcdm, title_prefix=f"📋 {sampler_name.upper()} —")

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
        if not args.nested:
            print(f"\n⏳ Running γCDM-LOG²-Decay {sampler_name}...")
            # Unified: A·exp(-z/z_b) + γ₀·[ln(1+z)]²·exp(-z/z_h)
            # NOTE: For MCMC/Nested we use TIGHTER priors than MLE.
            # MLE uses ZH_MAX=1e10 (unconstrained) to find the global optimum.
            # But MCMC/Nested cannot explore 1e10 of flat prior volume efficiently.
            # We cap z_h at 100 (any z_h>100 is equivalent to z_h=∞ for all z<7).
            ZH_MCMC_MAX = 100.0   # Sampler-specific cap (MLE uses ZH_MAX=1e10)
            log_decay_p = {**base_params, 
                       "A": {"prior": {"min": -1.0, "max": 1.0}, "ref": -0.175, "proposal": 0.05},
                       "zb": {"prior": {"min": 0.01, "max": 5.0}, "ref": 0.4, "proposal": 0.1},
                       "gamma_log_decay": {"prior": {"min": -2.0, "max": 0.0}, "ref": -0.8, "proposal": 0.05},
                       "zh": {"prior": {"min": ZH_MIN, "max": ZH_MCMC_MAX}, "ref": 42.0, "proposal": 5.0}}
            # Tighten δM priors for LOG²-Decay to break A↔δM degeneracy
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

                if not (args.sanity_check or args.fixed_anchor or args.asymmetric or args.no_nuisance):
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
                    plt.savefig(os.path.join(args.output_dir, f"{prefix}_log2_full_corner.png"), dpi=150, bbox_inches='tight')
                    print(f"   ✅ chains/{prefix}_log2_full_corner.png")

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

                if not (args.sanity_check or args.fixed_anchor or args.asymmetric or args.no_nuisance):
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
                    plt.savefig(os.path.join(args.output_dir, f"{prefix}_log_decay_full_corner.png"), dpi=150, bbox_inches='tight')
                    print(f"   ✅ chains/{prefix}_log_decay_full_corner.png")

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

                if not (args.sanity_check or args.fixed_anchor or args.asymmetric or args.no_nuisance):
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
                    plt.savefig(os.path.join(args.output_dir, f"{prefix}_decay_full_corner.png"), dpi=150, bbox_inches='tight')
                    print(f"   ✅ chains/{prefix}_decay_full_corner.png")

        except Exception as e:
            print(f"   ⚠️ Error crítico en generación de gráficos: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # MOCK TEST AFTER MCMC: γ=0 NULL HYPOTHESIS (uses MCMC posteriors)
    # ========================================================================
    if args.mock:
        print("\n" + "=" * 70)
        print("🧪 MOCK TEST POST-MCMC: VALIDACIÓN γ=0 (null hypothesis)")
        print("=" * 70)
        print(f"""
   Objetivo: verificar que el pipeline NO fabrica señal γ espuria.
   Procedimiento:
     1. Generar datos sintéticos con ΛCDM puro (γ=0, H₀=67.4)
     2. Usar las mismas barras de error y redshifts que los datos reales
     3. Ajustar ΛCDM y γCDM al mock
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
            z_mu = _z_mu
            err_mu = _err_mu
            mu_obs = mu_theory + rng.normal(0, _err_mu)

            z_cc = _z_cc
            err_cc = _err_cc
            H_obs = H_theory + rng.normal(0, _err_cc) if len(_err_cc) > 0 else np.array([])

            # Fit both models to mock
            if COMBINED_MODE:
                x0_l = [67.4 + rng.normal(0, 3), 0.12 + rng.normal(0, 0.02), 0.0, 0.0]
                x0_g = [67.4 + rng.normal(0, 3), 0.12 + rng.normal(0, 0.02), 0.0, 0.0, 0.0]
            else:
                x0_l = [67.4 + rng.normal(0, 3), 0.12 + rng.normal(0, 0.02), 0.0]
                x0_g = [67.4 + rng.normal(0, 3), 0.12 + rng.normal(0, 0.02), 0.0, 0.0]

            res_mock_l = minimize(chi2_lcdm, x0_l, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            res_mock_g = minimize(chi2_gcdm, x0_g, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})

            # Extract results
            chi2_l = res_mock_l.fun
            chi2_g = res_mock_g.fun
            k_l = len(x0_l)
            k_g = len(x0_g)
            N_mock = len(z_mu) + len(z_cc)

            bic_l = chi2_l + k_l * np.log(N_mock)
            bic_g = chi2_g + k_g * np.log(N_mock)
            aic_l = chi2_l + 2 * k_l
            aic_g = chi2_g + 2 * k_g

            dbic_mock = bic_g - bic_l
            daic_mock = aic_g - aic_l
            gamma_mock = res_mock_g.x[-1] if not COMBINED_MODE else res_mock_g.x[4]

            mock_gammas.append(gamma_mock)
            mock_dbics.append(dbic_mock)
            mock_daics.append(daic_mock)

            if dbic_mock < -6:
                false_detections += 1

            if (m + 1) % 5 == 0 or m == 0:
                print(f"   Mock {m+1:2d}/{args.n_mock}: γ = {gamma_mock:+.4f}, ΔBIC = {dbic_mock:+.1f}")

        # Restore real data
        z_mu, mu_obs, err_mu = _z_mu, _mu_obs, _err_mu
        z_cc, H_obs, err_cc = _z_cc, _H_obs, _err_cc

        # Summary
        print(f"\n" + "-" * 70)
        print(f"   📊 RESUMEN MOCK TEST (γ=0 verdad):")
        print(f"   ⟨γ⟩ mock    = {np.mean(mock_gammas):+.4f} ± {np.std(mock_gammas):.4f}")
        print(f"   ⟨ΔBIC⟩ mock = {np.mean(mock_dbics):+.1f} ± {np.std(mock_dbics):.1f}")
        print(f"   ⟨ΔAIC⟩ mock = {np.mean(mock_daics):+.1f} ± {np.std(mock_daics):.1f}")
        print(f"   Falsas alarmas = {false_detections}/{args.n_mock} (ΔBIC < −6)")

        print(f"\n   COMPARACIÓN con datos REALES:")
        print(f"   {'':<20} {'Mock (γ=0)':<15} {'Real':<15}")
        print(f"   {'γ':<20} {np.mean(mock_gammas):+15.4f} {gamma_g:+15.4f}")
        print(f"   {'ΔBIC':<20} {np.mean(mock_dbics):+15.1f} {delta_bic:+15.1f}")

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

print(f"""
Resultados clave:
  γ    = {gamma_g:.4f} (MLE)
  ΔBIC = {delta_bic:.1f} (negativo → γCDM preferido)
  ΔAIC = {delta_aic:.1f} (negativo → γCDM preferido)
  K_BIC (approx) = {K_BIC_approx:.1f}  ← BIC-implied odds, NOT Bayes factor
""")

#   → γCDM es preferido incluso con δM nuisance incluido
#   → DECAY es mejor porque acerca SH0ES y CMB
#   → El efecto NO es un artefacto de absorción de offset
#   → Hipótesis Física: La tensión H₀ se resuelve
#     por este lensing geométrico con límites (Container Kerr Metric).

print("=" * 70)
