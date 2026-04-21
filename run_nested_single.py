#!/usr/bin/env python3
"""
Single-model nested sampling runner - runs ONE PolyChord model per process.
Avoids MPI_FINALIZE issue by running in isolated process.

Usage:
    python run_nested_single.py <model> --nlive 100 --sigma-int-sne 0.1 --sigma-int-qso 0.4 --qso-err-cut 1.5
    
    model: lcdm, log2
"""
import argparse
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Shared physical constants and H0_local helper (single source of truth).
from cosmo_constants import (
    SH0ES_Z_PIVOT, Z_PIVOT,
)
from gammacdm_core import h0_local

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["lcdm", "log2", "decay", "log_decay"])
parser.add_argument("--nlive", type=int, default=100)
parser.add_argument("--sigma-int-sne", type=float, default=0.1, dest="sigma_int_sne")
parser.add_argument("--sigma-int-qso", type=float, default=0.4, dest="sigma_int_qso")
parser.add_argument("--qso-err-cut", type=float, default=1.5, dest="qso_err_cut")
parser.add_argument("--sne-err-cut", type=float, default=0.5, dest="sne_err_cut")
parser.add_argument("--z-min-sne", type=float, default=0.01, dest="z_min_sne")
parser.add_argument("--z-min-qso", type=float, default=0.0, dest="z_min_qso")
parser.add_argument("--revised", action="store_true", help="Use full_dataset_revisado.csv instead of full_dataset.csv")
parser.add_argument("--output-dir", type=str, default="chains", dest="output_dir")
parser.add_argument("--no-nuisance", action="store_true", help="Fix calibration offsets (M_sne, M_qso) to 0")
parser.add_argument("--asymmetric", action="store_true", help="γCDM without δM (test if γ absorbs offset)")
parser.add_argument("--no-quasars", action="store_true", help="Exclude Quasars from analysis")
parser.add_argument("--quasars-only", action="store_true", help="Evaluate Quasars only")
parser.add_argument("--fixed-anchor", action="store_true", help="Fix H0=67.4 and M=SH0ES (M=0) for ALL models")
parser.add_argument("--sanity-check", action="store_true", help="Internal sanity check: ΛCDM(H0=67.4,Ωm free,M free) vs γCDM/Decay(H0=67.4,M removed)")
parser.add_argument("--student", action="store_true", help="Use Student-t likelihood (robust to outliers, default ν=5)")
parser.add_argument("--cauchy", action="store_true", help="Use Cauchy likelihood (ν=1, maximum robustness)")
parser.add_argument("--nu", type=float, default=5.0, help="Degrees of freedom for Student-t (default: 5.0)")
parser.add_argument("--cov", type=str, default="none", choices=["none", "stat", "sys"],
                    help="Covariance matrix type for SNe Ia (default: 'none')")
parser.add_argument("--cmb", action="store_true",
                    help="Add CMB shift parameter prior (Planck 2018)")
parser.add_argument("--fit-scatter", action="store_true", dest="fit_scatter",
                    help="Fit intrinsic scatter σ_int,SNe and σ_int,QSO as free parameters")
args = parser.parse_args()

print(f"=" * 70)
print(f"🔮 NESTED SAMPLING: {args.model.upper()}")
if args.no_quasars:
    print(f"   🔭 MODE: NO QUASARS (SNe + CC only)")
elif args.quasars_only:
    print(f"   🔭 MODE: QUASARS ONLY (QSO + CC)")
print(f"=" * 70)

# Resolve likelihood type
if args.cauchy:
    _lk_type = 'cauchy'
    _nu_val = 1.0
    print(f"   ⚙️  Likelihood: CAUCHY (ν=1)")
elif args.student:
    _lk_type = 'student'
    _nu_val = args.nu
    print(f"   ⚙️  Likelihood: Student-t (ν={_nu_val:.1f})")
else:
    _lk_type = 'gaussian'
    _nu_val = 5.0
    print(f"   ⚙️  Likelihood: Gaussian")

# ============================================================================
# LOAD DATA (unified from full_dataset.csv)
# ============================================================================
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'full_dataset_revisado.csv' if args.revised else 'full_dataset.csv'
csv_path = os.path.join(script_dir, dataset_name)

if not os.path.exists(csv_path):
    # Try one level up if not in notebooks/
    csv_path = os.path.join(os.path.dirname(script_dir), dataset_name)

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Fallback to GitHub if local fails
    print("⚠️ Local dataset not found, falling back to GitHub...")
    df = pd.read_csv('https://raw.githubusercontent.com/indigenica/akashic-alpha-engine/main/' + dataset_name)

# Filter and extract exactly as in gammacdm_verification.py
sne = df[(df['probe'] == 'sne_ia') & (df['type'] == 'mu') & (df['err'] < args.sne_err_cut) & (df['z'] > args.z_min_sne)]
cc = df[(df['probe'] == 'cc') & (df['type'] == 'H')]

if args.no_quasars:
    qso = pd.DataFrame(columns=df.columns)
    mu_data = sne.copy()
    COMBINED_MODE = False
elif getattr(args, 'quasars_only', False):
    sne = pd.DataFrame(columns=df.columns)
    qso = df[(df['probe'] == 'quasar') & (df['type'] == 'mu') & (df['err'] < args.qso_err_cut) & (df['z'] > args.z_min_qso)]
    mu_data = qso.copy()
    COMBINED_MODE = False
else:
    qso = df[(df['probe'] == 'quasar') & (df['type'] == 'mu') & (df['err'] < args.qso_err_cut) & (df['z'] > args.z_min_qso)]
    mu_data = pd.concat([sne, qso])
    COMBINED_MODE = True

n_sne = len(sne)
n_qso = len(qso)
z_mu = mu_data['z'].values
mu_obs = mu_data['val'].values
err_mu = mu_data['err'].values

sne_mask = np.zeros(len(mu_data), dtype=bool)
sne_mask[:n_sne] = True

z_cc = cc['z'].values
H_obs = cc['val'].values
err_cc = cc['err'].values

N = len(mu_data) + len(cc)

SIGMA_INT_SNE = args.sigma_int_sne
SIGMA_INT_QSO = args.sigma_int_qso

# Precompute effective errors for normalization term
sigma_int_mu = np.where(sne_mask, SIGMA_INT_SNE, SIGMA_INT_QSO)
err_eff_mu = np.sqrt(err_mu**2 + sigma_int_mu**2)

print(f"📊 Dataset: {n_sne} SNe + {n_qso} QSOs + {len(cc)} CC")
print(f"📊 Total points N = {N}")

# ============================================================================
# COVARIANCE MATRIX (optional, Pantheon+)
# ============================================================================
C_inv_sne = None
ln_det_C_sne = None
_cov_evals_nested = None
_cov_evecs_nested = None

if args.cov != 'none' and n_sne > 0:
    cov_file = "Pantheon+SH0ES_STATONLY.cov" if args.cov == 'stat' else "Pantheon+SH0ES_STAT+SYS.cov"
    cov_path = os.path.join(script_dir, cov_file)

    if os.path.exists(cov_path):
        print(f"🧮 Loading SNe Covariance: {cov_file}")
        with open(cov_path, 'r') as f:
            C_dim = int(f.readline().strip())
        C_flat = np.loadtxt(cov_path, skiprows=1)
        C_full = C_flat.reshape(C_dim, C_dim)

        sne_all = df[(df['probe'] == 'sne_ia') & (df['type'] == 'mu')].copy()
        sne_all['cov_idx'] = np.arange(len(sne_all))
        sne_filt = sne_all[(sne_all['err'] < args.sne_err_cut) & (sne_all['z'] > args.z_min_sne)]
        sne_idx = sne_filt['cov_idx'].values.astype(int)

        C_sne = C_full[np.ix_(sne_idx, sne_idx)]

        _cov_evals_nested = None
        _cov_evecs_nested = None
        if args.fit_scatter:
            print(f"   Eigendecomposing base covariance for σ_int fitting...")
            _cov_evals_nested, _cov_evecs_nested = np.linalg.eigh(C_sne)
            _cov_evals_nested = np.maximum(_cov_evals_nested, 1e-15)

        if not args.fit_scatter and args.sigma_int_sne > 0:
            np.fill_diagonal(C_sne, C_sne.diagonal() + args.sigma_int_sne**2)

        C_inv_sne = np.linalg.inv(C_sne)
        sign, ln_det_C_sne = np.linalg.slogdet(C_sne)
        if sign <= 0:
            raise ValueError("SNe covariance determinant is not positive.")
        print(f"   ✅ Covariance ready ({len(sne_idx)}×{len(sne_idx)})")
    else:
        print(f"   ⚠️ Covariance file not found: {cov_path}, using diagonal errors")

# ============================================================================
# SETUP LIKELIHOOD (shared module)
# ============================================================================
from cobaya.run import run
from gammacdm_likelihoods import create_likelihoods

_cov_ev = _cov_evals_nested
_cov_ec = _cov_evecs_nested

LCDMLikelihood, GammaCDM_LOG2_Likelihood, DecayLikelihood, LogDecayLikelihood = \
    create_likelihoods(
        z_mu=z_mu, mu_obs=mu_obs, err_mu=err_mu,
        z_cc=z_cc, H_obs=H_obs, err_cc=err_cc,
        sne_mask=sne_mask, combined_mode=COMBINED_MODE,
        sigma_int_sne=SIGMA_INT_SNE, sigma_int_qso=SIGMA_INT_QSO,
        no_nuisance=args.no_nuisance, asymmetric=args.asymmetric,
        sanity_check=args.sanity_check,
        likelihood_type=_lk_type, nu=_nu_val,
        C_inv_sne=C_inv_sne, ln_det_C_sne=ln_det_C_sne,
        use_cmb=args.cmb, fit_scatter=args.fit_scatter,
        cov_evals=_cov_ev, cov_evecs=_cov_ec
    )
# Alias for backward compat with model selection below
LOG2Likelihood = GammaCDM_LOG2_Likelihood
GammaCDM_LOG_DECAY_Likelihood = LogDecayLikelihood
print("   ✅ Shared likelihoods loaded from gammacdm_likelihoods.py")


# ============================================================================
# SETUP SAMPLER
# ============================================================================
base_params = {
    "H0": {"prior": {"min": 40, "max": 100}, "ref": 70, "proposal": 2.0},
    "omch2": {"prior": {"min": 0.01, "max": 0.35}, "ref": 0.12, "proposal": 0.02},
    "ombh2": 0.0224, "ns": 0.965, "As": 2.1e-9, "tau": 0.06
}
# Gaussian calibration priors (externally justified):
#   M_sne ~ N(0, 0.05): Pantheon+ residual calibration uncertainty ~0.02-0.04 mag
#   M_qso ~ N(0, 0.15): Lusso+20 L_X-L_UV intercept uncertainty ~0.1-0.2 mag
# Same priors applied to ALL models for fair Bayesian comparison.
if COMBINED_MODE:
    base_params["M_sne"] = {"prior": {"dist": "norm", "loc": 0, "scale": 0.05}, "ref": 0.0, "proposal": 0.02}
    base_params["M_qso"] = {"prior": {"dist": "norm", "loc": 0, "scale": 0.15}, "ref": 0.0, "proposal": 0.05}
else:
    base_params["mabs"] = {"prior": {"dist": "norm", "loc": 0, "scale": 0.05}, "ref": 0.0, "proposal": 0.02}
if args.fit_scatter:
    if COMBINED_MODE:
        base_params["sigma_int_sne"] = {"prior": {"min": 0.001, "max": 0.5}, "ref": 0.1, "proposal": 0.01}
        base_params["sigma_int_qso"] = {"prior": {"min": 0.1, "max": 3.0}, "ref": 0.5, "proposal": 0.1}
    elif getattr(args, 'quasars_only', False):
        base_params["sigma_int_qso"] = {"prior": {"min": 0.1, "max": 3.0}, "ref": 0.5, "proposal": 0.1}
    else:
        base_params["sigma_int_sne"] = {"prior": {"min": 0.001, "max": 0.5}, "ref": 0.1, "proposal": 0.01}

defaultH0 = {"prior": {"min": 40, "max": 100}, "ref": 70, "proposal": 2.0}
defaultGammaLog2 = {"prior": {"min": -3.0, "max": 3.0}, "ref": -0.8, "proposal": 0.05}
defaultA = {"prior": {"min": -2.0, "max": 2.0}, "ref": -0.175, "proposal": 0.1}
defaultZd = {"prior": {"min": 0.01, "max": 10.0}, "ref": 3.5, "proposal": 0.1}
# Conservative priors: wider model-specific ranges make the Bayes factor more
# conservative (more prior volume to penalise).  Gaussian δM (in base_params)
# breaks the A↔δM degeneracy via external calibration info, not data dredging.
defaultA_unified = {"prior": {"min": -2.0, "max": 2.0}, "ref": -0.175, "proposal": 0.1}
defaultZb = {"prior": {"min": 0.01, "max": 5.0}, "ref": 0.4, "proposal": 0.1}
defaultGammaLogDecay = {"prior": {"min": -3.0, "max": 1.0}, "ref": -0.8, "proposal": 0.1}
# z_h is a scale parameter spanning orders of magnitude → sample log10(z_h)
# uniformly (log-uniform on z_h ∈ [1, 200]).
defaultLogZh = {"prior": {"min": 0.0, "max": 2.301}, "ref": 1.62, "proposal": 0.15}

# Apply constraints to base_params (ΛCDM always keeps M free)
if args.fixed_anchor:
    print("   ⚓ FIXED ANCHOR: H0=67.4")
    base_params["H0"] = 67.4
    # M_sne / M_qso remain FREE (inherited from base_params) unless --no-nuisance is passed
elif args.sanity_check:
    print("   🧠 SANITY CHECK: H0=67.4")
    base_params["H0"] = 67.4
    # ΛCDM fixes omch2, keeps M free (handled in model block)
    # γCDM/Decay keeps omch2 free, removes M (handled in model block)
else:
    pass
    # --no-nuisance / --asymmetric applied PER MODEL below (ΛCDM keeps M free)



if args.model == "lcdm":
    LikelihoodClass = LCDMLikelihood
    params = base_params.copy()
    # ΛCDM ALWAYS keeps M free (except --fixed-anchor)
    if args.sanity_check:
        print("   🧠 ΛCDM Sanity: M free, Ωm free (Fairer)")
        # params["omch2"] = 0.12  <-- REMOVED: Let it be free to fit Quasars
    output_prefix = os.path.join(args.output_dir, "nested_lcdm")
elif args.model == "log2":
    LikelihoodClass = LOG2Likelihood
    params = base_params.copy()
    params["gamma_log2"] = defaultGammaLog2
    if not args.fixed_anchor:
        params["H0"] = defaultH0
    # Apply --no-nuisance (fix M=0) or --asymmetric (remove M)
    # FOR γCDM models, these flags SHOULD work even with fixed-anchor
    if args.asymmetric or args.sanity_check:
        print("   ✂️  Asymmetric/Sanity: M REMOVED from γCDM-LOG²")
        if COMBINED_MODE:
            params["M_sne"] = 0.0
            params["M_qso"] = 0.0
        else:
            params["mabs"] = 0.0
    elif args.no_nuisance:
        print("   🔒 No-nuisance: M FIXED to 0 for γCDM-LOG²")
        if COMBINED_MODE:
            params["M_sne"] = 0.0
            params["M_qso"] = 0.0
        else:
            params["mabs"] = 0.0
    output_prefix = os.path.join(args.output_dir, "nested_log2")
elif args.model == "decay":
    LikelihoodClass = DecayLikelihood
    params = base_params.copy()
    # Decay H0 freedom should match other models for fair comparison
    if not args.fixed_anchor and not args.sanity_check:
        params["H0"] = defaultH0
    params["A"] = defaultA
    params["zd"] = defaultZd
    # Apply --no-nuisance (fix M=0) or --asymmetric (remove M)
    if args.asymmetric or args.sanity_check:
        print("   ✂️  Asymmetric/Sanity: M REMOVED from Decay model")
        if COMBINED_MODE:
            params["M_sne"] = 0.0
            params["M_qso"] = 0.0
        else:
            params["mabs"] = 0.0
    elif args.no_nuisance:
        print("   🔒 No-nuisance: M FIXED to 0 for Decay model")
        if COMBINED_MODE:
            params["M_sne"] = 0.0
            params["M_qso"] = 0.0
        else:
            params["mabs"] = 0.0
    output_prefix = os.path.join(args.output_dir, "nested_decay")
elif args.model == "log_decay":
    LikelihoodClass = LogDecayLikelihood
    params = base_params.copy()
    params["A"] = defaultA_unified
    params["zb"] = defaultZb
    params["gamma_log_decay"] = defaultGammaLogDecay
    params["log_zh"] = defaultLogZh
    params["zh"] = {"value": "lambda log_zh: 10**log_zh", "latex": r"z_h"}
    if not args.fixed_anchor and not args.sanity_check:
        params["H0"] = defaultH0
    if args.asymmetric or args.sanity_check:
        print("   ✂️  Asymmetric/Sanity: M REMOVED from γCDM-LOG²-DECAY")
        if COMBINED_MODE:
            params["M_sne"] = 0.0
            params["M_qso"] = 0.0
        else:
            params["mabs"] = 0.0
    elif args.no_nuisance:
        print("   🔒 No-nuisance: M FIXED to 0 for γCDM-LOG²-DECAY")
        if COMBINED_MODE:
            params["M_sne"] = 0.0
            params["M_qso"] = 0.0
        else:
            params["mabs"] = 0.0
    output_prefix = os.path.join(args.output_dir, "nested_log_decay")

# num_repeats = 5 * ndims (Handley+2015 recommendation for proper decorrelation)
ndims = sum(1 for v in params.values() if isinstance(v, dict) and "prior" in v)
sampler_cfg = {
    "polychord": {
        "nlive": args.nlive,
        "num_repeats": max(10, 5 * ndims),
        "precision_criterion": 0.01,
        "boost_posterior": 5.0
    }
}
print(f"   PolyChord config: nlive={args.nlive}, num_repeats={max(10, 5*ndims)} (ndims={ndims})")

info = {
    "likelihood": {"model": LikelihoodClass},
    "theory": {"camb": {"stop_at_error": True}},
    "params": params,
    "sampler": sampler_cfg,
    "output": output_prefix,
    "force": True
}

# Clean old chain files to prevent getdist from mixing old and new runs
import glob
for f in glob.glob(output_prefix + ".*") + glob.glob(output_prefix + "_*"):
    if not f.endswith("_results.json"): # keep json if needed, but it will be overwritten anyway
        try:
            os.remove(f)
        except OSError:
            pass

# ============================================================================
# RUN
# ============================================================================
print(f"\n⏳ Running PolyChord (nlive={args.nlive})...\n")
_, sampler = run(info)

# ============================================================================
# EXTRACT RESULTS
# ============================================================================
samples = sampler.products()["sample"]
logZ = sampler.products().get("logZ", None)
if logZ is None:
    logZ = getattr(sampler, 'logZ', None)

if args.fixed_anchor or args.sanity_check:
    # H0 is fixed
    H0_mean = 67.4
    H0_std = 0.0
else:
    if "H0" in samples:
        H0_mean = float(np.mean(samples["H0"]))
        H0_std = float(np.std(samples["H0"]))
    else:
        # Fallback if somehow missing but not strictly fixed by logic above
        H0_mean = 67.4
        H0_std = 0.0

def _get_val(key, default=0.0):
    try:
        # For SampleCollection/DataFrame
        if hasattr(samples, 'get'):
            val = samples.get(key)
            if val is not None:
                return float(np.mean(val))
        if key in samples.columns if hasattr(samples, 'columns') else key in samples:
            return float(np.mean(samples[key]))
    except Exception:
        pass
    return default

if COMBINED_MODE:
    M_sne_mean = _get_val("M_sne")
    M_qso_mean = _get_val("M_qso")
    delta_M_mean = (M_sne_mean + M_qso_mean) / 2
else:
    M_sne_mean = _get_val("mabs") if getattr(args, 'no_quasars', False) else 0.0
    M_qso_mean = _get_val("mabs") if getattr(args, 'quasars_only', False) else 0.0
    delta_M_mean = _get_val("mabs")

results = {
    "model": args.model,
    "logZ": float(logZ) if logZ is not None else None,
    "H0_mean": H0_mean,
    "H0_std": H0_std,
    "omch2_mean": float(np.mean(samples["omch2"])),
    "omch2_std": float(np.std(samples["omch2"])),
    "M_sne_mean": M_sne_mean,
    "M_qso_mean": M_qso_mean,
    "deltaM_mean": delta_M_mean,
    "ombh2": 0.0224  # Fixed parameter
}

if args.model == "log2":
    results["gamma_log2_mean"] = float(np.mean(samples["gamma_log2"]))
    results["gamma_log2_std"] = float(np.std(samples["gamma_log2"]))
elif args.model == "decay":
    results["A_mean"] = float(np.mean(samples["A"]))
    results["A_std"] = float(np.std(samples["A"]))
    results["zd_mean"] = float(np.mean(samples["zd"]))
    results["zd_std"] = float(np.std(samples["zd"]))
elif args.model == "log_decay":
    results["A_mean"] = float(np.mean(samples["A"]))
    results["A_std"] = float(np.std(samples["A"]))
    results["zb_mean"] = float(np.mean(samples["zb"]))
    results["zb_std"] = float(np.std(samples["zb"]))
    results["gamma_log_decay_mean"] = float(np.mean(samples["gamma_log_decay"]))
    results["gamma_log_decay_std"] = float(np.std(samples["gamma_log_decay"]))
    # zh is derived from log_zh; reconstruct if not directly available
    if "zh" in (samples.columns if hasattr(samples, 'columns') else samples):
        zh_arr = np.array(samples["zh"])
    else:
        zh_arr = 10**np.array(samples["log_zh"])
    results["zh_mean"] = float(np.mean(zh_arr))
    results["zh_std"] = float(np.std(zh_arr))


# Save results
results_file = output_prefix + "_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"\n" + "=" * 70)
print(f"📋 NESTED — {args.model.upper()}")
print(f"=" * 70)

# 1. Hubble Constant
print(f"   H₀    = {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc {'(Fixed)' if H0_std==0 else ''}")

# 2. Baryonic Matter (omch2)
om_val = float(np.mean(samples["omch2"]))
om_err = float(np.std(samples["omch2"]))
print(f"   Ωch²  = {om_val:.4f} ± {om_err:.4f}")
print(f"   Ωmh²  = {om_val + 0.0224:.4f} (Baryon check, Ωbh²=0.0224)")

# 3. Model Specific Parameters
if args.model == "log2":
    g_val = results['gamma_log2_mean']
    g_err = results['gamma_log2_std']
    print(f"   γ₀    = {g_val:.4f} ± {g_err:.4f}")
    # Spin implication
    beta = abs(g_val) * np.log(10) / 5
    alpha = beta / 2
    spin = np.sqrt(1 - ((1 - alpha) / (1 + alpha))**2) if alpha < 1 else 1.0
    print(f"   🌀 Spin Implied: a/M = {spin:.4f}")
elif args.model == "decay":
    a_val = results['A_mean']
    a_err = results['A_std']
    zd_val = results['zd_mean']
    zd_err = results['zd_std']
    print(f"   A     = {a_val:.3f} ± {a_err:.3f}")
    print(f"   zd    = {zd_val:.3f} ± {zd_err:.3f}")
    # Local H0 (strict z→0 limit; pass z_pivot=SH0ES_Z_PIVOT for rigorous SH0ES match)
    h0_loc = h0_local(H0_mean, A=a_val, z_b=zd_val, z_pivot=Z_PIVOT)
    print(f"   → H₀(local) implied: {h0_loc:.2f} km/s/Mpc")
elif args.model == "log_decay":
    a_val = results['A_mean']
    a_err = results['A_std']
    zb_val = results['zb_mean']
    zb_err = results['zb_std']
    g_val = results['gamma_log_decay_mean']
    g_err = results['gamma_log_decay_std']
    zh_val = results['zh_mean']
    zh_err = results['zh_std']
    print(f"   A     = {a_val:.4f} ± {a_err:.4f} (Bubble amplitude)")
    print(f"   z_b   = {zb_val:.3f} ± {zb_err:.3f} (Bubble decay scale)")
    print(f"   γ₀    = {g_val:.4f} ± {g_err:.4f} (Kerr geometry)")
    print(f"   z_h   = {zh_val:.3f} ± {zh_err:.3f} (Horizon decay scale)")
    # Spin from Kerr component
    beta = abs(g_val) * np.log(10) / 5
    alpha = beta / 2
    spin = np.sqrt(1 - ((1 - alpha) / (1 + alpha))**2) if alpha < 1 else 1.0
    print(f"   🌀 Spin Implied: a/M = {spin:.4f}")
    # Local H0 from bubble component (strict z→0 limit)
    h0_loc = h0_local(H0_mean, A=a_val, z_b=zb_val,
                      gamma_0=g_val, z_h=zh_val, z_pivot=Z_PIVOT)
    print(f"   → H₀(local) implied: {h0_loc:.2f} km/s/Mpc (Bubble A={a_val:.3f})")

# 4. Nuisance Parameters (M)
is_fixed_m = (args.sanity_check or args.no_nuisance or args.asymmetric) and args.model != "lcdm"
if is_fixed_m:
    print(f"   M_sne = 0.000 (Fixed), M_qso = 0.000 (Fixed)")
    print(f"   ⟨δM⟩  = 0.000 (Fixed)")
else:
    print(f"   M_sne = {M_sne_mean:.3f}, M_qso = {M_qso_mean:.3f}")
    print(f"   ⟨δM⟩  = {results['deltaM_mean']:.3f}")

if logZ:
    print(f"   log(Z) = {logZ:.2f}")

print(f"\n✅ Results saved to: {results_file}")

