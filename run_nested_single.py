#!/usr/bin/env python3
"""
Single-model nested sampling runner - runs ONE PolyChord model per process.
Avoids MPI_FINALIZE issue by running in isolated process.

Usage:
    python run_nested_single.py <model> --nlive 100 --sigma-int-sne 0.1 --sigma-int-qso 0.4 --qso-err-cut 10.0
    
    model: lcdm, log2
"""
import argparse
import json
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["lcdm", "log2"])
parser.add_argument("--nlive", type=int, default=100)
parser.add_argument("--sigma-int-sne", type=float, default=0.1, dest="sigma_int_sne")
parser.add_argument("--sigma-int-qso", type=float, default=0.0, dest="sigma_int_qso")
parser.add_argument("--qso-err-cut", type=float, default=0.5, dest="qso_err_cut")
parser.add_argument("--output-dir", type=str, default="chains", dest="output_dir")
parser.add_argument("--no-nuisance", action="store_true", help="Fix calibration offsets (M_sne, M_qso) to 0")
parser.add_argument("--asymmetric", action="store_true", help="γCDM without δM (test if γ absorbs offset)")
args = parser.parse_args()

print(f"=" * 70)
print(f"🔮 NESTED SAMPLING: {args.model.upper()}")
print(f"=" * 70)

# ============================================================================
# LOAD DATA (unified from full_dataset.csv)
# ============================================================================
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "full_dataset.csv")

if not os.path.exists(csv_path):
    # Try one level up if not in notebooks/
    csv_path = os.path.join(os.path.dirname(script_dir), "full_dataset.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Fallback to GitHub if local fails
    print("⚠️ Local dataset not found, falling back to GitHub...")
    df = pd.read_csv('https://raw.githubusercontent.com/indigenica/akashic-alpha-engine/main/full_dataset.csv')

# Filter and extract exactly as in gammacdm_anticheat_validation.py
sne = df[(df['probe'] == 'sne_ia') & (df['type'] == 'mu')]
cc = df[(df['probe'] == 'cc') & (df['type'] == 'H')]
qso = df[(df['probe'] == 'quasar') & (df['type'] == 'mu') & (df['err'] < args.qso_err_cut)]

n_sne = len(sne)
n_qso = len(qso)

mu_data = pd.concat([sne, qso])
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
# SETUP LIKELIHOOD
# ============================================================================
from cobaya.likelihood import Likelihood
from cobaya.run import run

class BaseLikelihood(Likelihood):
    model_type = "lcdm"

    def initialize(self):
        self.z_mu = z_mu
        self.mu_obs = mu_obs
        self.err_eff_mu = err_eff_mu
        self.sne_mask = sne_mask
        self.z_cc = z_cc
        self.H_obs = H_obs
        self.err_cc = err_cc
        self.norm_mu = np.sum(np.log(self.err_eff_mu**2))
        self.norm_cc = np.sum(np.log(self.err_cc**2)) if len(self.z_cc) > 0 else 0

    
    def get_requirements(self):
        return {"angular_diameter_distance": {"z": self.z_mu},
                "Hubble": {"z": self.z_cc}}
    
    def logp(self, _derived=None, **params):
        try:
            DA = self.provider.get_angular_diameter_distance(self.z_mu)
            DL = DA * (1 + self.z_mu)**2
        except:
            return -1e30
        
        mu_th = 5 * np.log10(np.maximum(DL, 1e-10)) + 25
        
        # Model-specific correction
        if self.model_type == "log2":
            gamma0 = params.get("gamma_log2", 0.0)
            ln1pz = np.log1p(self.z_mu)
            mu_th = mu_th + gamma0 * ln1pz**2
        
        # Add calibration offsets (ignored if --no-nuisance)
        if not args.no_nuisance:
            mu_th = mu_th + np.where(self.sne_mask, params.get('M_sne', 0), params.get('M_qso', 0))
        
        # -2*logL = chi2 + log(sigma^2)
        chi2_mu = np.sum(((self.mu_obs - mu_th) / self.err_eff_mu)**2)
        logL = -0.5 * (chi2_mu + self.norm_mu)
        
        if len(self.z_cc) > 0:
            H_th = self.provider.get_Hubble(self.z_cc)
            chi2_cc = np.sum(((self.H_obs - H_th) / self.err_cc)**2)
            logL += -0.5 * (chi2_cc + self.norm_cc)
        
        return logL

class LCDMLikelihood(BaseLikelihood):
    model_type = "lcdm"
    params = {"M_sne": None, "M_qso": None}

class LOG2Likelihood(BaseLikelihood):
    model_type = "log2"
    params = {"gamma_log2": None, "M_sne": None, "M_qso": None}




# ============================================================================
# SETUP SAMPLER
# ============================================================================
base_params = {
    "H0": {"prior": {"min": 40, "max": 100}, "ref": 67, "proposal": 2.0},
    "omch2": {"prior": {"min": 0.01, "max": 0.35}, "ref": 0.12, "proposal": 0.02},
    "M_sne": {"prior": {"min": -3.0, "max": 3.0}, "ref": 0.0, "proposal": 0.1} if not (args.no_nuisance or args.asymmetric) else 0.0,
    "M_qso": {"prior": {"min": -3.0, "max": 3.0}, "ref": 0.0, "proposal": 0.1} if not (args.no_nuisance or args.asymmetric) else 0.0,
    "ombh2": 0.0224, "ns": 0.965, "As": 2.1e-9, "tau": 0.06
}


if args.model == "lcdm":
    LikelihoodClass = LCDMLikelihood
    params = base_params.copy()
    output_prefix = os.path.join(args.output_dir, "anticheat_lcdm")
else:  # log2
    LikelihoodClass = LOG2Likelihood
    params = base_params.copy()
    params["gamma_log2"] = {"prior": {"min": -2.0, "max": 1.0}, "ref": -1.0, "proposal": 0.05}
    params["H0"]["ref"] = 70  # Neutral starting point
    output_prefix = os.path.join(args.output_dir, "anticheat_log2")

sampler_cfg = {
    "polychord": {
        "nlive": args.nlive,
        "num_repeats": 2 * 5,
        "precision_criterion": 0.01,
        "boost_posterior": 5.0
    }
}

info = {
    "likelihood": {"model": LikelihoodClass},
    "theory": {"camb": {"stop_at_error": True}},
    "params": params,
    "sampler": sampler_cfg,
    "output": output_prefix,
    "force": True
}

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

H0_mean = float(np.mean(samples["H0"]))
H0_std = float(np.std(samples["H0"]))
M_sne_mean = float(np.mean(samples["M_sne"])) if "M_sne" in samples else 0.0
M_qso_mean = float(np.mean(samples["M_qso"])) if "M_qso" in samples else 0.0

results = {
    "model": args.model,
    "logZ": float(logZ) if logZ is not None else None,
    "H0_mean": H0_mean,
    "H0_std": H0_std,
    "omch2_mean": float(np.mean(samples["omch2"])),
    "omch2_std": float(np.std(samples["omch2"])),
    "M_sne_mean": M_sne_mean,
    "M_qso_mean": M_qso_mean,
    "deltaM_mean": (M_sne_mean + M_qso_mean) / 2
}

if args.model == "log2":
    results["gamma_log2_mean"] = float(np.mean(samples["gamma_log2"]))
    results["gamma_log2_std"] = float(np.std(samples["gamma_log2"]))



# Save results
results_file = output_prefix + "_results.json"
with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print(f"\n" + "=" * 70)
print(f"📋 NESTED — {args.model.upper()}")
print(f"=" * 70)
print(f"   H₀    = {H0_mean:.2f} ± {H0_std:.2f} km/s/Mpc")
if args.model == "log2":
    print(f"   γ₀    = {results['gamma_log2_mean']:.4f} ± {results['gamma_log2_std']:.4f}")
print(f"   M_sne = {M_sne_mean:.3f}, M_qso = {M_qso_mean:.3f}")
print(f"   ⟨δM⟩  = {results['deltaM_mean']:.3f}")
if logZ:
    print(f"   log(Z) = {logZ:.2f}")
print(f"\n✅ Results saved to: {results_file}")
