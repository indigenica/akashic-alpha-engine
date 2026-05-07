import sys
import os
import numpy as np
import scipy.linalg
import arviz as az
from getdist import loadMCSamples

# ==============================================================================
# 1. LOAD DATA & GLOBALS (Via gammacdm_verification hook)
# ==============================================================================
print("Loading gammacdm_verification context...")
user_args = sys.argv[1:]
if not user_args:
    print("No arguments provided. Falling back to default: --revised --fixed-anchor --cov sys ...")
    user_args = [
        "--revised",
        "--fixed-anchor",
        "--cov", "sys",
        "--sigma-int-sne", "0.012",
        "--sigma-int-qso", "1.44",
        "--sne-err-cut", "3.0",
        "--qso-err-cut", "3.0",
        "--cmb",
        "--mcmc"
    ]

sys.argv = ["gammacdm_verification.py", "--loowaic"] + user_args

class StopExecution(Exception): pass
def fake_exit(code=0): raise StopExecution()
sys.exit = fake_exit

gv = {'__file__': 'gammacdm_verification.py'}
try:
    exec(open("gammacdm_verification.py").read(), gv)
except StopExecution:
    print("gammacdm_verification initialized successfully.")

# Extract variables from gv
z_mu = gv['z_mu']
mu_obs = gv['mu_obs']
err_mu = gv['err_mu']
sne_mask = gv['sne_mask']
qso_mask = gv['qso_mask']
z_cc = gv['z_cc']
H_obs = gv['H_obs']
err_cc = gv['err_cc']

args = gv['args']
sigma_int_sne = args.sigma_int_sne
sigma_int_qso = args.sigma_int_qso

err_eff = np.sqrt(err_mu**2 + np.where(sne_mask, sigma_int_sne, sigma_int_qso)**2)

# Extract SNe covariance
C_inv_sne = gv.get('C_inv_sne', None)
if C_inv_sne is not None:
    print("Applying Cholesky decomposition to SNe covariance...")
    # C_inv = L_inv.T @ L_inv, so C = L @ L.T
    C_sne = np.linalg.inv(C_inv_sne)
    L = scipy.linalg.cholesky(C_sne, lower=True)
    L_inv = scipy.linalg.inv(L)
    L_diag_log = np.log(np.diag(L))

# ==============================================================================
# 2. POINTWISE LIKELIHOOD FUNCTION
# ==============================================================================
def pointwise_logL(params, model_name):
    H0 = params.get('H0', 67.4)
    omch2 = params.get('omch2', 0.120)
    M_sne = params.get('M_sne', 0.0)
    M_qso = params.get('M_qso', 0.0)
    
    # Get base cosmology directly from gv's interpolator!
    _fast_camb_bg = gv['_fast_camb_bg']
    _mu_bg, _hz_bg, _da_bg = _fast_camb_bg(H0, omch2)
    
    # Corrections
    if model_name == "log_decay":
        A = params.get('A', 0.0)
        zb = params.get('zb', 1.0)
        g0 = params.get('gamma_log_decay', 0.0)
        zh = params.get('zh', 10.0)
        
        corr = A * np.exp(-z_mu / zb) + g0 * np.log1p(z_mu)**2 * np.exp(-z_mu / zh)
    else:
        corr = 0.0
        
    mu_th = _mu_bg + np.where(sne_mask, M_sne, M_qso) + corr
    
    # --- SNe Log-Likelihood ---
    res_sne = mu_obs[sne_mask] - mu_th[sne_mask]
    if C_inv_sne is not None:
        v_sne = L_inv @ res_sne
        ll_sne = -0.5 * np.log(2 * np.pi) - L_diag_log - 0.5 * v_sne**2
    else:
        err_sne = err_eff[sne_mask]
        ll_sne = -0.5 * np.log(2 * np.pi * err_sne**2) - 0.5 * (res_sne / err_sne)**2
    
    # --- QSO Log-Likelihood ---
    res_qso = mu_obs[qso_mask] - mu_th[qso_mask]
    err_qso = err_eff[qso_mask]
    ll_qso = -0.5 * np.log(2 * np.pi * err_qso**2) - 0.5 * (res_qso / err_qso)**2
    
    # --- CC Log-Likelihood ---
    if len(z_cc) > 0:
        res_cc = H_obs - _hz_bg
        ll_cc = -0.5 * np.log(2 * np.pi * err_cc**2) - 0.5 * (res_cc / err_cc)**2
    else:
        ll_cc = np.array([])
        
    # --- CMB Log-Likelihood ---
    # CMB acts as a single data point
    if gv['USE_CMB']:
        import cosmo_constants as cc_const
        compute_Omega_m = gv['compute_Omega_m']
        R_mod = np.sqrt(compute_Omega_m(H0, omch2)) * (H0 / cc_const.C_LIGHT_KMS) * ((1 + cc_const.Z_STAR) * _da_bg)
        pen = -0.5 * ((R_mod - cc_const.R_PLANCK) / cc_const.SIGMA_R_PLANCK) ** 2
        
        if model_name == "log_decay":
            dmu_star = A * np.exp(-cc_const.Z_STAR / zb) + g0 * np.log1p(cc_const.Z_STAR)**2 * np.exp(-cc_const.Z_STAR / zh)
            pen += -0.5 * (dmu_star / cc_const.SIGMA_CORR_CMB) ** 2
            
        ll_cmb = np.array([pen])
    else:
        ll_cmb = np.array([])
        
    return np.concatenate([ll_sne, ll_qso, ll_cc, ll_cmb])

# ==============================================================================
# 3. BUILD ARVIZ INFERENCE DATA
# ==============================================================================
def process_chain(model_key, model_name):
    print(f"\nProcessing {model_key}...")
    try:
        samples = loadMCSamples(f"chains/mcmc_{model_key}", settings={'ignore_rows': 0.3})
    except Exception as e:
        print(f"Skipping {model_key}: could not load chains. ({e})")
        return None
        
    pnames = samples.getParamNames().list()
    # Ensure parameter names match internal
    if 'gamma' in pnames:
        pnames = [p if p != 'gamma' else 'gamma_log_decay' for p in pnames]
        
    posterior_dict = {}
    n_samples = min(1000, samples.samples.shape[0]) # Thin to max 1000 for speed
    
    # Take evenly spaced samples
    idx = np.linspace(0, samples.samples.shape[0]-1, n_samples).astype(int)
    
    for i, p in enumerate(samples.getParamNames().list()):
        internal_p = p if p != 'gamma' else 'gamma_log_decay'
        posterior_dict[internal_p] = np.expand_dims(samples.samples[idx, i], axis=0) # shape (1, draws)
        
    log_likelihoods = []
    print(f"Evaluating {n_samples} samples pointwise logL for WAIC...")
    for i in range(n_samples):
        params = {p: posterior_dict[p][0, i] for p in posterior_dict.keys()}
        ll = pointwise_logL(params, model_name)
        log_likelihoods.append(ll)
        if (i+1) % 250 == 0: print(f"  {i+1}/{n_samples} done.")
        
    log_likelihoods = np.array(log_likelihoods) # shape (draws, points)
    log_likelihoods = np.expand_dims(log_likelihoods, axis=0) # shape (1, draws, points)
    
    idata = az.from_dict(
        posterior=posterior_dict,
        log_likelihood={"obs": log_likelihoods}
    )
    return idata

print("\nBuilding ArviZ traces...")
idata_lcdm = process_chain("lcdm", "lcdm")
idata_log_decay = process_chain("log_decay", "log_decay")

# ==============================================================================
# 4. LOO & WAIC COMPARISON
# ==============================================================================
models = {}
if idata_lcdm is not None: models["ΛCDM"] = idata_lcdm
if idata_log_decay is not None: models["γCDM-LOG²-Decay"] = idata_log_decay

if len(models) < 2:
    print("\nNeed at least two models to compare.")
else:
    print("\n" + "="*80)
    print("📈 ARVIZ WAIC / LOO-PSIS COMPARISON (Properly Decorrelated Covariance)")
    print("="*80)

    print("\n--- Leave-One-Out (LOO) Cross Validation ---")
    comp_loo = az.compare(models, ic="loo", scale="deviance")
    print(comp_loo)

    print("\n--- WAIC (Watanabe-Akaike Information Criterion) ---")
    comp_waic = az.compare(models, ic="waic", scale="deviance")
    print(comp_waic)

    if C_inv_sne is not None:
        print("\nNote on SNe Ia Covariance:")
        print("Because SNe are highly correlated, standard pointwise WAIC/LOO is mathematically invalid.")
        print("This script first uncorrelated the SNe residuals using the Cholesky decomposition of")
        print("the Pantheon+ STAT+SYS covariance matrix (v = L^{-1} r). The WAIC/LOO values are calculated")
        print("on these independent transformed variables, making the comparison covariance-aware and internally consistent.")
    else:
        print("\nNote on SNe Ia Covariance:")
        print("MCMC was run WITHOUT covariance matrix (--cov none).")
        print("Standard independent pointwise WAIC/LOO has been calculated.")
    print("="*80)

    # ==========================================================================
    # 5. PARETO-k DIAGNOSTIC
    # ==========================================================================
    # Build point-type labels: the pointwise logL is concatenated as
    #   [SNe(n_sne), QSO(n_qso), CC(n_cc), CMB(0 or 1)]
    n_sne_pts = int(sne_mask.sum())
    n_qso_pts = int(qso_mask.sum())
    n_cc_pts  = len(z_cc)
    n_cmb_pts = 1 if gv['USE_CMB'] else 0

    # Redshift array aligned with the pointwise axis
    z_sne_arr = z_mu[sne_mask]
    z_qso_arr = z_mu[qso_mask]
    z_all_pts = np.concatenate([
        z_sne_arr, z_qso_arr, z_cc,
        np.array([1089.0]) if n_cmb_pts else np.array([])
    ])

    probe_labels = (
        ["SNe"] * n_sne_pts +
        ["QSO"] * n_qso_pts +
        ["CC"]  * n_cc_pts  +
        ["CMB"] * n_cmb_pts
    )
    probe_labels = np.array(probe_labels)

    print("\n" + "=" * 80)
    print("🔬 PARETO-k DIAGNOSTIC (LOO-PSIS reliability)")
    print("=" * 80)

    for model_name, idata in models.items():
        print(f"\n┌── {model_name} ──")
        loo_result = az.loo(idata, pointwise=True)
        khat = loo_result.pareto_k.values

        n_total = len(khat)
        n_good    = int(np.sum(khat <= 0.5))
        n_ok      = int(np.sum((khat > 0.5) & (khat <= 0.7)))
        n_bad     = int(np.sum((khat > 0.7) & (khat <= 1.0)))
        n_vbad    = int(np.sum(khat > 1.0))

        print(f"│  Total points:       {n_total}")
        print(f"│  k̂ ≤ 0.5  (good):    {n_good}  ({100*n_good/n_total:.1f}%)")
        print(f"│  0.5 < k̂ ≤ 0.7 (ok): {n_ok}  ({100*n_ok/n_total:.1f}%)")
        print(f"│  0.7 < k̂ ≤ 1.0 (bad):{n_bad}  ({100*n_bad/n_total:.1f}%)")
        print(f"│  k̂ > 1.0  (v.bad):   {n_vbad}  ({100*n_vbad/n_total:.1f}%)")

        # Identify the problematic points (k > 0.7)
        bad_mask = khat > 0.7
        if np.any(bad_mask):
            bad_idx = np.where(bad_mask)[0]
            bad_k = khat[bad_mask]
            bad_probes = probe_labels[bad_idx]
            bad_z = z_all_pts[bad_idx]

            # Count by probe
            from collections import Counter
            probe_counts = Counter(bad_probes)
            print(f"│")
            print(f"│  Breakdown of k̂ > 0.7 by probe:")
            for ptype in ["SNe", "QSO", "CC", "CMB"]:
                if ptype in probe_counts:
                    print(f"│    {ptype}: {probe_counts[ptype]} points")

            # Show top-10 worst offenders
            worst_order = np.argsort(-bad_k)[:10]
            print(f"│")
            print(f"│  Top offenders (up to 10):")
            print(f"│  {'idx':>6}  {'k̂':>6}  {'probe':>5}  {'z':>8}")
            print(f"│  {'─'*30}")
            for wi in worst_order:
                print(f"│  {bad_idx[wi]:>6}  {bad_k[wi]:>6.3f}  {bad_probes[wi]:>5}  {bad_z[wi]:>8.4f}")
        else:
            print(f"│  ✅ All points have k̂ ≤ 0.7 — LOO is fully reliable!")

        print(f"└{'─'*40}")

    print("\n" + "=" * 80)
    print("Interpretation:")
    print("  k̂ ≤ 0.5  → excellent, no concern")
    print("  0.5–0.7  → marginal, usually acceptable")
    print("  0.7–1.0  → unreliable for that point, check if QSO outliers")
    print("  > 1.0    → PSIS importance weights fail, consider moment matching")
    print("  A few QSO outliers at high-z with k̂ > 0.7 are expected and benign.")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 5. BEAUTIFUL ARVIZ PLOTS
    # -------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import os
        
        # Premium Light Theme setup (ArviZ documentation style)
        az.style.use("arviz-doc")
        plt.rcParams.update({
            'figure.facecolor': '#ffffff',
            'axes.facecolor': '#ffffff',
            'axes.edgecolor': '#e0e0e0',
            'axes.grid': True,
            'grid.color': '#e0e0e0',
            'grid.linestyle': '-',
            'grid.alpha': 0.8,
            'text.color': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#555555',
            'ytick.color': '#555555',
            'font.family': 'sans-serif'
        })

        os.makedirs("plots", exist_ok=True)
        
        # 1. LOO Comparison Plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        
        # ArviZ's built-in compare plot with default elegant colors
        az.plot_compare(
            comp_loo, 
            ax=ax, 
            plot_ic_diff=True,  # Shows the difference directly
            insample_dev=True,  # Shows in-sample deviance (open circles)
            textsize=14
        )
        
        # Beautify titles and labels
        ax.set_title("LOO-PSIS Model Comparison (Deviance Scale)\nLower is better", 
                    fontsize=16, fontweight='bold', pad=15, color='#111111')
        ax.set_xlabel("Deviance (-2 × Expected Log Predictive Density)", fontsize=14, labelpad=10)
        
        # Re-draw the y-axis labels to use mathematical font if possible
        labels = [item.get_text() for item in ax.get_yticklabels()]
        ax.set_yticklabels(labels, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_path = "plots/waic_loo_comparison.png"
        plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        
        print(f"\n🎨 ✓ ArviZ plot generated successfully: {plot_path}")
        print("   (Open circle = in-sample deviance; Closed circle = LOO deviance; Lines = Standard Error)")
        
        # 2. Forest Plot: Parameter comparison (omch2)
        print("\nGenerando Forest Plot (comparativa de parámetros)...")
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        az.plot_forest(
            list(models.values()),
            model_names=list(models.keys()),
            var_names=["omch2"],
            combined=True,
            colors=['#1f77b4', '#ff7f0e'], # Blue for first, Orange for second
            ax=ax
        )
        ax.set_title("$\Omega_c h^2$ Comparison (94% HDI)", fontsize=16, pad=15)
        plt.tight_layout()
        forest_path = "plots/arviz_forest_omch2.png"
        plt.savefig(forest_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        # 3. Posterior Plot: Extra parameters of γCDM-LOG²-Decay
        print("Generando Posterior KDE Plots para γCDM...")
        if "γCDM-LOG²-Decay" in models:
            idata_gcdm = models["γCDM-LOG²-Decay"]
            # Extract variables that actually exist in the posterior
            var_names = [v for v in ["gamma_log_decay", "A", "zh", "zb"] if v in idata_gcdm.posterior.data_vars]
            if var_names:
                axes = az.plot_posterior(
                    idata_gcdm, 
                    var_names=var_names,
                    hdi_prob=0.95,
                    point_estimate='mode',
                    kind='kde',
                    figsize=(12, 4),
                    textsize=12
                )
                fig = axes.flatten()[0].figure if isinstance(axes, np.ndarray) else axes.figure
                fig.suptitle("Posterior Distributions of Physical Parameters (γCDM)", fontsize=16, y=1.05)
                posterior_path = "plots/arviz_posterior_gcdm.png"
                plt.savefig(posterior_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
                plt.close(fig)
                
        # 4. Trace Plot (solo visual, útil para ver mixing y KDE lado a lado)
        print("Generando Trace Plots...")
        if "γCDM-LOG²-Decay" in models:
            axes = az.plot_trace(
                models["γCDM-LOG²-Decay"], 
                var_names=["omch2"] + var_names[:2], # Limit to 3 vars to avoid huge plot
                compact=True,
                figsize=(12, 6)
            )
            fig = axes.flatten()[0].figure if isinstance(axes, np.ndarray) else axes.figure
            fig.suptitle("KDE & MCMC Chains (Trace Plot)", fontsize=16, y=1.05)
            trace_path = "plots/arviz_trace_gcdm.png"
            plt.savefig(trace_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
            
        print("🎨 ✓ Todos los plots adicionales de ArviZ generados en la carpeta plots/")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate ArviZ plot: {e}")
