# Topological Horizon Lensing Framework (γCDM)

**Preprint:** [Topological Horizon Lensing: γCDM Correction to Luminosity Distances](https://doi.org/10.5281/zenodo.18401999)  
**Author:** Bautista, 2026

This repository contains the **reproduction code** and dataset required to independently verify the numerical results, tables, phenomenological conclusions and methodology presented in the associated preprint.

## Overview
The results reported in the paper were originally obtained using a custom research framework implemented in JAX, incorporating machine learning–assisted exploration tools (internally referred to as the Akashic Alpha Engine). That framework was used to explore patterns, test phenomenological hypotheses, and guide physical intuition.

To ensure full transparency, accessibility, and reproducibility, this repository distills the final validated analysis into a standalone and dependency-light reproduction package, independent of the original research engine.

The goal of this repository is verification, not methodological novelty.

---

## LOG²-Decay Unification (Feb 23, 2026)

The final model combines two independent components—a local bubble and a persistent geometric correction:

$$\Delta\mu(z) = A \cdot e^{-z/z_b} + \gamma_0 \cdot [\ln(1+z)]^2 \cdot e^{-z/z_h}$$

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| A | −0.175 | Local bubble amplitude (mag) |
| z_b | 0.40 | Bubble decay scale |
| γ₀ | −0.807 | Kerr geometry coefficient |
| z_h | 41.8 | Horizon damping scale |
| H₀ | 67.4 (fixed) | Planck anchor |
| Ωm | 0.338 | Matter density |

**Key result:** H₀(local) = 67.4 × 10^(0.035) = **73.05 km/s/Mpc** — matching SH0ES to 0.1%.

### Model Evolution

| Gen. | Model | k | Dataset | ΔBIC | ΔAIC | Verdict |
|------|-------|---|---------|------|------|---------|
| I | LOG | 3 | SNe+CC | −6.2 | −8.2 | First signal |
| II | LOG² | 4 | SNe+QSO+CC | −739 | −745 | Dominant, but CMB-incompatible |
| II | LINEAR | 4 | SNe+QSO+CC | −678 | −685 | Strong, but weaker than LOG² |
| II | LOG³ | 4 | SNe+QSO+CC | −696 | −702 | Not cubic—truly quadratic |
| III | Decay | 5 | SNe+QSO+CC | −540 | −553 | Alleviated H₀, no structure |
| III | **LOG²-Decay** | **7** | **SNe+QSO+CC** | **−735** | **−760** | **Full alleviation** |

<p align="center">
  <img width="48%" alt="mcmc_log_decay_full_corner" src="https://github.com/user-attachments/assets/0f4a2469-79e4-4ddd-9652-361313c0d1f1" />
  <img width="48%" alt="nested_log_decay_full_corner" src="https://github.com/user-attachments/assets/3efeb680-8d23-4abe-927e-a7a9a0de0d67" />
</p>

<p align="center">
  <em><b>Left:</b> Cobaya MCMC. <b>Right:</b> Cobaya Nested Sampling (PolyChord). </em>
</p>

---

## Addendum Update (Feb 10, 2026)
The analysis was refined by incorporating 2,397 high-redshift quasars (z ≤ 7.08), revealing that the lensing effect scales quadratically with the logarithmic expansion: **Δμ = γ₀ · [ln(1+z)]²**. This refined model (**γCDM-LOG²**) achieves a **ΔBIC ≈ −739** against ΛCDM and alleviates the Hubble tension with **H₀ = 73.4 ± 2.1 km/s/Mpc (MCMC)**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea20f332-56c4-4c30-b468-9ec6072d0aa7" width="46%" />
  <img src="https://github.com/user-attachments/assets/cf3a529a-5d68-4bff-a789-cfe6ea50eec1" width="50%" />
</p>

<p align="center">
  <em><b>Left:</b> Nested posterior distributions for H₀, Ωch², and calibration offsets (δM). <b>Right:</b> Hubble tension alleviation (γCDM-LOG² vs ΛCDM). </em>
</p>

---

## Hardened Robustness Protocol
To ensure the statistical validity of the results, a validation suite is provided to test the model under a strict anti-cheat protocol:
*   **Physical Priors:** Enforces **Ωm < 1** (flat ΛCDM with ΩΛ ≥ 0).
*   **Calibration Neutrality:** Independent nuisance parameters (**δM**) for SNe and Quasars with identical wide priors.
*   **Intrinsic Scatter:** Incorporates conservative scatter for Quasars (**σ_int = 0.40 mag**) in the likelihood normalization.
*   **Mock Pipeline Test:** Verifies the pipeline does not fabricate spurious signals (signal separation: **11.1σ** from null hypothesis).

---

## Contents
*   `gammacdm_preprint_repro.ipynb`: Reproduction notebook for the original linear-log analysis (Gen I).
*   `gammacdm_addendum_verification.py`: Main validation suite. Implements multi-start MLE, MCMC (Cobaya), Nested Sampling (PolyChord), and Mock Tests for all model generations.
*   `run_nested_single.py`: Runner for isolated Nested Sampling processes (requires PolyChord).
*   `full_dataset.csv`: The curated dataset (SNe Ia + Quasars + Cosmic Chronometers).

---

## Scope and Interpretation
The analysis implemented here is phenomenological. It is designed to test relative consistency and probe separation between luminosity-based (null geodesic) and clock-based (timelike geodesic) observables under a common correction term.

The reproduction package does not implement the full official likelihood pipelines (e.g. full Pantheon+ covariance matrices), and therefore the results should be interpreted as diagnostic and exploratory, not as definitive cosmological parameter constraints.

---

## Usage

### Canonical Reproduction (LOG²-Decay — Preprint values)
```bash
pip install camb pandas numpy scipy matplotlib cobaya

python gammacdm_addendum_verification.py \
    --revised --legacy --fixed-anchor --mock --nested
```

### Original Preprint (Gen I)
```bash
jupyter notebook gammacdm_preprint_repro.ipynb
```

---

## Acknowledgements

This work makes use of the following open-source scientific software:

- **Cobaya** — Bayesian Analysis in Cosmology  
  https://cobaya.readthedocs.io/

- **PolyChord** — Nested Sampling algorithm  
  https://github.com/PolyChord/PolyChordLite

- **CAMB** — Code for Anisotropies in the Microwave Background  
  https://camb.info/

- **SciPy** — Scientific computing in Python  
  https://scipy.org/

- **PySR** — Symbolic Regression for scientific discovery  
  https://github.com/MilesCranmer/PySR

I am grateful to the authors and maintainers of these tools and others used in this project..

As this work has been developed outside a formal academic setting, the present acknowledgements are intended as a good-faith attribution of the main software dependencies used during its development. A more complete and conventional set of citations and acknowledgements will be provided in any future refined or peer-reviewed version.
