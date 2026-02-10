# Reproduction Package: Topological Horizon Lensing (γCDM)

**Preprint:** [Topological Horizon Lensing and the Hubble Tension](https://doi.org/10.5281/zenodo.18401999)  
**Addendum:** [Quadratic Evolution Alleviation of the Hubble Tension](https://doi.org/10.5281/zenodo.18401999)  
**Author:** Bautista, 2026

This repository contains the **reproduction code** and dataset required to independently verify the numerical results, tables, and phenomenological conclusions presented in the associated preprint and the addendum.

## Overview
The results reported in the paper were originally obtained using a custom research framework implemented in JAX, incorporating machine learning–assisted exploration tools (internally referred to as the Akashic Alpha Engine). That framework was used to explore patterns, test phenomenological hypotheses, and guide physical intuition.

## Addendum Update (Feb 2026)
The analysis has been refined by incorporating 2,397 high-redshift quasars (z ≤ 7.08), revealing that the lensing effect scales quadratically with the logarithmic expansion: **Δμ = γ₀ · [ln(1+z)]²**. This refined model (**γCDM-LOG²**) achieves a **ΔBIC ≈ −699** against ΛCDM and alleviates the Hubble tension with **H₀ = 73.4 ± 2.1 km/s/Mpc (MCMC)**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea20f332-56c4-4c30-b468-9ec6072d0aa7" width="46%" />
  <img src="https://github.com/user-attachments/assets/cf3a529a-5d68-4bff-a789-cfe6ea50eec1" width="50%" />
</p>

<p align="center">
  <em><b>Left:</b> Nested posterior distributions for H₀, Ωch², and calibration offsets (δM). <b>Right:</b> Hubble tension alleviation (γCDM-LOG² vs ΛCDM). </em>
</p>

## Hardened Robustness Protocol
To ensure the statistical validity of the results, a validation suite is provided to test the model under a strict anti-cheat protocol:
*   **Physical Priors:** Enforces **Ωm < 1** (flat ΛCDM with ΩΛ ≥ 0).
*   **Calibration Neutrality:** Independent nuisance parameters (**δM**) for SNe and Quasars with identical wide priors.
*   **Intrinsic Scatter:** Incorporates conservative scatter for Quasars (**σ_int = 0.40 mag**) in the likelihood normalization.
*   **Mock Pipeline Test:** Verifies the pipeline does not fabricate spurious signals (signal separation: **11.1σ** from null hypothesis).

To ensure full transparency, accessibility, and reproducibility, this repository distills the final validated analysis into a standalone and dependency-light reproduction package, independent of the original research engine.

The goal of this repository is verification, not methodological novelty.

## Contents
*   `gammacdm_preprint_repro.ipynb`: The main reproduction notebook for the original linear-log analysis (v5).
*   `gammacdm_addendum_verification.py`: The main validation suite for the Addendum (v6). Implements multi-start MLE, MCMC, and Mock Tests.
*   `run_nested_single.py`: Runner for isolated Nested Sampling processes (requires PolyChord).
*   `full_dataset.csv`: The curated dataset used for the analysis (SNe Ia + Quasars + Cosmic Chronometers).

## Scope and Interpretation
The analysis implemented here is phenomenological. It is designed to test relative consistency and probe separation between luminosity-based (null geodesic) and clock-based (timelike geodesic) observables under a common correction term.

The reproduction package does not implement the full official likelihood pipelines (e.g. full Pantheon+ covariance matrices), and therefore the results should be interpreted as diagnostic and exploratory, not as definitive cosmological parameter constraints.

## Usage (Original Preprint)
The notebook can be executed locally or on Google Colab to reproduce all reported tables and derived quantities.

```bash
pip install camb pandas numpy scipy matplotlib cobaya
jupyter notebook gammacdm_preprint_repro.ipynb
```

## Usage (Addendum & Robustness)
Verification of the quadratic model and the hardened robustness protocol:

```bash
python gammacdm_addendum_verification.py --starts 5 --mcmc --mock --sigma-int-sne 0.1 --sigma-int-qso 0.4 --qso-err-cut 10.0
```
**Note on Nested Sampling:** The suite supports **PolyChord** for absolute Bayesian Evidence (log Z) calculations. Due to local computational constraints, full Nested Sampling convergence for the 4,007-point dataset was not performed by the author; however, the massive ΔBIC ≈ −699 and MCMC posteriors provide decisive evidence for the model preference.

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

I am grateful to the authors and maintainers of these tools.

As this work has been developed outside a formal academic setting, the present acknowledgements are intended as a good-faith attribution of the main software dependencies used during its development. A more complete and conventional set of citations and acknowledgements will be provided in any future refined or peer-reviewed version.
