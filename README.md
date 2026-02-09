# Reproduction Package: Topological Horizon Lensing (γCDM)

**Preprint:** [Topological Horizon Lensing and the Hubble Tension](https://doi.org/10.5281/zenodo.18401999)  
**Addendum:** [Quadratic Evolution Alleviation of the Hubble Tension](https://doi.org/10.5281/zenodo.18401999)  
**Author:** Bautista, 2026

This repository contains the **reproduction code** and dataset required to independently verify the numerical results, tables, and phenomenological conclusions presented in the associated preprint and the addendum.

## Overview
The results reported in the paper were originally obtained using a custom research framework implemented in JAX, incorporating machine learning–assisted exploration tools (internally referred to as the Akashic Alpha Engine). That framework was used to explore patterns, test phenomenological hypotheses, and guide physical intuition.

## Addendum Update (Feb 2026)
The analysis has been refined by incorporating 2,397 high-redshift quasars ($z \leq 7.08$), revealing that the lensing effect scales quadratically with the logarithmic expansion: $\Delta\mu = \gamma_0 \cdot [\ln(1+z)]^2$. This refined model ($\gamma$CDM-LOG$^2$) achieves a $\Delta$BIC $\approx -699$ against $\Lambda$CDM and alleviates the Hubble tension with $H_0 = 73.4 \pm 2.1$ km/s/Mpc (MCMC).

## Hardened Robustness Protocol
To ensure the statistical validity of the results, a validation suite is provided to test the model under a strict verification protocol:
*   **Physical Priors:** Enforces $\Omega_m < 1$ (flat $\Lambda$CDM with $\Omega_\Lambda \geq 0$).
*   **Calibration Neutrality:** Independent nuisance parameters ($\delta M$) for SNe and Quasars with identical wide priors.
*   **Intrinsic Scatter:** Incorporates conservative scatter for Quasars ($\sigma_{int} = 0.40$ mag) in the likelihood normalization.
*   **Mock Pipeline Test:** Verifies the pipeline does not fabricate spurious signals (signal separation: $11.1\sigma$ from null hypothesis).

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

**Note on Nested Sampling:** The suite supports **PolyChord** for absolute Bayesian Evidence ($\log Z$) calculations. Due to local computational constraints, full Nested Sampling convergence for the 4,007-point dataset was not performed by the author; however, the massive $\Delta$BIC ($\approx -699$) and MCMC posteriors provide decisive evidence for the model preference.
