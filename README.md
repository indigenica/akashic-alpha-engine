# Reproduction Package: Topological Horizon Lensing (γCDM)

**Preprint:** [Topological Horizon Lensing and the Hubble Tension](https://doi.org/10.5281/zenodo.18401999)  
**Author:** Bautista, 2026

This repository contains the **reproduction code** necessary to verify the final values and tables presented in the preprint.

## Overview

The findings presented in the paper were derived using a custom research engine built with **JAX** and **Deep Learning** techniques (Akashic Alpha Engine), which was used to detect patterns, test phenomenological hypotheses, and refine physical intuitions.

To ensure transparency and reproducibility without requiring the complex dependencies of the full research engine, this package distills the final analysis into a standalone, verifiable format.

## Contents

- **`gammacdm_preprint_repro.ipynb`**: The main reproduction notebook. It performs the Maximum Likelihood Estimation (MLE), Bifurcation Test, and Consistency Checks exactly as described in the preprint.
- **`full_dataset.csv`**: The curated dataset used for the analysis (SNe Ia + Cosmic Chronometers), identical to the one used in the paper.

## Usage

You can run the notebook locally or on Google Colab to verify the results (Tables and derived quantities).

```bash
pip install camb pandas numpy scipy matplotlib cobaya
jupyter notebook gammacdm_preprint_repro.ipynb
```
