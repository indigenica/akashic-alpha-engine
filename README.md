# Reproduction Package: Topological Horizon Lensing (γCDM)

**Preprint:** [Topological Horizon Lensing and the Hubble Tension](https://doi.org/10.5281/zenodo.18401999)  
**Author:** Bautista, 2026

This repository contains the **reproduction code** and dataset required to independently verify the numerical results, tables, and phenomenological conclusions presented in the associated preprint.

## Overview

The results reported in the paper were originally obtained using a custom research framework implemented in JAX, incorporating machine learning–assisted exploration tools (internally referred to as the Akashic Alpha Engine). That framework was used to explore patterns, test phenomenological hypotheses, and guide physical intuition.

To ensure full transparency, accessibility, and reproducibility, this repository distills the final validated analysis into a standalone and dependency-light reproduction package, independent of the original research engine.

The goal of this repository is verification, not methodological novelty.

## Contents

- **`gammacdm_preprint_repro.ipynb`**: The main reproduction notebook. It performs the Maximum Likelihood Estimation (MLE), Bifurcation / null–timelike probe separation test, and internal consistency checks.
- **`full_dataset.csv`**: The curated dataset used for the analysis (SNe Ia + Cosmic Chronometers), identical to the one used to generate the results reported in the paper.

## Scope and Interpretation

The analysis implemented here is **phenomenological**.
It is designed to test **relative consistency and probe separation** between luminosity-based (null geodesic) and clock-based (timelike geodesic) observables under a common correction term.

The reproduction package **does not implement the full official likelihood pipelines** (e.g. full Pantheon+ covariance matrices), and therefore the results should be interpreted as **diagnostic and exploratory**, not as definitive cosmological parameter constraints.

## Usage

The notebook can be executed locally or on Google Colab to reproduce all reported tables and derived quantities.

```bash
pip install camb pandas numpy scipy matplotlib cobaya
jupyter notebook gammacdm_preprint_repro.ipynb
```
