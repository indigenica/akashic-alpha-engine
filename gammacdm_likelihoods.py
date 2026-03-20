#!/usr/bin/env python3
"""
γCDM Cobaya Likelihood Module
==============================
Single source of truth for all model likelihoods used in both
MCMC (gammacdm_verification.py) and Nested Sampling (run_nested_single.py).

Models implemented:
  1. ΛCDM                — Standard model (no correction)
  2. γCDM-LOG²           — Δμ = γ₀·[ln(1+z)]²
  3. γCDM-Decay          — Δμ = A·exp(-z/zd)
  4. γCDM-LOG²-Decay     — Δμ = A·exp(-z/zb) + γ₀·[ln(1+z)]²·exp(-z/zh)

Log-likelihood (default Gaussian):
  -2 ln L = Σ [(μ_obs - μ_th)² / σ_eff²]  +  Σ [ln(σ_eff²)]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^
            chi-squared term                  normalization term

  σ_eff² = σ_obs² + σ_int²  (intrinsic scatter added in quadrature)

Robust alternatives (--student / --cauchy):
  Student-t(ν):  ln L_i = −ln(σ) − (ν+1)/2 · ln(1 + r²/(ν·σ²))
  Cauchy:        Student-t with ν = 1 (heaviest tails, maximum robustness)

References:
  - Pantheon+: Brout et al. (2022), arXiv:2202.04077
  - Quasar Hubble diagram: Lusso et al. (2020), arXiv:2008.08586
  - γCDM framework: [this work]
"""

import numpy as np


def create_likelihoods(z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc,
                       sne_mask, combined_mode,
                       sigma_int_sne=0.0, sigma_int_qso=0.0,
                       no_nuisance=False, asymmetric=False, sanity_check=False,
                       likelihood_type='gaussian', nu=5.0,
                       C_inv_sne=None, ln_det_C_sne=None):
    """
    Factory that creates 4 configured Cobaya Likelihood classes.

    Parameters
    ----------
    z_mu : np.ndarray
        Redshifts for distance modulus data (SNe Ia + Quasars).
    mu_obs : np.ndarray
        Observed distance moduli.
    err_mu : np.ndarray
        Observational errors on μ.
    z_cc : np.ndarray
        Redshifts for Cosmic Chronometers H(z) data.
    H_obs : np.ndarray
        Observed H(z) values.
    err_cc : np.ndarray
        Errors on H(z).
    sne_mask : np.ndarray (bool)
        True for SNe Ia entries, False for Quasars.
    combined_mode : bool
        If True, use separate M_sne / M_qso offsets.
        If False, use single delta_M offset.
    sigma_int_sne : float
        Intrinsic scatter for SNe Ia (mag). Default 0.
    sigma_int_qso : float
        Intrinsic scatter for Quasars (mag). Default 0.
    no_nuisance : bool
        If True, fix M=0 for γCDM/Decay models (ΛCDM keeps M free).
    asymmetric : bool
        If True, remove M entirely from γCDM/Decay (ΛCDM keeps M free).
    sanity_check : bool
        If True, H0 fixed, M removed from γCDM/Decay.
    likelihood_type : str
        'gaussian' (default), 'student', or 'cauchy'.
    nu : float
        Degrees of freedom for Student-t (default 5.0, ignored for Gaussian/Cauchy).
    C_inv_sne : np.ndarray or None
        Inverse of the SNe Ia covariance matrix (already includes σ_int on diagonal).
        When provided, SNe are evaluated with the full correlated Gaussian likelihood.
    ln_det_C_sne : float or None
        log-determinant of the SNe Ia covariance matrix.

    Returns
    -------
    tuple of (LCDMLikelihood, GammaCDM_LOG2_Likelihood,
              DecayLikelihood, GammaCDM_LOG_DECAY_Likelihood)
    """
    from cobaya.likelihood import Likelihood

    # ── Precompute effective errors (σ_obs ⊕ σ_int) ──
    # In combined mode, SNe and QSOs get their respective σ_int.
    # In non-combined mode (--no-quasars or --quasars-only), we infer the
    # appropriate scatter: if there is a sne_mask and it has any True entries
    # we use sigma_int_sne; otherwise we use sigma_int_qso.
    if combined_mode:
        sigma_int = np.where(sne_mask, sigma_int_sne, sigma_int_qso)
    else:
        has_sne = sne_mask is not None and np.any(sne_mask)
        sigma_int_val = sigma_int_sne if has_sne else sigma_int_qso
        sigma_int = np.full_like(err_mu, sigma_int_val)
    err_eff = np.sqrt(err_mu**2 + sigma_int**2)

    norm_cc = np.sum(np.log(err_cc**2)) if len(z_cc) > 0 else 0.0

    # Should this model have its M zeroed out?
    restrict_m = no_nuisance or asymmetric or sanity_check

    # ── Covariance bookkeeping ──
    _use_cov = C_inv_sne is not None and ln_det_C_sne is not None
    _qso_mask = ~sne_mask if (combined_mode and sne_mask is not None) else None

    # Resolve effective ν for Student-t / Cauchy
    _ltype = likelihood_type.lower()
    if _ltype == 'cauchy':
        _nu_eff = 1.0
        _ltype = 'student'   # Cauchy = Student-t(ν=1)
    elif _ltype == 'student':
        _nu_eff = float(nu)
    else:
        _nu_eff = None        # Gaussian — unused

    # ================================================================
    # Helper: log-likelihood for μ residuals (robust or Gaussian)
    # ================================================================
    #   residuals = μ_obs − μ_th   (array, full length)
    #   sigma     = σ_eff           (array, same shape)
    #
    # When C_inv_sne is provided, SNe Ia are evaluated with the full
    # correlated Gaussian: R_sne^T C^{-1} R_sne + ln|C|.
    # Remaining points (QSOs, or all if no cov) use the diagonal evaluator.
    # CC H(z) data keeps plain Gaussian — no outlier problem there.
    # ================================================================
    def _logL_mu(residuals, sigma):
        """Compute log-likelihood for distance-modulus data."""
        logL = 0.0

        if _use_cov:
            if combined_mode and sne_mask is not None:
                res_sne = residuals[sne_mask]
                res_diag = residuals[_qso_mask]
                sig_diag = sigma[_qso_mask]
            else:
                res_sne = residuals
                res_diag = np.array([])
                sig_diag = np.array([])

            logL += -0.5 * (res_sne @ C_inv_sne @ res_sne + ln_det_C_sne)
        else:
            res_diag = residuals
            sig_diag = sigma

        if len(res_diag) > 0:
            if _ltype == 'student':
                r2 = res_diag**2
                s2 = sig_diag**2
                logL += np.sum(-np.log(sig_diag)
                               - (_nu_eff + 1) / 2 * np.log1p(r2 / (_nu_eff * s2 + 1e-30)))
            else:
                logL += -0.5 * (np.sum((res_diag / sig_diag)**2)
                                + np.sum(np.log(sig_diag**2)))

        return logL


    # ================================================================
    # 1. ΛCDM  —  No model correction
    # ================================================================
    if combined_mode:
        lcdm_params = {'M_sne': None, 'M_qso': None}
    else:
        lcdm_params = {'mabs': None}

    class LCDMLikelihood(Likelihood):
        """
        Standard ΛCDM: μ_th = 5·log10(DL) + 25.
        ΛCDM always retains calibration freedom (M free) for fair comparison.
        """
        params = lcdm_params

        def initialize(self):
            self.z_mu = z_mu
            self.mu_obs = mu_obs
            self.err_eff = err_eff
            self.sne_mask = sne_mask
            self.combined = combined_mode
            self.z_cc = z_cc
            self.H_obs = H_obs
            self.err_cc = err_cc
            self.norm_cc = norm_cc

        def get_requirements(self):
            reqs = {"angular_diameter_distance": {"z": np.concatenate([self.z_mu, self.z_cc])}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            return reqs

        def logp(self, **pv):
            try:
                da = self.provider.get_angular_diameter_distance(self.z_mu)
                dl = da * (1 + self.z_mu)**2
            except Exception:
                return -1e30

            mu_th = 5 * np.log10(np.maximum(dl, 1e-10)) + 25

            # ΛCDM ALWAYS gets calibration offset (not restricted by flags)
            if self.combined:
                mu_th = mu_th + np.where(self.sne_mask,
                                         pv.get('M_sne', 0.0),
                                         pv.get('M_qso', 0.0))
            else:
                mu_th = mu_th + pv.get('mabs', 0.0)

            # Log-likelihood (may be Gaussian, Student-t, or Cauchy)
            logL = _logL_mu(self.mu_obs - mu_th, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            return logL

    # ================================================================
    # 2. γCDM-LOG²  —  Δμ = γ₀·[ln(1+z)]²
    # ================================================================
    if combined_mode:
        log2_params = {'gamma_log2': None, 'M_sne': None, 'M_qso': None}
    else:
        log2_params = {'gamma_log2': None, 'mabs': None}

    class GammaCDM_LOG2_Likelihood(Likelihood):
        """
        γCDM-LOG²: Distance modulus correction Δμ = γ₀·[ln(1+z)]².
        Phenomenological logarithmic correction to luminosity distances.
        """
        params = log2_params

        def initialize(self):
            self.z_mu = z_mu
            self.mu_obs = mu_obs
            self.err_eff = err_eff
            self.sne_mask = sne_mask
            self.combined = combined_mode
            self.z_cc = z_cc
            self.H_obs = H_obs
            self.err_cc = err_cc
            self.norm_cc = norm_cc
            self.restrict_m = restrict_m

        def get_requirements(self):
            reqs = {"angular_diameter_distance": {"z": np.concatenate([self.z_mu, self.z_cc])}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            return reqs

        def logp(self, **pv):
            try:
                da = self.provider.get_angular_diameter_distance(self.z_mu)
                dl = da * (1 + self.z_mu)**2
            except Exception:
                return -1e30

            gamma0 = pv.get('gamma_log2', 0.0)
            corr = gamma0 * np.log1p(self.z_mu)**2

            mu_th = 5 * np.log10(np.maximum(dl, 1e-10)) + 25 + corr

            if not self.restrict_m:
                if self.combined:
                    mu_th = mu_th + np.where(self.sne_mask,
                                             pv.get('M_sne', 0.0),
                                             pv.get('M_qso', 0.0))
                else:
                    mu_th = mu_th + pv.get('mabs', 0.0)

            logL = _logL_mu(self.mu_obs - mu_th, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            return logL

    # ================================================================
    # 3. γCDM-Decay  —  Δμ = A·exp(-z/zd)
    # ================================================================
    if combined_mode:
        decay_params = {'A': None, 'zd': None, 'M_sne': None, 'M_qso': None}
    else:
        decay_params = {'A': None, 'zd': None, 'mabs': None}

    class DecayLikelihood(Likelihood):
        """
        γCDM-Decay: Exponential distance modulus correction Δμ = A·exp(-z/zd).
        At z=0: Δμ = A, providing a local calibration shift.
        Both A and zd are free parameters.
        """
        params = decay_params

        def initialize(self):
            self.z_mu = z_mu
            self.mu_obs = mu_obs
            self.err_eff = err_eff
            self.sne_mask = sne_mask
            self.combined = combined_mode
            self.z_cc = z_cc
            self.H_obs = H_obs
            self.err_cc = err_cc
            self.norm_cc = norm_cc
            self.restrict_m = restrict_m

        def get_requirements(self):
            reqs = {"angular_diameter_distance": {"z": np.concatenate([self.z_mu, self.z_cc])}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            return reqs

        def logp(self, **pv):
            try:
                da = self.provider.get_angular_diameter_distance(self.z_mu)
                dl = da * (1 + self.z_mu)**2
            except Exception:
                return -1e30

            A = pv.get('A', 0.0)
            zd = pv.get('zd', 1.0)
            corr = A * np.exp(-self.z_mu / zd)

            mu_th = 5 * np.log10(np.maximum(dl, 1e-10)) + 25 + corr

            if not self.restrict_m:
                if self.combined:
                    mu_th = mu_th + np.where(self.sne_mask,
                                             pv.get('M_sne', 0.0),
                                             pv.get('M_qso', 0.0))
                else:
                    mu_th = mu_th + pv.get('mabs', 0.0)

            logL = _logL_mu(self.mu_obs - mu_th, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            return logL

    # ================================================================
    # 4. γCDM-LOG²-DECAY  —  Two-component additive damped correction
    #
    #    Δμ(z) = A·exp(-z/zb)  +  γ₀·[ln(1+z)]²·exp(-z/zh)
    #            ─────────────     ────────────────────────────
    #            Local component    Long-range component
    #            (at z→0: = A)      (at z→0: = 0)
    #
    #    Free parameters: A, zb, gamma_log_decay, zh
    #    Both terms → 0 as z → ∞ (CMB protection)
    # ================================================================
    if combined_mode:
        log_decay_params = {'A': None, 'zb': None, 'gamma_log_decay': None, 'zh': None,
                            'M_sne': None, 'M_qso': None}
    else:
        log_decay_params = {'A': None, 'zb': None, 'gamma_log_decay': None, 'zh': None,
                            'mabs': None}

    class GammaCDM_LOG_DECAY_Likelihood(Likelihood):
        """
        γCDM-LOG²-DECAY: Two-component additive damped correction.

        Δμ(z) = A·exp(-z/zb) + γ₀·[ln(1+z)]²·exp(-z/zh)

        The local component A·exp(-z/zb) provides a calibration shift at z≈0,
        while γ₀·[ln(1+z)]²·exp(-z/zh) captures long-range distance modulus
        anomalies. Both terms decay to zero at high redshift, preserving
        consistency with CMB observations.

        Parameters
        ----------
        A : float
            Amplitude of the local exponential component.
        zb : float
            Decay scale of the local component (small: local effect).
        gamma_log_decay : float
            Amplitude of the logarithmic-squared component.
        zh : float
            Decay scale of the logarithmic component (large: long-range).
        """
        params = log_decay_params

        def initialize(self):
            self.z_mu = z_mu
            self.mu_obs = mu_obs
            self.err_eff = err_eff
            self.sne_mask = sne_mask
            self.combined = combined_mode
            self.z_cc = z_cc
            self.H_obs = H_obs
            self.err_cc = err_cc
            self.norm_cc = norm_cc
            self.restrict_m = restrict_m

        def get_requirements(self):
            reqs = {"angular_diameter_distance": {"z": np.concatenate([self.z_mu, self.z_cc])}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            return reqs

        def logp(self, **pv):
            try:
                da = self.provider.get_angular_diameter_distance(self.z_mu)
                dl = da * (1 + self.z_mu)**2
            except Exception:
                return -1e30

            A  = pv.get('A', 0.0)
            zb = pv.get('zb', 0.1)
            g0 = pv.get('gamma_log_decay', 0.0)
            zh = pv.get('zh', 10.0)

            # Two-component additive correction
            local_term = A * np.exp(-self.z_mu / zb)
            long_range = g0 * np.log1p(self.z_mu)**2 * np.exp(-self.z_mu / zh)
            corr = local_term + long_range

            mu_th = 5 * np.log10(np.maximum(dl, 1e-10)) + 25 + corr

            if not self.restrict_m:
                if self.combined:
                    mu_th = mu_th + np.where(self.sne_mask,
                                             pv.get('M_sne', 0.0),
                                             pv.get('M_qso', 0.0))
                else:
                    mu_th = mu_th + pv.get('mabs', 0.0)

            logL = _logL_mu(self.mu_obs - mu_th, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            return logL

    return (LCDMLikelihood, GammaCDM_LOG2_Likelihood,
            DecayLikelihood, GammaCDM_LOG_DECAY_Likelihood)
