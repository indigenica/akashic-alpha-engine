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

# Single source of truth for physical constants (Planck 2018 shift, c, etc.).
# Do NOT hard-code Z_STAR, R_PLANCK, SIGMA_R_PLANCK, SIGMA_CORR_CMB or
# C_LIGHT_KMS here; they come from cosmo_constants.
from cosmo_constants import (
    Z_STAR as _Z_STAR_CONST,
    R_PLANCK as _R_PLANCK_CONST,
    SIGMA_R_PLANCK as _SIG_R_CONST,
    SIGMA_CORR_CMB as _SIG_CORR_CONST,
    C_LIGHT_KMS as _C_LIGHT_CONST,
)


def create_likelihoods(z_mu, mu_obs, err_mu, z_cc, H_obs, err_cc,
                       sne_mask, combined_mode,
                       sigma_int_sne=0.0, sigma_int_qso=0.0,
                       no_nuisance=False, asymmetric=False, sanity_check=False,
                       likelihood_type='gaussian', nu=5.0,
                       C_inv_sne=None, ln_det_C_sne=None,
                       use_cmb=False, fit_scatter=False,
                       cov_evals=None, cov_evecs=None,
                       no_bubble=False):
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
    use_cmb : bool
        If True, add Planck 2018 CMB shift parameter prior (R = 1.7502 ± 0.0046)
        and correction penalty at z* = 1089.92.
    fit_scatter : bool
        If True, treat σ_int,SNe and σ_int,QSO as free sampled parameters.
    cov_evals, cov_evecs : np.ndarray or None
        Eigenvalues/vectors of C_base (no σ_int) for efficient scatter fitting.

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

    # ── CMB shift parameter (Planck 2018) — constants come from cosmo_constants ──
    _Z_STAR = _Z_STAR_CONST
    _R_PLANCK = _R_PLANCK_CONST
    _SIG_R = _SIG_R_CONST
    _SIG_CORR = _SIG_CORR_CONST
    _C_LIGHT = _C_LIGHT_CONST
    _use_cmb = use_cmb

    _z_req_cmb = np.array([_Z_STAR]) if _use_cmb else np.array([])

    def _cmb_logL(provider, delta_mu_star=0.0):
        """Gaussian log-likelihood penalty from Planck shift parameter R."""
        if not _use_cmb:
            return 0.0
        try:
            H0 = provider.get_param("H0")
            Om = provider.get_param("omegam")
            DA_star = provider.get_angular_diameter_distance(_z_req_cmb)[0]
            DC_star = (1 + _Z_STAR) * DA_star
            R_mod = np.sqrt(Om) * (H0 / _C_LIGHT) * DC_star
            pen = -0.5 * ((R_mod - _R_PLANCK) / _SIG_R) ** 2
            if abs(delta_mu_star) > 1e-10:
                pen += -0.5 * (delta_mu_star / _SIG_CORR) ** 2
            return pen
        except Exception:
            return -1e10

    # ── Scatter fitting state ──
    _fit_scatter = fit_scatter
    _cov_evals = cov_evals
    _cov_evecs = cov_evecs
    _err_mu_raw = err_mu

    def _logL_mu_scatter(residuals, sig_sne, sig_qso):
        """Gaussian log-likelihood with variable σ_int (for --fit-scatter).
        
        NOTE: When --fit-scatter is active, the likelihood is always Gaussian,
        even if --student or --cauchy were requested.  Fitting σ_int already
        accounts for heavy tails via an enlarged scatter; combining both would
        double-count the outlier absorption and is not statistically well-defined.
        """
        logL = 0.0
        if _cov_evals is not None:
            if combined_mode and sne_mask is not None:
                r_sne = residuals[sne_mask]
                v = _cov_evecs.T @ r_sne
                lam_eff = _cov_evals + sig_sne**2
                logL += -0.5 * (np.sum(v**2 / lam_eff) + np.sum(np.log(lam_eff)))
                r_qso = residuals[_qso_mask]
                e_qso = _err_mu_raw[_qso_mask]
                s2 = e_qso**2 + sig_qso**2
                logL += -0.5 * np.sum(r_qso**2 / s2 + np.log(s2))
            else:
                v = _cov_evecs.T @ residuals
                lam_eff = _cov_evals + sig_sne**2
                logL += -0.5 * (np.sum(v**2 / lam_eff) + np.sum(np.log(lam_eff)))
        else:
            if combined_mode and sne_mask is not None:
                r_sne = residuals[sne_mask]
                e_sne = _err_mu_raw[sne_mask]
                s2_sne = e_sne**2 + sig_sne**2
                logL += -0.5 * np.sum(r_sne**2 / s2_sne + np.log(s2_sne))
                r_qso = residuals[_qso_mask]
                e_qso = _err_mu_raw[_qso_mask]
                s2_qso = e_qso**2 + sig_qso**2
                logL += -0.5 * np.sum(r_qso**2 / s2_qso + np.log(s2_qso))
            else:
                s2 = _err_mu_raw**2 + sig_sne**2
                logL += -0.5 * np.sum(residuals**2 / s2 + np.log(s2))
        return logL

    # ================================================================
    # 1. ΛCDM  —  No model correction
    # ================================================================
    if combined_mode:
        lcdm_params = {'M_sne': None, 'M_qso': None}
    else:
        lcdm_params = {'mabs': None}
    if _fit_scatter:
        if combined_mode:
            lcdm_params['sigma_int_sne'] = None
            lcdm_params['sigma_int_qso'] = None
        else:
            if has_sne:
                lcdm_params['sigma_int_sne'] = None
            else:
                lcdm_params['sigma_int_qso'] = None

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
            _zall = np.concatenate([self.z_mu, self.z_cc, _z_req_cmb])
            reqs = {"angular_diameter_distance": {"z": _zall}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            if _use_cmb:
                reqs["H0"] = None
                reqs["omegam"] = None
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
            res = self.mu_obs - mu_th
            if _fit_scatter:
                if self.combined:
                    s_sne = pv.get('sigma_int_sne', 0.1)
                    s_qso = pv.get('sigma_int_qso', 0.5)
                else:
                    if has_sne:
                        s_sne = pv.get('sigma_int_sne', 0.1)
                        s_qso = s_sne
                    else:
                        s_qso = pv.get('sigma_int_qso', 0.5)
                        s_sne = s_qso
                logL = _logL_mu_scatter(res, s_sne, s_qso)
            else:
                logL = _logL_mu(res, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            logL += _cmb_logL(self.provider)
            return logL

    # ================================================================
    # 2. γCDM-LOG²  —  Δμ = γ₀·[ln(1+z)]²
    # ================================================================
    if combined_mode:
        log2_params = {'gamma_log2': None, 'M_sne': None, 'M_qso': None}
    else:
        log2_params = {'gamma_log2': None, 'mabs': None}
    if _fit_scatter:
        if combined_mode:
            log2_params['sigma_int_sne'] = None
            log2_params['sigma_int_qso'] = None
        else:
            if has_sne:
                log2_params['sigma_int_sne'] = None
            else:
                log2_params['sigma_int_qso'] = None

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
            _zall = np.concatenate([self.z_mu, self.z_cc, _z_req_cmb])
            reqs = {"angular_diameter_distance": {"z": _zall}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            if _use_cmb:
                reqs["H0"] = None
                reqs["omegam"] = None
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

            res = self.mu_obs - mu_th
            if _fit_scatter:
                if self.combined:
                    s_sne = pv.get('sigma_int_sne', 0.1)
                    s_qso = pv.get('sigma_int_qso', 0.5)
                else:
                    if has_sne:
                        s_sne = pv.get('sigma_int_sne', 0.1)
                        s_qso = s_sne
                    else:
                        s_qso = pv.get('sigma_int_qso', 0.5)
                        s_sne = s_qso
                logL = _logL_mu_scatter(res, s_sne, s_qso)
            else:
                logL = _logL_mu(res, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            dmu_star = gamma0 * np.log1p(_Z_STAR)**2
            logL += _cmb_logL(self.provider, dmu_star)
            return logL

    # ================================================================
    # 3. γCDM-Decay  —  Δμ = A·exp(-z/zd)
    # ================================================================
    if combined_mode:
        decay_params = {'A': None, 'zd': None, 'M_sne': None, 'M_qso': None}
    else:
        decay_params = {'A': None, 'zd': None, 'mabs': None}
    if _fit_scatter:
        if combined_mode:
            decay_params['sigma_int_sne'] = None
            decay_params['sigma_int_qso'] = None
        else:
            if has_sne:
                decay_params['sigma_int_sne'] = None
            else:
                decay_params['sigma_int_qso'] = None

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
            _zall = np.concatenate([self.z_mu, self.z_cc, _z_req_cmb])
            reqs = {"angular_diameter_distance": {"z": _zall}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            if _use_cmb:
                reqs["H0"] = None
                reqs["omegam"] = None
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

            res = self.mu_obs - mu_th
            if _fit_scatter:
                if self.combined:
                    s_sne = pv.get('sigma_int_sne', 0.1)
                    s_qso = pv.get('sigma_int_qso', 0.5)
                else:
                    if has_sne:
                        s_sne = pv.get('sigma_int_sne', 0.1)
                        s_qso = s_sne
                    else:
                        s_qso = pv.get('sigma_int_qso', 0.5)
                        s_sne = s_qso
                logL = _logL_mu_scatter(res, s_sne, s_qso)
            else:
                logL = _logL_mu(res, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            dmu_star = A * np.exp(-_Z_STAR / zd)
            logL += _cmb_logL(self.provider, dmu_star)
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
    if _fit_scatter:
        if combined_mode:
            log_decay_params['sigma_int_sne'] = None
            log_decay_params['sigma_int_qso'] = None
        else:
            if has_sne:
                log_decay_params['sigma_int_sne'] = None
            else:
                log_decay_params['sigma_int_qso'] = None

    class GammaCDM_LOG_DECAY_Likelihood(Likelihood):
        """
        γCDM-LOG²-DECAY: Two-component additive damped correction.

        Δμ(z) = A·exp(-z/zb) + γ₀·[ln(1+z)]²·exp(-z/zh)
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
            _zall = np.concatenate([self.z_mu, self.z_cc, _z_req_cmb])
            reqs = {"angular_diameter_distance": {"z": _zall}}
            if len(self.z_cc) > 0:
                reqs["Hubble"] = {"z": self.z_cc}
            if _use_cmb:
                reqs["H0"] = None
                reqs["omegam"] = None
            return reqs

        def logp(self, **pv):
            try:
                da = self.provider.get_angular_diameter_distance(self.z_mu)
                dl = da * (1 + self.z_mu)**2
            except Exception:
                return -1e30

            g0 = pv.get('gamma_log_decay', 0.0)
            zh = pv.get('zh', 10.0)

            # Two-component additive correction (--no-bubble suppresses local term)
            if no_bubble:
                local_term = 0.0
                dmu_star_local = 0.0
            else:
                A  = pv.get('A', 0.0)
                zb = pv.get('zb', 0.1)
                local_term = A * np.exp(-self.z_mu / zb)
                dmu_star_local = A * np.exp(-_Z_STAR / zb)
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

            res = self.mu_obs - mu_th
            if _fit_scatter:
                if self.combined:
                    s_sne = pv.get('sigma_int_sne', 0.1)
                    s_qso = pv.get('sigma_int_qso', 0.5)
                else:
                    if has_sne:
                        s_sne = pv.get('sigma_int_sne', 0.1)
                        s_qso = s_sne
                    else:
                        s_qso = pv.get('sigma_int_qso', 0.5)
                        s_sne = s_qso
                logL = _logL_mu_scatter(res, s_sne, s_qso)
            else:
                logL = _logL_mu(res, self.err_eff)

            if len(self.z_cc) > 0:
                H_th = self.provider.get_Hubble(self.z_cc)
                logL += -0.5 * (np.sum(((self.H_obs - H_th) / self.err_cc)**2) + self.norm_cc)

            dmu_star = dmu_star_local + g0 * np.log1p(_Z_STAR)**2 * np.exp(-_Z_STAR / zh)
            logL += _cmb_logL(self.provider, dmu_star)
            return logL

    return (LCDMLikelihood, GammaCDM_LOG2_Likelihood,
            DecayLikelihood, GammaCDM_LOG_DECAY_Likelihood)
