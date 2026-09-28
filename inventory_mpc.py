#!/usr/bin/env python
"""TabMPC: model predictive control with TabPFN forecasts, demonstrated on a synthetic retailer.

Each day the controller (1) forecasts external conditions 14 days ahead with TabPFN-3.5
(or a baseline forecaster), (2) turns the forecasts plus historical forecast-error
trajectories into scenarios, (3) solves a linear program whose future orders adapt to
information revealed along each scenario, (4) executes only today's order in an exact
simulator, then observes the outcome and replans. Ridge and gradient boosting use the same
features, scenarios and LP; a clairvoyant run is an information reference.

    python inventory_mpc.py --selfcheck      # consistency checks (seconds, no TabPFN)
    python inventory_mpc.py --quick          # smoke run: 1 seed x 28 days -> results_quick/
    python inventory_mpc.py                  # full paired evaluation -> results/
    python inventory_mpc.py --replot-only    # figures and tables from saved results, no inference
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.stats import t as student_t
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

TARGETS = ["propensity", "arrivals", "price_fast", "price_slow", "cap_fast", "cap_slow"]
# Supplier quotes (prices, capacities) for day t are known at decision time t.
# Propensity and customer arrivals for day t are only revealed after day t.
IS_QUOTE = np.array([False, False, True, True, True, True])
MODELS = ["ridge", "gbm", "tabpfn", "oracle"]
LABELS = {"ridge": "Ridge", "tabpfn": "TabPFN-3.5", "gbm": "Gradient boosting",
          "oracle": "Clairvoyant (diagnostic)"}
COLORS = {"tabpfn": "#2a78d6", "ridge": "#eb6834", "gbm": "#1baf7a", "oracle": "#7a7a7a"}
MIN_ORIGIN = 28  # longest rolling-window feature
FEATURE_LAGS = (0, 1, 6)
FEATURE_ROLL = (7, 28)
CAL_PERIODS = ((7, 1), (7, 2), (90, 1), (180, 1))  # (period, harmonic) of target date


# ----------------------------------------------------------------------------- config
@dataclass
class Config:
    # --- experiment
    out_dir: str = "results"
    cache_dir: str = ""         # forecast cache (default <out_dir>/cache)
    eval_seeds: tuple = (201, 202, 203, 204, 205)
    n_train: int = 360          # pre-evaluation history used only for learning
    n_calib: int = 90           # rolling-origin forecasts that seed the residual bank
    n_eval: int = 180           # closed-loop evaluated days per seed
    horizon: int = 14           # forecast and planning horizon (days)
    n_scenarios: int = 24
    deterministic: bool = False  # debugging: single point-forecast scenario
    refit_every: int = 7
    window: int = 360           # rolling training window (origins)
    context_cap: int = 2000     # max training rows per target (all models)
    device: str = "auto"
    tabpfn_version: str = "v3.5"
    tabpfn_n_estimators: int = 4
    # Baseline hyper-parameters, selected per target on development seeds 11-13 by
    # rolling-origin MAE (TabPFN not involved) and then frozen.
    ridge_alphas: tuple = (100.0, 100.0, 1000.0, 1000.0, 1000.0, 10000.0)
    gbm_params: tuple = ((0.05, 500), (0.05, 200), (0.05, 200), (0.05, 200), (0.05, 200), (0.1, 200))  # (lr, iters)
    # --- business
    retail_price: float = 10.0
    holding_cost: float = 0.08      # per unit of end-of-day stock per day
    disposal_penalty: float = 2.0   # per unit disposed because storage overflowed
    storage_cap: float = 400.0
    budget: float = 1450.0          # daily purchasing budget (order-date cash)
    lead_fast: int = 1
    lead_slow: int = 3
    rho: float = 0.98               # daily retention
    eta: float = 0.3                # customers lost per unit of lost demand
    customers0: float = 1000.0
    inventory0: float = 250.0
    pipeline0_slow: float = 180.0   # slow deliveries due on eval days 1 and 2
    # --- external environment: seasonal/AR drivers
    prop_base: float = 0.1899       # propensity level
    prop_slow_amp: float = 0.25     # 90-day log-amplitude
    prop_week_amp: float = 0.10     # weekly log-amplitude
    prop_week_boost: float = 0.06   # weekly amplitude grows in high season (interaction)
    prop_market_damp: float = 0.05  # demand dampening per sd of last week's market spike
    prop_ar_phi: float = 0.85
    prop_ar_sd: float = 0.045       # innovation sd (log scale)
    demand_market_corr: float = 0.4  # share of demand innovation from market innovation
    price_fast_base: float = 6.5
    price_slow_base: float = 5.0
    price_fast_amp: float = 0.06
    price_slow_amp: float = 0.08
    market_phi: float = 0.95
    market_sd: float = 0.015
    supplier_phi: float = 0.7
    supplier_sd: float = 0.02
    cap_fast_base: float = 110.0
    cap_slow_base: float = 190.0
    cap_amp: float = 0.05
    cap_noise_sd: float = 0.02
    disruption_recovery: int = 3
    arrivals_base: float = 20.0
    arrivals_amp: float = 0.10      # 180-day relative amplitude
    arrivals_phi: float = 0.5
    arrivals_sd: float = 1.5
    # --- external environment: observable context with nonlinear effects
    temp_mean: float = 17.0          # deg C, yearly cycle
    temp_amp: float = 9.0
    temp_sd: float = 2.0             # AR(1) innovation sd (phi 0.8)
    temp_fc_sd0: float = 1.0         # weather-forecast error sd at h=0 ...
    temp_fc_sd_slope: float = 0.3    # ... growing linearly with lead time
    event_rate: float = 0.10         # scheduled local events (schedule is public)
    event_size: tuple = (0.5, 1.0)
    event_effect: float = 0.35       # log-effect of a size-1 event in good weather
    event_weekend_boost: float = 0.6
    event_temp_mid: float = 14.0     # outdoor events fade in cold weather (logistic)
    event_temp_scale: float = 2.5
    heat_threshold: float = 27.0     # demand rises sharply above this temperature
    heat_effect: float = 0.25        # log-effect per 4 deg above threshold
    promo_period: int = 28           # fixed promotion calendar
    promo_len: int = 3
    promo_effect: float = 0.20       # larger in low season, halved on event days
    congestion_phi: float = 0.93
    disruption_base_hazard: float = 0.002
    disruption_cong_hazard: float = 0.15
    congestion_threshold: float = 0.6
    congestion_cap_drag: float = 0.12
    fast_price_spillover: float = 0.12   # fast price rises while the slow supplier is disrupted
    slow_congestion_premium: float = 0.05
    # --- planner (identical for every forecaster)
    controller: str = "affine"      # affine: future orders react to observations; shared: one open-loop plan
    terminal_customer_kappa: float = 0.5
    terminal_inventory_factor: float = 0.8
    terminal_ref_days: int = 90

    def cache_key_env(self) -> dict:
        """Environment parameters hashed into the forecast-cache key. The names and values
        reproduce the key under which the committed forecasts were computed (the world was then
        labelled "rich", its propensity level "rich_prop_base", and four parameters of a retired
        seasonal-only world were still hashed), so the expensive cache remains valid."""
        d = {k: getattr(self, k) for k in _ENV_FIELDS}
        d["rich_prop_base"] = d.pop("prop_base")
        d.update(_RETIRED_KEY_PARAMS)
        return d

    @property
    def t_eval0(self):
        return self.n_train + self.n_calib

    @property
    def path_len(self):
        return self.n_train + self.n_calib + self.n_eval + self.horizon


_names = [f.name for f in dataclasses.fields(Config)]
_ENV_FIELDS = tuple(_names[_names.index("prop_base"):_names.index("slow_congestion_premium") + 1])
_RETIRED_KEY_PARAMS = {"env_variant": "rich", "prop_base": 0.2, "disruption_rate": 1 / 60,
                       "disruption_len": (3, 8), "disruption_depth": (0.25, 0.65)}


# ------------------------------------------------------------------ external environment
def _ar1(innov, phi):
    x = np.empty_like(innov)
    x[0] = innov[0] / np.sqrt(1 - phi**2)
    for i in range(1, len(innov)):
        x[i] = phi * x[i - 1] + innov[i]
    return x


def _hazard_disruptions(rng, g, cfg):
    """Disruptions whose onset hazard and severity rise with observed congestion g.
    Returns the capacity factor and 'days since the current disruption began' (0 = normal)."""
    n = len(g)
    fac, dsd = np.ones(n), np.zeros(n)
    R = cfg.disruption_recovery
    d = 0
    while d < n:
        haz = cfg.disruption_base_hazard + cfg.disruption_cong_hazard * max(
            0.0, g[d] - cfg.congestion_threshold) / (1 - cfg.congestion_threshold)
        if rng.uniform() < haz:
            depth = float(np.clip(0.75 - 0.6 * g[d], 0.15, 0.7))
            dur = 2 + int(round(8 * g[d])) + int(rng.integers(0, 2))
            seg = np.concatenate([np.full(dur, depth), depth + (1 - depth) * np.arange(1, R + 1) / (R + 1)])
            k = min(len(seg), n - d)
            fac[d:d + k] = seg[:k]
            dsd[d:d + k] = np.arange(1, k + 1)
            d += k
        else:
            d += 1
    return fac, dsd


def generate_external(cfg: Config, seed: int, length: int | None = None) -> pd.DataFrame:
    """Hidden ground-truth paths of the six forecast targets."""
    return generate_world(cfg, seed, length)[0]


def generate_world(cfg: Config, seed: int, length: int | None = None):
    """Hidden ground-truth external paths (independent of any controller action) and the
    observable covariates. Returns (targets DataFrame, covariate dict)."""
    n = cfg.path_len if length is None else length
    # Seven streams are spawned so that every path stays bit-identical to the committed
    # results; streams 5-6 belonged to a retired capacity model and are unused.
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence([20260927, seed]).spawn(7)]
    r_phase, r_dem, r_mkt, r_sup, _, _, r_arr = rngs
    d = np.arange(n)
    ph = r_phase.uniform(0, 2 * np.pi, size=6)
    tau = 2 * np.pi
    # Shared market factor drives both supplier prices; its innovations also enter demand.
    e_m = r_mkt.standard_normal(n)
    market = _ar1(cfg.market_sd * e_m, cfg.market_phi)
    c = cfg.demand_market_corr
    e_z = np.sqrt(1 - c**2) * r_dem.standard_normal(n) + c * e_m
    z = _ar1(cfg.prop_ar_sd * e_z, cfg.prop_ar_phi)
    s90 = np.sin(tau * d / 90 + ph[0])
    w7 = np.sin(tau * d / 7 + ph[1])
    # Interpretable nonlinearity: last week's market spike (visible in lagged prices)
    # dampens demand, and weekly swings are stronger in high season.
    m_sd = cfg.market_sd / np.sqrt(1 - cfg.market_phi**2)
    cs = np.concatenate([[0.0], np.cumsum(market)])
    mk7 = np.zeros(n)
    mk7[7:] = (cs[7:n] - cs[0:n - 7]) / 7 / m_sd
    logm = (cfg.prop_slow_amp * s90 + (cfg.prop_week_amp + cfg.prop_week_boost * s90) * w7
            + z - cfg.prop_market_damp * np.maximum(0.0, mk7))
    # Observable context: weather (with a noisy forecast), public event schedule, promotions.
    r_t, r_fc, r_ev, r_cf, r_cs, r_pr = [np.random.default_rng(s) for s in
                                         np.random.SeedSequence([20260927, seed, 77]).spawn(6)]
    H = cfg.horizon
    dd = np.arange(n + H)
    temp_all = (cfg.temp_mean + cfg.temp_amp * np.sin(tau * dd / 365 + r_t.uniform(0, tau))
                + _ar1(cfg.temp_sd * r_t.standard_normal(n + H), 0.8))
    temp = temp_all[:n]
    hs = np.arange(H)
    # Weather forecast issued on day o for day o+h: truth + error growing with lead time.
    temp_fc = (temp_all[d[:, None] + hs[None]]
               + (cfg.temp_fc_sd0 + cfg.temp_fc_sd_slope * hs)[None] * r_fc.standard_normal((n, H)))
    weekend = (d % 7 >= 5).astype(float)
    event = np.where(r_ev.uniform(size=n) < cfg.event_rate, r_ev.uniform(*cfg.event_size, size=n), 0.0)
    promo = (((d - r_pr.integers(cfg.promo_period)) % cfg.promo_period) < cfg.promo_len).astype(float)
    outdoor = 1 / (1 + np.exp(-(temp - cfg.event_temp_mid) / cfg.event_temp_scale))
    f = (cfg.event_effect * event * (1 + cfg.event_weekend_boost * weekend) * outdoor
         + cfg.heat_effect * np.maximum(0.0, temp - cfg.heat_threshold) / 4 * (1 + 0.5 * weekend)
         + cfg.promo_effect * promo * (1 - 0.6 * s90) * (1 - 0.5 * (event > 0)))
    logm = logm + f
    cov = dict(event=event, promo=promo, weekend=weekend, temp=temp, temp_fc=temp_fc)
    prop = np.clip(cfg.prop_base * np.exp(logm), 0.01, 0.9)
    v_f = _ar1(cfg.supplier_sd * r_sup.standard_normal(n), cfg.supplier_phi)
    v_s = _ar1(cfg.supplier_sd * r_sup.standard_normal(n), cfg.supplier_phi)
    pf = cfg.price_fast_base * np.exp(cfg.price_fast_amp * np.sin(tau * d / 90 + ph[2]) + market + v_f)
    ps = cfg.price_slow_base * np.exp(cfg.price_slow_amp * np.sin(tau * d / 90 + ph[3]) + market + v_s)
    arr = cfg.arrivals_base * (1 + cfg.arrivals_amp * np.sin(tau * d / 180 + ph[5]))
    arr = np.maximum(0.0, arr + _ar1(cfg.arrivals_sd * r_arr.standard_normal(n), cfg.arrivals_phi))
    # Observable supplier congestion drives disruption hazard/severity; status is announced.
    phi = cfg.congestion_phi
    out = {}
    for key, r, base, ph_c in (("f", r_cf, cfg.cap_fast_base, ph[4]), ("s", r_cs, cfg.cap_slow_base, ph[4] + np.pi / 2)):
        g = 1 / (1 + np.exp(-(1.6 * _ar1(np.sqrt(1 - phi**2) * r.standard_normal(n), phi) - 0.3)))
        fac, dsd = _hazard_disruptions(r, g, cfg)
        cap = (base * (1 + cfg.cap_amp * np.sin(tau * d / 90 + ph_c)) * (1 + cfg.cap_noise_sd * r.standard_normal(n))
               * (1 - cfg.congestion_cap_drag * g) * fac)
        out[key] = (g, fac, dsd, np.maximum(cap, 0.0))
    capf, caps = out["f"][3], out["s"][3]
    pf = pf * np.exp(cfg.fast_price_spillover * (1 - out["s"][1]))
    ps = ps * np.exp(cfg.slow_congestion_premium * out["s"][0])
    cov.update(cong_f=out["f"][0], cong_s=out["s"][0], dsd_f=out["f"][2], dsd_s=out["s"][2])
    return pd.DataFrame({"propensity": prop, "arrivals": arr, "price_fast": pf, "price_slow": ps,
                         "cap_fast": np.maximum(capf, 0.0), "cap_slow": np.maximum(caps, 0.0)}), cov


# ----------------------------------------------------------------- target transforms
def to_model_space(Y: np.ndarray) -> np.ndarray:
    """Raw targets -> modelling space (identical for both forecasters)."""
    T = np.array(Y, dtype=float, copy=True)
    p = np.clip(T[..., 0], 1e-3, 1 - 1e-3)
    T[..., 0] = np.log(p / (1 - p))
    T[..., 2:4] = np.log(T[..., 2:4])
    return T


def from_model_space(T: np.ndarray, cfg: Config) -> np.ndarray:
    """Inverse transform plus physical bounds."""
    Y = np.array(T, dtype=float, copy=True)
    Y[..., 0] = np.clip(1 / (1 + np.exp(-T[..., 0])), 0.001, 0.95)
    Y[..., 1] = np.maximum(T[..., 1], 0.0)
    Y[..., 2:4] = np.exp(T[..., 2:4])
    Y[..., 4] = np.clip(T[..., 4], 0.0, 1.5 * cfg.cap_fast_base)
    Y[..., 5] = np.clip(T[..., 5], 0.0, 1.5 * cfg.cap_slow_base)
    return Y


# -------------------------------------------------------------------------- features
def last_observed(origins: np.ndarray, j: int) -> np.ndarray:
    """Index of the newest value of series j observable at decision time `origin`."""
    return origins if IS_QUOTE[j] else origins - 1


def origin_features(T: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Lags and rolling means of every observed series, as known at each origin."""
    cs = np.vstack([np.zeros((1, T.shape[1])), np.cumsum(T, axis=0)])
    cols = []
    for j in range(T.shape[1]):
        last = last_observed(origins, j)
        for lag in FEATURE_LAGS:
            cols.append(T[last - lag, j])
        for w in FEATURE_ROLL:
            cols.append((cs[last + 1, j] - cs[last + 1 - w, j]) / w)
    return np.column_stack(cols)


def calendar_features(days: np.ndarray) -> np.ndarray:
    cols = []
    for period, k in CAL_PERIODS:
        a = 2 * np.pi * k * days / period
        cols += [np.sin(a), np.cos(a)]
    return np.column_stack(cols)


def covariate_origin_features(cov: dict, origins: np.ndarray) -> np.ndarray:
    """Congestion (today and 7-day mean) and disruption status are announced at decision
    time; actual temperature is known up to yesterday."""
    cols = []
    for k in ("cong_f", "cong_s"):
        cs = np.r_[0.0, np.cumsum(cov[k])]
        cols += [cov[k][origins], (cs[origins + 1] - cs[origins - 6]) / 7]
    cols += [cov["dsd_f"][origins], cov["dsd_s"][origins]]
    cs = np.r_[0.0, np.cumsum(cov["temp"])]
    cols += [cov["temp"][origins - 1], (cs[origins] - cs[origins - 7]) / 7]
    return np.column_stack(cols)


def covariate_row_features(cov: dict, origins: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    """Known-in-advance covariates of the target date: event schedule, promotion calendar,
    weekend flag, and the weather forecast issued at the origin for that date."""
    tgt = origins + horizons
    return np.column_stack([cov["event"][tgt], cov["promo"][tgt], cov["weekend"][tgt],
                            cov["temp_fc"][origins, horizons]])


def row_features(Xo: np.ndarray, origins: np.ndarray, horizons: np.ndarray, cov: dict) -> np.ndarray:
    """Pooled direct multi-horizon rows: origin features + horizon + target-date calendar +
    known target-date covariates."""
    return np.column_stack([Xo[origins], horizons.astype(float), calendar_features(origins + horizons),
                            covariate_row_features(cov, origins, horizons)])


def target_horizons(j: int, H: int) -> np.ndarray:
    # Today's quotes are known exactly, so quote targets are only forecast for h >= 1.
    return np.arange(1, H) if IS_QUOTE[j] else np.arange(0, H)


def training_rows(r: int, j: int, cfg: Config, seed: int):
    """(origin, horizon) pairs whose label is observed by refit origin r, capped."""
    H = cfg.horizon
    o = np.arange(max(MIN_ORIGIN, r - cfg.window), r)
    hs = target_horizons(j, H)
    oo, hh = np.meshgrid(o, hs, indexing="ij")
    oo, hh = oo.ravel(), hh.ravel()
    label_day_limit = r if IS_QUOTE[j] else r - 1
    keep = oo + hh <= label_day_limit
    oo, hh = oo[keep], hh[keep]
    if len(oo) > cfg.context_cap:
        rng = np.random.default_rng([seed, r, j, 7])
        sel = np.sort(rng.choice(len(oo), cfg.context_cap, replace=False))
        oo, hh = oo[sel], hh[sel]
    return oo, hh


# ------------------------------------------------------------------------ forecasters
def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TabPFNFactory:
    def __init__(self, cfg: Config):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        self.version = ModelVersion(cfg.tabpfn_version)
        self.device = resolve_device(cfg.device)
        self.model = TabPFNRegressor.create_default_for_version(
            self.version, device=self.device, n_estimators=cfg.tabpfn_n_estimators,
            random_state=0, ignore_pretraining_limits=True)
        self.model_id = Path(str(self.model.model_path)).name

    def __call__(self, j):
        return self.model


def model_info(cfg: Config) -> dict:
    info = {"tabpfn_version": cfg.tabpfn_version, "tabpfn_n_estimators": cfg.tabpfn_n_estimators,
            "device": resolve_device(cfg.device)}
    try:
        import tabpfn
        import torch
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import ModelSource
        src = {"v3.5": ModelSource.get_v3_5, "v3.5-fast": ModelSource.get_v3_5_fast}.get(cfg.tabpfn_version)
        info.update(tabpfn_package=tabpfn.__version__ if hasattr(tabpfn, "__version__") else _pkg_version("tabpfn"),
                    torch=torch.__version__, tabpfn_model_version_enum=str(ModelVersion(cfg.tabpfn_version)),
                    tabpfn_checkpoint=src().default_filename if src else None,
                    tabpfn_repo=src().repo_id if src else None)
    except Exception as exc:  # recorded, never hidden
        info["tabpfn_import_error"] = repr(exc)
    return info


def _pkg_version(name):
    from importlib.metadata import version
    try:
        return version(name)
    except Exception:
        return None


def forecast_key(cfg: Config, model: str, seed: int, alphas) -> str:
    payload = {"env": cfg.cache_key_env(), "n_train": cfg.n_train, "n_calib": cfg.n_calib, "n_eval": cfg.n_eval,
               "H": cfg.horizon, "refit": cfg.refit_every, "window": cfg.window, "cap": cfg.context_cap,
               "model": model, "seed": seed, "features": [FEATURE_LAGS, FEATURE_ROLL, CAL_PERIODS], "v": 3}
    if model in ("ridge", "gbm"):
        payload["alphas"] = list(alphas)
        payload["sklearn"] = _pkg_version("scikit-learn")
    else:
        payload.update(version=cfg.tabpfn_version, n_est=cfg.tabpfn_n_estimators,
                       pkg=_pkg_version("tabpfn"), device=resolve_device(cfg.device))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def alphas_for(cfg: Config, model: str):
    return cfg.gbm_params if model == "gbm" else cfg.ridge_alphas


def make_estimator(model: str, alpha: float, tabpfn_factory, j: int):
    if model == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if model == "gbm":
        # Strong generic nonlinear baseline on the same inputs; no random early-stopping split.
        lr, iters = alpha
        return HistGradientBoostingRegressor(learning_rate=lr, max_iter=int(iters), max_leaf_nodes=15,
                                             min_samples_leaf=20, early_stopping=False, random_state=0)
    return tabpfn_factory(j)


def rolling_forecasts(cfg: Config, T: np.ndarray, model: str, seed: int, alphas=None,
                      tabpfn_factory=None, desc="", cov=None) -> dict:
    """Direct multi-horizon forecasts for every origin in [n_train, n_train+n_calib+n_eval).

    Refit every `refit_every` days on rows whose labels are observed at the refit origin;
    daily forecasts use each origin's own (causal) lag features. Because the external
    targets do not depend on the controller, computing all origins up front is
    equivalent to computing them online, and lets every controller reuse them.
    """
    H = cfg.horizon
    alphas = alphas_for(cfg, model) if alphas is None else alphas
    cov = generate_world(cfg, seed)[1] if cov is None else cov
    start, end = cfg.n_train, cfg.n_train + cfg.n_calib + cfg.n_eval
    all_o = np.arange(MIN_ORIGIN, end)
    feats = np.column_stack([origin_features(T, all_o), covariate_origin_features(cov, all_o)])
    Xo = np.full((end, feats.shape[1]), np.nan)
    Xo[MIN_ORIGIN:] = feats
    F = np.full((end - start, H, 6), np.nan)
    fit_t = pred_t = 0.0
    n_rows = []
    refits = range(start, end, cfg.refit_every)
    for r in tqdm(refits, desc=desc or f"{model} seed {seed}", leave=False, dynamic_ncols=True):
        test_o = np.arange(r, min(r + cfg.refit_every, end))
        for j in range(6):
            oo, hh = training_rows(r, j, cfg, seed)
            X, y = row_features(Xo, oo, hh, cov), T[oo + hh, j]
            hs = target_horizons(j, H)
            to, th = np.meshgrid(test_o, hs, indexing="ij")
            Xt = row_features(Xo, to.ravel(), th.ravel(), cov)
            est = make_estimator(model, alphas[j], tabpfn_factory, j)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X, y)
                t1 = time.perf_counter()
                pred = np.asarray(est.predict(Xt), dtype=float)
            t2 = time.perf_counter()
            fit_t += t1 - t0
            pred_t += t2 - t1
            n_rows.append(len(y))
            F[(to - start).ravel(), th.ravel(), j] = pred
            if IS_QUOTE[j]:
                F[test_o - start, 0, j] = T[test_o, j]  # known quote, zero uncertainty
    assert np.isfinite(F).all()
    return {"F": F, "fit_seconds": fit_t, "predict_seconds": pred_t, "n_refits": len(refits),
            "mean_train_rows": float(np.mean(n_rows))}


def oracle_forecasts(cfg: Config, T: np.ndarray) -> np.ndarray:
    """Clairvoyant diagnostic: the true future in model space (never used for learning)."""
    o = np.arange(cfg.n_train, cfg.n_train + cfg.n_calib + cfg.n_eval)
    return np.stack([T[o + h] for h in range(cfg.horizon)], axis=1)


def get_forecasts(cfg, T, model, seed, cache_dir: Path, tabpfn_factory=None, alphas=None):
    if model == "oracle":
        return {"F": oracle_forecasts(cfg, T), "fit_seconds": 0.0, "predict_seconds": 0.0, "n_refits": 0,
                "mean_train_rows": 0.0, "cached": False}
    alphas = alphas_for(cfg, model) if alphas is None else alphas
    key = forecast_key(cfg, model, seed, alphas)
    path = cache_dir / f"forecast_{model}_seed{seed}_{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        meta["cached"] = True
        return {"F": z["F"], **meta}
    if model == "tabpfn" and tabpfn_factory is None:
        raise RuntimeError("TabPFN forecasts not cached and TabPFN is not loaded.")
    out = rolling_forecasts(cfg, T, model, seed, alphas, tabpfn_factory)
    meta = {k: v for k, v in out.items() if k != "F"}
    if model == "tabpfn":
        meta["model_id"] = tabpfn_factory.model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, F=out["F"], meta=json.dumps(meta))
    meta["cached"] = False
    return {"F": out["F"], **meta}


def residual_bank(cfg: Config, T: np.ndarray, F: np.ndarray) -> np.ndarray:
    """E[o, h, j] = truth - forecast in model space; the trajectory of origin o is fully
    observed once o + H - 1 <= t - 1, i.e. usable from origin t >= o + H."""
    start, H = cfg.n_train, cfg.horizon
    o = start + np.arange(F.shape[0])
    truth = np.stack([T[o + h] for h in range(H)], axis=1)
    E = truth - F
    E[:, 0, IS_QUOTE] = 0.0
    return E


def available_bank(cfg: Config, t: int) -> np.ndarray:
    """Positions (into F/E) of residual trajectories observed by origin t."""
    return np.arange(0, max(0, t - cfg.horizon - cfg.n_train + 1))


def scenario_indices(cfg: Config, seed: int, t: int) -> np.ndarray:
    """Shared across models: same seed/day -> same historical trajectories."""
    avail = available_bank(cfg, t)
    rng = np.random.default_rng([seed, t, 11])
    return rng.choice(avail, size=cfg.n_scenarios, replace=True)


def build_scenarios(cfg: Config, F: np.ndarray, E: np.ndarray, seed: int, t: int) -> np.ndarray:
    f = F[t - cfg.n_train]
    if cfg.deterministic:
        return from_model_space(f[None], cfg)
    idx = scenario_indices(cfg, seed, t)
    return from_model_space(f[None] + E[idx], cfg)  # [N, H, 6]


# ---------------------------------------------------------------------------- planner
@dataclass
class PlanResult:
    qf: float
    qs: float
    ok: bool
    status: int
    message: str
    solve_seconds: float
    residual: float
    withheld_frac: float
    withheld_first_day_frac: float
    planned_profit: float
    objective: float = np.nan
    future_order_spread: float = np.nan  # mean across-scenario sd of planned future orders (affine only)


def terminal_values(cfg: Config, prop_hist: np.ndarray, ps_hist: np.ndarray):
    """Conservative terminal values from observed history (identical for both models).

    Customer: kappa * (retail - mean slow price) * mean propensity / (1 - rho), i.e. a
    haircut on the retention-weighted lifetime margin. Stock and paid pipeline:
    factor * mean slow price, strictly below replacement cost so the LP never buys
    purely for terminal value.
    """
    n = cfg.terminal_ref_days
    p_ref, c_ref = float(np.mean(prop_hist[-n:])), float(np.mean(ps_hist[-n:]))
    v_c = cfg.terminal_customer_kappa * max(cfg.retail_price - c_ref, 0.0) * p_ref / (1 - cfg.rho)
    v_i = cfg.terminal_inventory_factor * c_ref
    return v_c, v_i


def affine_inputs(scen: np.ndarray) -> np.ndarray:
    """z[s, k, :] for the restricted affine order rule: information revealed before ordering on
    future day k (that day's capacity quotes, the previous day's observed propensity), centred
    across scenarios per day and scaled by the pooled sd. Zero for k = 0 (today is shared)."""
    N, H = scen.shape[:2]
    raw = np.zeros((N, H, 3))
    raw[:, 1:, 0], raw[:, 1:, 1], raw[:, 1:, 2] = scen[:, 1:, 4], scen[:, 1:, 5], scen[:, :-1, 0]
    z = raw - raw.mean(axis=0, keepdims=True)
    sd = z[:, 1:].reshape(-1, 3).std(axis=0) if H > 1 else np.ones(3)
    z = z / np.where(sd > 1e-9, sd, np.inf)
    z[:, 0] = 0.0
    return z


def solve_plan(cfg: Config, I0: float, C0: float, pipeline: np.ndarray, scen: np.ndarray,
               v_c: float, v_i: float, controller: str | None = None) -> PlanResult:
    """Scenario LP. Today's order is always shared by all scenarios.

    controller="shared": ONE future order schedule for all scenarios (open loop); future
    orders must fit every scenario's budget and the minimum capacity across scenarios.
    controller="affine": future orders follow q[j,k,s] = a[j,k] + b[j]·z[s,k], with z the
    information revealed before ordering on day k (non-anticipative), so each scenario's
    own capacity and budget apply. Slopes b are shared across days (6 parameters) to limit
    over-fitting to the sampled trajectories; the problem stays an LP.

    pipeline[k]: known deliveries arriving on day t+k (pipeline[0] already settled).
    scen[s, k, :]: propensity, arrivals, price_fast, price_slow, cap_fast, cap_slow.
    Day-0 sales are fixed to the exact min(demand, stock). For k >= 1 sales are relaxed
    to sales <= demand, sales <= stock (not sales = min(.)); the simulator enforces
    exact fulfillment and the planned withholding is reported as a diagnostic.
    """
    N, H = scen.shape[0], scen.shape[1]
    Lf, Ls = cfg.lead_fast, cfg.lead_slow
    prop, arr, pf, ps, capf, caps = (scen[..., j] for j in range(6))
    affine = (controller or cfg.controller) == "affine"
    Z = affine_inputs(scen) if affine else None
    nz = 3 if affine else 0
    n_state = 2 * H + 5 * H * N
    nv = n_state + 2 * nz
    s_idx = np.arange(N)
    base = 2 * H + 5 * H * s_idx

    def qterms(j, k, coef):
        """Linear expression of supplier j's order on day k in every scenario, times coef."""
        coef = np.broadcast_to(coef, (N,))
        terms = [(np.full(N, j * H + k), coef)]
        if affine and k >= 1:
            terms += [(np.full(N, n_state + j * nz + m), coef * Z[:, k, m]) for m in range(nz)]
        return terms

    def delivered(k, coef):
        t = []
        if k >= Lf:
            t += qterms(0, k - Lf, coef)
        if k >= Ls:
            t += qterms(1, k - Ls, coef)
        return t

    def sales(k):
        return base + k

    def lost(k):
        return base + H + k

    def inv(k):  # end-of-day inventory of day k-1, k = 1..H
        return base + 2 * H + (k - 1)

    def cust(k):  # customers at start of day k, k = 1..H
        return base + 3 * H + (k - 1)

    def disp(k):
        return base + 4 * H + k

    er, ec, ev, eb = [], [], [], []
    ur, uc, uv, ub = [], [], [], []
    row = [0]

    def add_eq(terms, rhs):
        r = row[0] + s_idx
        for cols, vals in terms:
            er.append(r); ec.append(cols); ev.append(np.broadcast_to(vals, (N,)).astype(float))
        eb.append(np.broadcast_to(rhs, (N,)).astype(float))
        row[0] += N

    urow = [0]

    def add_ub(terms, rhs):
        r = urow[0] + s_idx
        for cols, vals in terms:
            ur.append(r); uc.append(cols); uv.append(np.broadcast_to(vals, (N,)).astype(float))
        ub.append(np.broadcast_to(rhs, (N,)).astype(float))
        urow[0] += N

    for k in range(H):
        # inventory balance: I_{k+1} = I_k + deliveries_k - disp_k - sales_k
        terms = [(inv(k + 1), 1.0), (disp(k), 1.0), (sales(k), 1.0)] + delivered(k, -1.0)
        rhs = pipeline[k]
        if k >= 1:
            terms.append((inv(k), -1.0))
        else:
            rhs = rhs + I0
        add_eq(terms, rhs)
        # demand split: sales + lost = propensity * customers (propensity fixed per scenario)
        terms = [(sales(k), 1.0), (lost(k), 1.0)]
        if k >= 1:
            terms.append((cust(k), -prop[:, k]))
            add_eq(terms, 0.0)
        else:
            add_eq(terms, prop[:, 0] * C0)
        # customer dynamics
        terms = [(cust(k + 1), 1.0), (lost(k), cfg.eta)]
        rhs = arr[:, k]
        if k >= 1:
            terms.append((cust(k), -cfg.rho))
        else:
            rhs = rhs + cfg.rho * C0
        add_eq(terms, rhs)
        # storage after deliveries/disposal (day 0 already settled)
        if k >= 1:
            add_ub([(inv(k), 1.0), (disp(k), -1.0)] + delivered(k, 1.0), cfg.storage_cap - pipeline[k])
        # scenario-wise budget (for the shared controller: a conservative sampled-robust constraint)
        add_ub(qterms(0, k, pf[:, k]) + qterms(1, k, ps[:, k]), cfg.budget)
        if affine and k >= 1:  # each scenario's own capacity; orders non-negative
            for j, cap in ((0, capf), (1, caps)):
                add_ub(qterms(j, k, 1.0), cap[:, k])
                add_ub(qterms(j, k, -1.0), 0.0)

    A_eq = sp.csr_matrix((np.concatenate(ev), (np.concatenate(er), np.concatenate(ec))), shape=(row[0], nv))
    b_eq = np.concatenate(eb)
    A_ub = sp.csr_matrix((np.concatenate(uv), (np.concatenate(ur), np.concatenate(uc))), shape=(urow[0], nv))
    b_ub = np.concatenate(ub)

    c = np.zeros(nv)
    p, h = cfg.retail_price, cfg.holding_cost
    for k in range(H):
        for j, price, lead in ((0, pf, Lf), (1, ps, Ls)):
            unit = price[:, k] - (v_i if k + lead >= H else 0.0)  # purchase cost net of terminal value
            for cols, vals in qterms(j, k, unit / N):
                c[cols[0]] += vals.sum()
        c[sales(k)] -= p / N
        c[inv(k + 1)] += h / N
        c[disp(k)] += cfg.disposal_penalty / N
    c[inv(H)] -= v_i / N
    c[cust(H)] -= v_c / N

    lb = np.zeros(nv)
    ubd = np.full(nv, np.inf)
    ubd[:H] = capf.min(axis=0)
    ubd[H:2 * H] = caps.min(axis=0)
    if affine:  # future intercepts and slopes are free; per-scenario rows above bound the orders
        lb[1:H] = lb[H + 1:2 * H] = -np.inf
        ubd[1:H] = ubd[H + 1:2 * H] = np.inf
        lb[n_state:] = -np.inf
    ubd[disp(0)] = 0.0
    # Day-0 fulfillment is exogenous (stock I0 is settled, today's orders cannot arrive
    # today), so fix it to the exact physical value; only days k >= 1 stay relaxed.
    lb[sales(0)] = ubd[sales(0)] = np.minimum(prop[:, 0] * C0, I0)
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=np.column_stack([lb, ubd]),
                  method="highs")
    dt = time.perf_counter() - t0
    if res.status != 0 or res.x is None:
        return PlanResult(0.0, 0.0, False, int(res.status), str(res.message), dt, np.nan, np.nan, np.nan, np.nan)
    x = res.x
    resid = max(np.abs(A_eq @ x - b_eq).max(), np.maximum(A_ub @ x - b_ub, 0).max(),
                np.maximum(lb - x, 0).max(), np.maximum(x - ubd, 0).max())
    scale = max(1.0, np.abs(b_eq).max(), np.abs(b_ub).max())
    ok = resid <= 1e-6 * scale
    # Voluntary withholding: stock left over while demand was lost = min(lost, end stock).
    S = np.stack([x[sales(k)] for k in range(H)], 1)
    Lo = np.stack([x[lost(k)] for k in range(H)], 1)
    Ie = np.stack([x[inv(k + 1)] for k in range(H)], 1)
    wh = np.minimum(Lo, Ie)
    dem = S + Lo
    Q = np.zeros((2, N, H))  # planned orders per supplier, scenario, day
    for j in range(2):
        for k in range(H):
            Q[j, :, k] = sum(x[cols] * vals for cols, vals in qterms(j, k, 1.0))
    beyond = sum(Q[0, :, k] for k in range(H) if k + Lf >= H) + sum(Q[1, :, k] for k in range(H) if k + Ls >= H)
    terminal = v_i * (x[inv(H)].mean() + np.mean(beyond)) + v_c * x[cust(H)].mean()
    return PlanResult(float(x[0]), float(x[H]), bool(ok), 0, str(res.message) if ok else f"residual {resid:.2e}",
                      dt, float(resid), float(wh.sum() / max(dem.sum(), 1e-9)),
                      float(wh[:, 0].sum() / max(dem[:, 0].sum(), 1e-9)), float(-res.fun - terminal),
                      objective=float(-res.fun), future_order_spread=float(Q[:, :, 1:].std(axis=1).mean()) if H > 1 else 0.0)


# -------------------------------------------------------------------------- simulator
class Simulator:
    """Exact physical dynamics. Deliveries due on day t settle at the START of day t
    (before the order decision); overflow beyond storage is disposed immediately and
    penalised. Orders are paid on the order date; demand is served automatically:
    sales = min(demand, available)."""

    def __init__(self, cfg: Config, ext: pd.DataFrame, t0: int):
        self.cfg, self.ext, self.t = cfg, ext.to_numpy(), t0
        self.inventory = cfg.inventory0
        self.customers = cfg.customers0
        self.pipeline = np.zeros(len(ext) + 10)
        self.pipeline[t0 + 1] += cfg.pipeline0_slow
        self.pipeline[t0 + 2] += cfg.pipeline0_slow
        self.disposed_today = 0.0
        self.delivered_total = 0.0

    def settle(self):
        arriving = self.pipeline[self.t]
        self.pipeline[self.t] = 0.0
        self.delivered_total += arriving
        avail = self.inventory + arriving
        self.disposed_today = max(0.0, avail - self.cfg.storage_cap)
        self.inventory = avail - self.disposed_today
        return arriving

    def quotes(self):
        e = self.ext[self.t]
        return {"price_fast": e[2], "price_slow": e[3], "cap_fast": e[4], "cap_slow": e[5]}

    def pipeline_view(self, H):
        return self.pipeline[self.t:self.t + H].copy()

    def step(self, qf, qs):
        cfg, e, t = self.cfg, self.ext[self.t], self.t
        assert qf >= 0 and qs >= 0 and qf <= e[4] + 1e-7 and qs <= e[5] + 1e-7
        cost = e[2] * qf + e[3] * qs
        assert cost <= cfg.budget * (1 + 1e-7) + 1e-7
        self.pipeline[t + cfg.lead_fast] += qf
        self.pipeline[t + cfg.lead_slow] += qs
        avail = self.inventory
        demand = e[0] * self.customers
        sales = min(demand, avail)
        lost = demand - sales
        inv_end = avail - sales
        revenue = cfg.retail_price * sales
        holding = cfg.holding_cost * inv_end
        disposal_cost = cfg.disposal_penalty * self.disposed_today
        rec = {"day": t, "customers": self.customers, "available": avail, "demand": demand, "sales": sales,
               "lost": lost, "inventory_end": inv_end, "disposal": self.disposed_today, "order_fast": qf,
               "order_slow": qs, "purchase_cost": cost, "revenue": revenue, "holding_cost": holding,
               "disposal_cost": disposal_cost, "profit": revenue - cost - holding - disposal_cost,
               "observed_propensity": demand / self.customers if self.customers > 1e-9 else np.nan}
        self.customers = cfg.rho * self.customers + e[1] - cfg.eta * lost
        self.inventory = inv_end
        self.t += 1
        return rec


def run_closed_loop(cfg: Config, ext: pd.DataFrame, F: np.ndarray, E: np.ndarray, seed: int,
                    model: str, progress=True) -> pd.DataFrame:
    """Daily MPC loop: settle deliveries, build scenarios, solve the LP, execute today's
    order in the exact simulator, observe demand, repeat."""
    raw = ext.to_numpy()
    sim = Simulator(cfg, ext, cfg.t_eval0)
    H = cfg.horizon
    recs = []
    it = range(cfg.t_eval0, cfg.t_eval0 + cfg.n_eval)
    for t in tqdm(it, desc=f"MPC {LABELS[model]} seed {seed}", leave=False, disable=not progress, dynamic_ncols=True):
        sim.settle()
        q = sim.quotes()
        # Observed history interface: completed days < t, quotes up to t.
        prop_hist, ps_hist = raw[:t, 0], raw[:t + 1, 3]
        scen = build_scenarios(cfg, F, E, seed, t)
        assert np.allclose(scen[:, 0, 2:], [q["price_fast"], q["price_slow"], q["cap_fast"], q["cap_slow"]])
        v_c, v_i = terminal_values(cfg, prop_hist, ps_hist)
        plan = solve_plan(cfg, sim.inventory, sim.customers, sim.pipeline_view(H), scen, v_c, v_i)
        if plan.ok:
            qf = float(np.clip(plan.qf, 0, q["cap_fast"]))
            qs = float(np.clip(plan.qs, 0, q["cap_slow"]))
            cost = q["price_fast"] * qf + q["price_slow"] * qs
            if cost > cfg.budget:  # numerical noise only
                qf, qs = qf * cfg.budget / cost, qs * cfg.budget / cost
        else:
            qf = qs = 0.0  # documented fallback: no new orders, physical fulfillment continues
        rec = sim.step(qf, qs)
        if np.isfinite(rec["observed_propensity"]):
            assert abs(rec["observed_propensity"] - raw[t, 0]) < 1e-9  # forecasts saw the same data
        rec.update(seed=seed, model=model, eval_day=t - cfg.t_eval0, lp_ok=plan.ok, lp_status=plan.status,
                   lp_message=plan.message if not plan.ok else "", solve_seconds=plan.solve_seconds,
                   lp_residual=plan.residual, withheld_frac=plan.withheld_frac,
                   withheld_first_day_frac=plan.withheld_first_day_frac, planned_profit=plan.planned_profit,
                   future_order_spread=plan.future_order_spread,
                   pipeline_after=float(sim.pipeline[sim.t:].sum()), v_customer=v_c, v_inventory=v_i)
        recs.append(rec)
    df = pd.DataFrame(recs)
    df["cum_profit"] = df["profit"].cumsum()
    return df


# --------------------------------------------------------------------------- metrics
def forecast_metrics(cfg: Config, ext: pd.DataFrame, F: np.ndarray, E: np.ndarray, seed: int, model: str,
                     cov: dict):
    """MAE (original units) and empirical 80% coverage of the residual-bank predictive
    distribution on evaluation origins. Returns (rows aggregated over each target's horizons,
    rows per horizon and regime: all / event target day / disrupted target day / normal)."""
    raw = ext.to_numpy()
    H = cfg.horizon
    origins = np.arange(cfg.t_eval0, cfg.t_eval0 + cfg.n_eval)
    n = len(origins)
    P, LO, HI = (np.empty((n, H, 6)) for _ in range(3))
    for i, t in enumerate(origins):
        f = F[t - cfg.n_train]
        P[i] = from_model_space(f, cfg)
        dist = from_model_space(f[None] + E[available_bank(cfg, t)], cfg)
        LO[i], HI[i] = np.quantile(dist, 0.1, axis=0), np.quantile(dist, 0.9, axis=0)
    tgt = origins[:, None] + np.arange(H)[None]
    TR = raw[tgt]
    AE, CV = np.abs(P - TR), (TR >= LO) & (TR <= HI)
    ev = cov["event"][tgt] > 0
    dis = (cov["dsd_f"][tgt] > 0) | (cov["dsd_s"][tgt] > 0)
    regimes = {"all": np.ones((n, H), bool), "event": ev, "disrupted": dis, "normal": ~ev & ~dis}
    rows, rows_h = [], []
    for j in range(6):
        hs = target_horizons(j, H)
        ae = AE[:, hs, j]
        rows.append({"seed": seed, "model": model, "target": TARGETS[j], "mae": ae.mean(),
                     "rel_mae": ae.mean() / np.mean(raw[origins[0]:origins[-1] + H, j]),
                     "coverage80": CV[:, hs, j].mean(), "n": ae.size})
        for h in hs:
            for reg, mask in regimes.items():
                mk = mask[:, h]
                if mk.any():
                    rows_h.append({"seed": seed, "model": model, "target": TARGETS[j], "h": int(h), "regime": reg,
                                   "mae": AE[mk, h, j].mean(), "coverage80": CV[mk, h, j].mean(), "n": int(mk.sum())})
    return rows, rows_h


def summarize_seed(df: pd.DataFrame, cfg: Config) -> dict:
    last = df.iloc[-1]
    ordered = df.order_fast.sum() + df.order_slow.sum()
    return {"seed": int(df.seed.iloc[0]), "model": df.model.iloc[0], "profit": df.profit.sum(),
            "final_customers": last.customers_next,
            "lost_frac": df.lost.sum() / df.demand.sum(), "stockout_days": int((df.lost > 1e-6).sum()),
            "mean_inventory": df.inventory_end.mean(), "disposal": df.disposal.sum(),
            "fast_share": df.order_fast.sum() / max(ordered, 1e-9), "customer_days": df.customers.sum(),
            "final_inventory": last.inventory_end, "final_pipeline": last.pipeline_after,
            "fallbacks": int((~df.lp_ok).sum()), "withheld_frac_mean": df.withheld_frac.mean(),
            "withheld_frac_max": df.withheld_frac.max(), "withheld_first_day_max": df.withheld_first_day_frac.max(),
            "material_withholding_days": int((df.withheld_frac > 0.01).sum()),
            "solve_seconds": df.solve_seconds.sum(), "max_lp_residual": df.lp_residual.max(),
            **_regime_summary(df)}


def _regime_summary(df: pd.DataFrame) -> dict:
    """Decision quality on event days and on days when either supplier is disrupted."""
    out = {}
    for reg in ("event_day", "disrupted_day"):
        if reg in df:
            g = df[df[reg].astype(bool)]
            out[f"n_{reg}s"] = len(g)
            out[f"lost_frac_{reg}"] = g.lost.sum() / max(g.demand.sum(), 1e-9)
            out[f"profit_{reg}"] = g.profit.sum()
    return out


# -------------------------------------------------------------------------- plotting
def make_plots(out: Path, daily: pd.DataFrame, summary: pd.DataFrame, info: dict):
    """Main figure: cumulative profit, customer base, paired profit per seed, lost demand."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.fontsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 0.8, "axes.edgecolor": "#888888"})
    seeds = sorted(summary.seed.unique())
    models = [m for m in ["tabpfn", "gbm", "ridge"] if m in set(summary.model)]
    cmp_models = [m for m in ["ridge", "gbm", "tabpfn"] if m in models]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    s = info.get("settings", {})
    fig.suptitle("TabMPC — forecast-driven MPC on a synthetic retailer: TabPFN-3.5 vs baseline forecasters",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.text(0.5, 0.925, f"{len(seeds)} paired seeds × {s.get('n_eval', '?')} days · {s.get('horizon', '?')}-day horizon · "
             f"{s.get('n_scenarios', '?')} scenarios · adaptive (affine) future orders · identical features, LP "
             f"and external paths\nLines = mean across seeds; shaded bands = min–max range across seeds "
             f"(seed variability, not confidence intervals)", ha="center", fontsize=10.5, color="#444444")

    for ax, col, scale, ylab, title in [
            (axes[0, 0], "cum_profit", 1e3, "Cumulative realized profit (thousand $)", "Cumulative operating profit"),
            (axes[0, 1], "customers", 1, "Active customers (start of day)", "Customer base")]:
        for m in models:
            piv = daily[daily.model == m].pivot(index="eval_day", columns="seed", values=col) / scale
            ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(piv.index, piv.mean(axis=1), color=COLORS[m], lw=2, label=LABELS[m],
                    ls="-" if m == "tabpfn" else (0, (5, 2)))
        if col == "cum_profit" and "oracle" in set(daily.model):
            piv = daily[daily.model == "oracle"].pivot(index="eval_day", columns="seed", values=col) / scale
            ax.plot(piv.index, piv.mean(axis=1), color=COLORS["oracle"], lw=1.5, ls=":", label=LABELS["oracle"])
        ax.set_xlabel("Evaluation day")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, loc="upper left" if col == "cum_profit" else "lower left")

    ax = axes[1, 0]
    wide = summary.pivot(index="seed", columns="model", values="profit")[cmp_models] / 1e3
    xs = {m: i for i, m in enumerate(cmp_models)}
    xl = len(cmp_models) - 1
    for sd, r in wide.iterrows():
        ax.plot(range(len(cmp_models)), r.to_numpy(), color="#9a9a9a", lw=1.2, zorder=1)
    # seed labels beside the right-most dots, nudged apart so they never overlap
    order = wide[cmp_models[-1]].sort_values()
    gap = 0.045 * (wide.to_numpy().max() - wide.to_numpy().min() + 1e-9)
    ys = list(order.to_numpy())
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (sd, y0), y in zip(order.items(), ys):
        ax.annotate(f"seed {sd}", (xl, y0), xytext=(xl + 0.06, y), fontsize=9, color="#555555", va="center",
                    textcoords="data")
    for m in cmp_models:
        ax.scatter(np.full(len(wide), xs[m]), wide[m], s=70, color=COLORS[m], edgecolor="white", lw=1.5,
                   zorder=3, label=LABELS[m])
    ticklabels = [LABELS[m].replace("Gradient boosting", "Gradient\nboosting") for m in cmp_models]
    ax.set_xticks(range(len(cmp_models)), ticklabels)
    ax.set_xlim(-0.35, xl + 0.45)
    ax.set_ylabel(f"Profit over {s.get('n_eval', '?')} days (thousand $)")
    title = "Paired total profit per seed"
    if {"tabpfn", "ridge"} <= set(cmp_models):
        d = wide["tabpfn"] - wide["ridge"]
        title += f"\nTabPFN − Ridge: mean {d.mean():+.2f}k$, {int((d > 0).sum())}/{len(d)} seeds positive"
    if {"tabpfn", "gbm"} <= set(cmp_models):
        d2 = wide["tabpfn"] - wide["gbm"]
        title += f"\nTabPFN − Gradient boosting: mean {d2.mean():+.2f}k$, {int((d2 > 0).sum())}/{len(d2)} positive"
    ax.set_title(title, loc="left")

    ax = axes[1, 1]
    wide_l = summary.pivot(index="seed", columns="model", values="lost_frac")[cmp_models] * 100
    for sd, r in wide_l.iterrows():
        ax.plot(range(len(cmp_models)), r.to_numpy(), color="#c8c8c8", lw=1, zorder=1)
    for m in cmp_models:
        ax.scatter(np.full(len(wide_l), xs[m]), wide_l[m], s=60, color=COLORS[m], edgecolor="white", lw=1.5, zorder=3)
        ax.hlines(wide_l[m].mean(), xs[m] - 0.18, xs[m] + 0.18, color=COLORS[m], lw=3, zorder=2)
        ax.annotate(f"mean {wide_l[m].mean():.2f}%", (xs[m] + 0.2, wide_l[m].mean()), fontsize=9.5,
                    va="center", color="#333333")
    ax.set_xticks(range(len(cmp_models)), ticklabels)
    ax.set_xlim(-0.35, xl + 0.55)
    ax.set_ylim(0, max(0.1, 1.25 * float(np.nanmax(wide_l.to_numpy()))))
    ax.set_ylabel("Lost demand / attempted demand (%)")
    ax.set_title("Lost-demand fraction\ndots = seeds, bar = mean, grey lines pair seeds", loc="left")
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=2.5, w_pad=3)
    fig.savefig(out / "comparison.png", dpi=170)
    plt.close(fig)


def make_horizon_plot(out: Path, fmh: pd.DataFrame, info: dict):
    """Supplementary: forecast MAE by lead time (mean over seeds, dots = seeds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = fmh[fmh.regime == "all"]
    models = [m for m in ["ridge", "gbm", "tabpfn"] if m in set(sub.model)]
    targets = [("propensity", "Purchase propensity"), ("cap_fast", "Fast-supplier capacity (units)"),
               ("cap_slow", "Slow-supplier capacity (units)"), ("price_fast", "Fast-supplier price ($)")]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (t, name) in zip(axs.ravel(), targets):
        for m in models:
            g = sub[(sub.target == t) & (sub.model == m)]
            mean = g.groupby("h").mae.mean()
            ax.scatter(g.h + {"ridge": -0.18, "gbm": 0.0, "tabpfn": 0.18}[m], g.mae, s=9, color=COLORS[m], alpha=0.35, lw=0)
            ax.plot(mean.index, mean.values, color=COLORS[m], lw=2, marker="o", ms=4, label=LABELS[m])
        ax.axvspan(0.5, 3.5, color="#999999", alpha=0.08, lw=0)
        ax.set_title(name, loc="left")
        ax.set_ylabel("MAE (original units)")
        ax.set_ylim(bottom=0)
    for ax in axs[1]:
        ax.set_xlabel("Forecast lead time h (days ahead of the decision)")
    axs[0, 0].legend(frameon=False, loc="lower right")
    s = info.get("settings", {})
    fig.suptitle(f"Forecast error by lead time — {len(sub.seed.unique())} evaluation seeds × {s.get('n_eval', '?')} "
                 f"origins (lines = mean, dots = seeds; shaded = lead times 1–3 d, the supplier delays)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "forecast_horizons.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------- reporting
def paired_stats(summary: pd.DataFrame, col: str, a: str = "tabpfn", b: str = "ridge") -> dict:
    w = summary.pivot(index="seed", columns="model", values=col)
    d = (w[a] - w[b]).to_numpy()
    n = len(d)
    out = {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if n > 1 else np.nan,
           "min_diff": float(d.min()), "max_diff": float(d.max()), "n_positive": int((d > 0).sum()), "n": n,
           "per_seed": {int(k): float(v) for k, v in zip(w.index, d)}}
    if n > 1:
        half = student_t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        out["t95_ci"] = [float(d.mean() - half), float(d.mean() + half)]
    return out


def print_report(summary: pd.DataFrame, fm: pd.DataFrame, info: dict, fmh: pd.DataFrame | None = None):
    cols = ["profit", "final_customers", "lost_frac", "mean_inventory", "disposal", "fast_share",
            "customer_days", "final_inventory", "final_pipeline", "fallbacks", "withheld_frac_mean"]
    cols += [c for c in summary.columns if c.startswith(("lost_frac_", "profit_", "n_"))]
    print("\nMean over seeds:")
    print(summary.groupby("model")[cols].mean().T.to_string(float_format=lambda v: f"{v:,.4g}"))
    for key, st in info.get("paired_all", {"tabpfn - ridge": info["paired"]}).items():
        print(f"\nPaired profit diff ({key}):", json.dumps({k: v for k, v in st["profit"].items() if k != "per_seed"}))
    print("\nForecast metrics, all horizons (mean over seeds):")
    print(fm.groupby(["target", "model"])[["mae", "rel_mae", "coverage80"]].mean().unstack().to_string(
        float_format=lambda v: f"{v:.4g}"))
    if fmh is not None and len(fmh):
        sub = fmh[fmh.h.isin([0, 1, 2, 3])]
        print("\nMAE by horizon and regime (mean over seeds):")
        print(sub.groupby(["regime", "target", "h", "model"]).mae.mean().unstack().to_string(
            float_format=lambda v: f"{v:.4g}"))


# ------------------------------------------------------------------------ experiments
def versions() -> dict:
    import matplotlib
    import sklearn
    import tqdm as tq
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "tqdm": tq.__version__, "tabpfn": _pkg_version("tabpfn"), "torch": _pkg_version("torch"),
            "platform": platform.platform(), "machine": platform.machine(), "processor": _cpu_name()}


def _cpu_name():
    try:
        import subprocess
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor()


def evaluate(cfg: Config, models=tuple(MODELS)):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cfg.cache_dir) if cfg.cache_dir else out / "cache"
    t_start = time.time()
    factory = None
    prev_path = out / "run_info.json"
    prev_info = json.loads(prev_path.read_text()) if prev_path.exists() else {}
    info = {"settings": dataclasses.asdict(cfg), "versions": versions(), "model": model_info(cfg)}
    if "tabpfn" in models:
        # Load only if at least one TabPFN forecast is not already cached.
        need = any(not (cache / f"forecast_tabpfn_seed{s}_{forecast_key(cfg, 'tabpfn', s, alphas_for(cfg, 'ridge'))}.npz").exists()
                   for s in cfg.eval_seeds)
        if need:
            factory = TabPFNFactory(cfg)
            info["model"]["tabpfn_model_id"] = factory.model_id
            info["model"]["tabpfn_model_file"] = Path(str(factory.model.model_path)).name
    daily, fms, fmhs, runtime = [], [], [], []
    ext_hashes = {}
    for seed in tqdm(cfg.eval_seeds, desc="seeds", dynamic_ncols=True):
        ext, cov = generate_world(cfg, seed)
        ext_hashes[seed] = hashlib.sha256(ext.to_numpy().tobytes()).hexdigest()[:16]
        T = to_model_space(ext.to_numpy())
        for m in models:
            fc = get_forecasts(cfg, T, m, seed, cache, factory)
            if m == "tabpfn" and "model_id" in fc:
                info["model"]["tabpfn_model_id"] = fc["model_id"]
            E = residual_bank(cfg, T, fc["F"])
            ext_replay = generate_external(cfg, seed)
            assert ext_replay.equals(ext)  # identical paired external path
            t0 = time.perf_counter()
            # The clairvoyant diagnostic plans on the single true future (no scenarios).
            c_run = dataclasses.replace(cfg, deterministic=True, n_scenarios=1) if m == "oracle" else cfg
            df = run_closed_loop(c_run, ext_replay, fc["F"], E, seed, m)
            loop_s = time.perf_counter() - t0
            df["event_day"] = cov["event"][df.day.to_numpy()] > 0
            df["disrupted_day"] = (cov["dsd_f"][df.day.to_numpy()] > 0) | (cov["dsd_s"][df.day.to_numpy()] > 0)
            df["customers_next"] = np.r_[df.customers.to_numpy()[1:], np.nan]
            last = df.iloc[-1]
            df.loc[df.index[-1], "customers_next"] = cfg.rho * last.customers + ext.arrivals.iloc[int(last.day)] - cfg.eta * last.lost
            daily.append(df)
            if m != "oracle":
                r_agg, r_h = forecast_metrics(cfg, ext, fc["F"], E, seed, m, cov)
                fms += r_agg
                fmhs += r_h
            runtime.append({"seed": seed, "model": m, "fit_seconds": fc["fit_seconds"],
                            "predict_seconds": fc["predict_seconds"], "forecast_cached": fc["cached"],
                            "n_refits": fc["n_refits"], "mean_train_rows": fc["mean_train_rows"],
                            "control_loop_seconds": loop_s})
    daily = pd.concat(daily, ignore_index=True)
    summary = pd.DataFrame([summarize_seed(g, cfg) for _, g in daily.groupby(["seed", "model"])])
    rt = pd.DataFrame(runtime)
    summary = summary.merge(rt, on=["seed", "model"])
    fm, fmh = pd.DataFrame(fms), pd.DataFrame(fmhs)
    daily.to_csv(out / "daily.csv", index=False)
    summary.to_csv(out / "seed_summary.csv", index=False)
    fm.to_csv(out / "forecast_metrics.csv", index=False)
    fmh.to_csv(out / "forecast_metrics_by_horizon.csv", index=False)
    info["external_path_hashes"] = ext_hashes
    wall = time.time() - t_start
    from_cache = bool(rt[rt.model != "oracle"].forecast_cached.all())
    info["this_run"] = {"date": time.strftime("%Y-%m-%d"), "wall_seconds": wall, "forecasts_from_cache": from_cache}
    # Provenance of the (expensive) forecasts: a cached replay keeps the original record, so
    # its short wall time never replaces the measured inference time.
    if not from_cache:
        info["forecast_run"] = {"date": time.strftime("%Y-%m-%d"), "wall_seconds_total_run": wall,
                                "versions": info["versions"], "model": info["model"],
                                "note": "forecasts computed in this run"}
    elif "forecast_run" in prev_info:
        info["forecast_run"] = prev_info["forecast_run"]
    else:
        info["forecast_run"] = {"note": "forecasts loaded from cache; original run metadata unavailable"}
    pcols = ["profit", "final_customers", "lost_frac", "customer_days"]
    if {"tabpfn", "ridge"} <= set(models):
        info["paired"] = {c: paired_stats(summary, c) for c in pcols}
        info["paired_all"] = {f"{a} - {b}": {c: paired_stats(summary, c, a, b) for c in pcols}
                              for a, b in [("tabpfn", "ridge"), ("tabpfn", "gbm"), ("gbm", "ridge"),
                                           ("oracle", "ridge"), ("oracle", "tabpfn")] if {a, b} <= set(models)}
    info["runtime_totals"] = summary.groupby("model")[["fit_seconds", "predict_seconds", "solve_seconds"]].sum().to_dict()
    (out / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    make_plots(out, daily, summary, info)
    make_horizon_plot(out, fmh, info)
    if "paired" in info:
        print_report(summary, fm, info, fmh)
    print(f"\nWrote results to {out.resolve()} (wall {wall / 60:.1f} min; forecasts from cache: {from_cache})")
    return daily, summary, fm, info


def replot(cfg: Config):
    out = Path(cfg.out_dir)
    info = json.loads((out / "run_info.json").read_text())
    daily = pd.read_csv(out / "daily.csv")
    summary = pd.read_csv(out / "seed_summary.csv")
    fm = pd.read_csv(out / "forecast_metrics.csv")
    fmh = pd.read_csv(out / "forecast_metrics_by_horizon.csv")
    make_plots(out, daily, summary, info)
    make_horizon_plot(out, fmh, info)
    if "paired" in info:
        print_report(summary, fm, info, fmh)
    print(f"Replotted into {out.resolve()}")


# ------------------------------------------------------------------------ self-checks
def selfchecks(cfg: Config):
    ok = []
    H = cfg.horizon

    # 1) lead-time indexing + inventory conservation + customer arithmetic + exact fulfillment
    c = dataclasses.replace(cfg, n_train=40, n_calib=20, n_eval=40)
    ext = generate_external(c, 999)
    sim = Simulator(c, ext, c.t_eval0)
    rng = np.random.default_rng(0)
    inv0, delivered_expected = sim.inventory, {}
    tot_sales = tot_disp = 0.0
    for k in range(30):
        arrived = sim.settle()
        assert abs(arrived - delivered_expected.pop(sim.t, 0.0) - (c.pipeline0_slow if k in (1, 2) else 0.0)) < 1e-9
        q = sim.quotes()
        qf = rng.uniform(0, 1) * q["cap_fast"] * 0.5
        qs = rng.uniform(0, 1) * q["cap_slow"] * 0.5
        C, e = sim.customers, ext.iloc[sim.t]
        rec = sim.step(qf, qs)
        delivered_expected[rec["day"] + c.lead_fast] = delivered_expected.get(rec["day"] + c.lead_fast, 0) + qf
        delivered_expected[rec["day"] + c.lead_slow] = delivered_expected.get(rec["day"] + c.lead_slow, 0) + qs
        assert abs(rec["sales"] - min(rec["demand"], rec["available"])) < 1e-9
        assert abs(rec["demand"] - e.propensity * C) < 1e-9
        assert abs(sim.customers - (c.rho * C + e.arrivals - c.eta * rec["lost"])) < 1e-9
        tot_sales += rec["sales"]
        tot_disp += rec["disposal"]
    assert abs(inv0 + sim.delivered_total - tot_sales - tot_disp - sim.inventory) < 1e-6
    assert 0 <= c.eta <= c.rho <= 1
    ok.append("lead times, inventory conservation, customer-loss arithmetic, exact sales=min(demand, stock)")

    # 2) causal features and labels
    ext = generate_external(cfg, 999)
    T = to_model_space(ext.to_numpy())
    for o in [MIN_ORIGIN, 100, 400]:
        Tm = T.copy()
        Tm[o:, ~IS_QUOTE] = np.nan   # unknown at decision time o
        Tm[o + 1:, IS_QUOTE] = np.nan
        a, b = origin_features(T, np.array([o])), origin_features(Tm, np.array([o]))
        assert np.isfinite(b).all() and np.allclose(a, b)
    for j in range(6):
        oo, hh = training_rows(400, j, cfg, 999)
        assert (oo + hh <= (400 if IS_QUOTE[j] else 399)).all() and (oo < 400).all()
    ok.append("features use only data observable at the origin; training labels observed by refit origin")

    # 3) scenarios: day-0 quotes known exactly; shared indices only use complete trajectories
    c = dataclasses.replace(cfg, n_train=120, n_calib=42, n_eval=14, window=90, context_cap=600)
    ext = generate_external(c, 5)
    T = to_model_space(ext.to_numpy())
    F = rolling_forecasts(c, T, "ridge", 5, desc="selfcheck ridge")["F"]
    E = residual_bank(c, T, F)
    for t in [c.t_eval0, c.t_eval0 + 5]:
        idx = scenario_indices(c, 5, t)
        assert (c.n_train + idx + H - 1 <= t - 1).all()
        sc = build_scenarios(c, F, E, 5, t)
        assert np.allclose(sc[:, 0, 2:], ext.to_numpy()[t, 2:])
    ok.append("scenario first-day quotes equal known quotes; residual trajectories complete before use")

    # 4) paired external paths identical and deterministic closed loop
    assert generate_external(c, 5).equals(ext) and not generate_external(c, 6).equals(ext)
    d1 = run_closed_loop(c, ext, F, E, 5, "ridge", progress=False)
    F2 = rolling_forecasts(c, T, "ridge", 5, desc="selfcheck ridge")["F"]
    d2 = run_closed_loop(c, ext, F2, residual_bank(c, T, F2), 5, "ridge", progress=False)
    assert d1.drop(columns="solve_seconds").equals(d2.drop(columns="solve_seconds"))
    assert d1.lp_ok.all() and (d1.lp_residual < 1e-5).all()
    ok.append(f"identical paired paths; deterministic reproducibility; {len(d1)} LP solves feasible (status 0)")

    # 5) checkable planning example: constant demand 50/day, empty stock, no pipeline.
    # Fast must cover day 1-2 demand (slow cannot arrive before day 3); slow is cheaper
    # and covers day 3 onward -> today's orders: fast = 50 (for day 1), slow = 50 (for day 3).
    c = dataclasses.replace(cfg, rho=1.0, eta=0.0, budget=1e6, storage_cap=1e6, horizon=6)
    scen = np.zeros((1, 6, 6))
    scen[..., 0], scen[..., 1], scen[..., 2], scen[..., 3], scen[..., 4], scen[..., 5] = 0.5, 0, 6.0, 5.0, 1e3, 1e3
    plan = solve_plan(c, 0.0, 100.0, np.zeros(6), scen, v_c=0.0, v_i=0.0, controller="shared")
    assert plan.ok and abs(plan.qf - 50) < 1e-6 and abs(plan.qs - 50) < 1e-6, plan
    # With a fast capacity of 20 the missing 30 units on day 1 are simply lost.
    scen[..., 4] = 20.0
    plan = solve_plan(c, 0.0, 100.0, np.zeros(6), scen, v_c=0.0, v_i=0.0, controller="shared")
    assert plan.ok and abs(plan.qf - 20) < 1e-6 and abs(plan.qs - 50) < 1e-6, plan
    ok.append("deterministic planning example: fast=50, slow=50 (and fast=20 when capacity-limited)")

    # 5b) affine controller (the default): contains the shared open-loop plan (never worse), reduces to it with one
    # scenario, and uses scenario-specific capacity. Demand is 60/day, stock is empty, today's fast
    # capacity is 60 and tomorrow's is 20 or 100 (equally likely): the shared plan can buy only 20
    # tomorrow for day 2, the affine plan buys 60 in the scenario where 100 is available.
    for nsc in (1, 2):
        sc = np.zeros((nsc, 6, 6))
        sc[..., 0], sc[..., 2], sc[..., 3], sc[..., 4], sc[..., 5] = 0.6, 6.0, 5.0, 1e3, 1e3
        sc[:, 0, 4] = 60.0  # today's fast capacity covers only day 1
        if nsc == 2:
            sc[0, 1, 4], sc[1, 1, 4] = 20.0, 100.0
        pl_s = solve_plan(c, 0.0, 100.0, np.zeros(6), sc, v_c=0.0, v_i=0.0, controller="shared")
        pl_a = solve_plan(c, 0.0, 100.0, np.zeros(6), sc, v_c=0.0, v_i=0.0, controller="affine")
        assert pl_s.ok and pl_a.ok and pl_a.objective >= pl_s.objective - 1e-6
        if nsc == 1:
            assert abs(pl_a.objective - pl_s.objective) < 1e-6
        else:
            assert pl_a.objective > pl_s.objective + 1.0 and pl_a.future_order_spread > 1.0
    ext = generate_external(cfg, 999)
    T = to_model_space(ext.to_numpy())
    cq = dataclasses.replace(cfg, n_train=120, n_calib=42, n_eval=14, window=90, context_cap=600)
    Fr = rolling_forecasts(cq, T, "ridge", 999, desc="selfcheck ridge")["F"]
    Er = residual_bank(cq, T, Fr)
    for t in (cq.t_eval0, cq.t_eval0 + 7):
        sc = build_scenarios(cq, Fr, Er, 999, t)
        pipe = np.r_[0.0, 150.0, 150.0, np.zeros(H - 3)]
        a_ = solve_plan(cq, 150.0, 1000.0, pipe, sc, 20.0, 4.0, "affine")
        s_ = solve_plan(cq, 150.0, 1000.0, pipe, sc, 20.0, 4.0, "shared")
        assert a_.ok and s_.ok and a_.objective >= s_.objective - 1e-6 * max(1, abs(s_.objective))
    ok.append("affine controller: >= shared objective, equal with one scenario, adapts to scenario capacity")

    # 6) observable covariates: causal, deterministic, disruption status consistent
    ext_r, cov = generate_world(cfg, 999)
    ext_r2, cov2 = generate_world(cfg, 999)
    assert ext_r.equals(ext_r2) and all(np.array_equal(cov[k], cov2[k]) for k in cov)
    for o in [MIN_ORIGIN, 200, 400]:
        masked = {k: v.astype(float).copy() for k, v in cov.items()}
        for k in ("cong_f", "cong_s", "dsd_f", "dsd_s"):  # announced today
            masked[k][o + 1:] = np.nan
        masked["temp"][o:] = np.nan                         # actual weather known up to yesterday
        a_ = covariate_origin_features(cov, np.array([o]))
        b_ = covariate_origin_features(masked, np.array([o]))
        assert np.isfinite(b_).all() and np.allclose(a_, b_)
    onsets = np.flatnonzero(cov["dsd_s"] == 1)
    assert len(onsets) > 0 and np.all(ext_r.cap_slow.to_numpy()[onsets] < cfg.cap_slow_base)
    ok.append("covariate features causal, deterministic, disruption status consistent")

    # 7) LP variables continuous: linprog is called without `integrality`
    import inspect
    assert "integrality" not in inspect.getsource(solve_plan).split('"""')[2]
    ok.append("planning variables continuous (no integrality)")
    print("Self-checks passed:")
    for s in ok:
        print("  ✓", s)


# -------------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    p.add_argument("--out-dir", default=None, help=f"output directory (default {d.out_dir}; quick: results_quick)")
    p.add_argument("--cache-dir", default=None, help="forecast cache directory (default <out-dir>/cache)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help=f"evaluation seeds (default {d.eval_seeds})")
    p.add_argument("--n-eval", type=int, default=None, help=f"evaluated days per seed (default {d.n_eval})")
    p.add_argument("--n-train", type=int, default=None, help=f"training history days (default {d.n_train})")
    p.add_argument("--n-calib", type=int, default=None, help=f"calibration days (default {d.n_calib})")
    p.add_argument("--horizon", type=int, default=None, help=f"forecast/planning horizon (default {d.horizon})")
    p.add_argument("--scenarios", type=int, default=None, help=f"scenario count (default {d.n_scenarios})")
    p.add_argument("--refit-every", type=int, default=None, help=f"refit interval days (default {d.refit_every})")
    p.add_argument("--window", type=int, default=None, help=f"rolling training window (default {d.window})")
    p.add_argument("--context-cap", type=int, default=None, help=f"max training rows (default {d.context_cap})")
    p.add_argument("--device", default=None, help="auto|cpu|mps|cuda (default auto)")
    p.add_argument("--tabpfn-version", default=None, choices=["v3.5", "v3.5-fast"])
    p.add_argument("--tabpfn-n-estimators", type=int, default=None, help=f"default {d.tabpfn_n_estimators}")
    p.add_argument("--models", nargs="+", default=None, choices=MODELS, help=f"default: {' '.join(MODELS)}")
    p.add_argument("--deterministic", action="store_true", help="single point-forecast scenario (debug)")
    p.add_argument("--quick", action="store_true", help="smoke run: 1 seed, 28 eval days, small context")
    p.add_argument("--replot-only", action="store_true", help="figures/tables from saved results, no inference")
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args(argv)


def config_from_args(a) -> Config:
    cfg = Config()
    if a.quick:
        cfg = dataclasses.replace(cfg, out_dir="results_quick", eval_seeds=cfg.eval_seeds[:1], n_calib=42, n_eval=28,
                                  n_scenarios=16, context_cap=1000)
    upd = {"out_dir": a.out_dir, "cache_dir": a.cache_dir, "eval_seeds": tuple(a.seeds) if a.seeds else None,
           "n_eval": a.n_eval, "n_train": a.n_train, "n_calib": a.n_calib, "horizon": a.horizon,
           "n_scenarios": a.scenarios, "refit_every": a.refit_every, "window": a.window,
           "context_cap": a.context_cap, "device": a.device, "tabpfn_version": a.tabpfn_version,
           "tabpfn_n_estimators": a.tabpfn_n_estimators}
    cfg = dataclasses.replace(cfg, **{k: v for k, v in upd.items() if v is not None})
    if a.deterministic:
        cfg = dataclasses.replace(cfg, deterministic=True, n_scenarios=1)
    assert cfg.n_calib >= cfg.horizon + 1, "calibration must yield at least one complete residual trajectory"
    return cfg


def main(argv=None):
    a = parse_args(argv)
    cfg = config_from_args(a)
    if a.selfcheck:
        selfchecks(cfg)
    elif a.replot_only:
        replot(cfg)
    else:
        evaluate(cfg, tuple(a.models) if a.models else tuple(MODELS))


if __name__ == "__main__":
    main()
