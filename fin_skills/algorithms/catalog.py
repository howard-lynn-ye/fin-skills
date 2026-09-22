"""Reviewed capability inventory. Rankings and sample floors are local routing policy.

External packages stay optional. A catalog-only entry is never advertised as runnable.
Existing skill texts carry the fuller implementation and licence caveats.
"""
from .core import Algorithm, Registry
from . import runtime as r
from . import extended as e
from . import technical as t
from . import strategy as s

SOURCES = {
    "PyPortfolioOpt": "https://pyportfolioopt.readthedocs.io/en/latest/OtherOptimizers.html",
    "skfolio": "https://skfolio.org/",
    "Riskfolio-Lib": "https://riskfolio-lib.readthedocs.io/en/latest/",
    "statsmodels": "https://www.statsmodels.org/stable/tsa.html",
    "arch": "https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html",
    "scikit-learn": "https://scikit-learn.org/stable/supervised_learning.html",
    "lightgbm": "https://lightgbm.readthedocs.io/en/latest/",
    "xgboost": "https://xgboost.readthedocs.io/en/stable/",
    "statsforecast": "https://nixtlaverse.nixtla.io/statsforecast/docs/models/autoarima.html",
    "QuantLib": "https://www.quantlib.org/",
    "hmmlearn": "https://hmmlearn.readthedocs.io/en/latest/",
    "ruptures": "https://centre-borelli.github.io/ruptures-docs/",
    "pyqlib": "https://qlib.readthedocs.io/en/latest/component/model.html",
}


def default_registry() -> Registry:
    """Return a fresh registry so application extensions cannot mutate other callers."""
    registry = Registry()

    def add(id, name, task, inputs, objectives, *, handler=None, library="fin-skills",
            module=None, tags=(), capabilities=(), complexity="low", minimum=0,
            priority=0, skill="", source="", caveat="Validate on later, unseen data.",
            check=None, verified_on="2026-09-14"):
        registry.register(Algorithm(
            id=id, name=name, task=task, library=library, inputs=tuple(inputs.split()),
            objectives=tuple(objectives.split()), tags=tuple(tags), capabilities=tuple(capabilities),
            complexity=complexity, min_observations=minimum, priority=priority,
            module=module, skill=skill, source=source or SOURCES.get(library, f"skill:{skill}"),
            caveat=caveat, verified_on=verified_on), handler,
            preflight=check or (r.preflight(id, task) if handler else None))

    portfolio = dict(task="portfolio", inputs="asset_returns", skill="portfolio-optimizers",
                     capabilities=("long_only",))
    for id, name, objectives, tags, minimum, priority in (
        ("equal_weight", "Equal weight", "allocation", ("simple", "small_sample"), 1, 1),
        ("inverse_volatility", "Inverse volatility", "allocation risk_balance",
         ("simple", "unequal_volatility"), 2, 3),
        ("min_variance", "Long-only minimum variance", "allocation min_variance",
         ("correlated_assets",), 20, 2),
        ("hrp", "Hierarchical risk parity", "allocation risk_balance",
         ("correlated_assets", "robust"), 20, 2),
    ):
        add(id, name, objectives=objectives, tags=tags, minimum=minimum, priority=priority,
            handler=r.portfolio(id), complexity="medium" if id == "hrp" else "low",
            caveat=("Explicit linkage required. Weights are fit on the supplied history; "
                    "apply only to subsequent returns." if id == "hrp" else
                    "Long-only static allocation fit on supplied returns; evaluate on later returns."),
            **portfolio)
    for backend, library, skill in (("pypfopt", "PyPortfolioOpt", "lib-pyportfolioopt"),
                                    ("skfolio", "skfolio", "lib-skfolio"),
                                    ("riskfolio", "Riskfolio-Lib", "lib-riskfolio")):
        for method in ("hrp", "min_variance"):
            add(f"{backend}_{method}", f"{library} {method}", "portfolio", "asset_returns",
                "allocation risk_balance" if method == "hrp" else "allocation min_variance",
                handler=r.portfolio(method, backend), library=library, module=backend,
                tags=("correlated_assets",), capabilities=("long_only",), complexity="medium",
                minimum=20, skill=skill,
                caveat="Existing optimizer bridge enforces return units, linkage and solver checks.")

    for method, name, tags, minimum, priority in (
        ("naive", "Last-observation forecast", ("simple", "price_level"), 1, 3),
        ("mean", "Historical-mean forecast", ("simple", "stationary"), 2, 1),
        ("drift", "Random walk with drift", ("trend",), 2, 0),
        ("seasonal_naive", "Seasonal naive forecast", ("seasonal",), 2, 1),
    ):
        add(method, name, "forecast", "series seasonal_period" if method == "seasonal_naive"
            else "series", "forecast", handler=r.forecast(method), tags=tags,
            minimum=minimum, priority=priority, skill="time-series-forecasting-models",
            caveat="Forecast begins after the final input row; series must be chronological.")
    add("arima", "ARIMA(1,1,0)", "forecast", "series", "forecast", library="statsmodels",
        module="statsmodels", handler=r.forecast("arima"), minimum=30, complexity="medium",
        tags=("autocorrelation",), skill="time-series-forecasting-models",
        caveat="Adapter fixes order=(1,1,0); no automatic order search or performance guarantee.")
    add("auto_arima", "AutoARIMA", "forecast", "series seasonal_period", "forecast",
        library="statsforecast", module="statsforecast", handler=e.auto_arima,
        complexity="medium", minimum=30, tags=("seasonal",),
        skill="time-series-forecasting-models", caveat="AICc order selection within training history.")

    for method, tags, priority in (("historical", ("simple",), 1),
                                    ("ewma", ("recent_changes",), 2)):
        add(f"{method}_volatility", f"{method.upper()} volatility", "volatility", "returns",
            "volatility", handler=r.volatility(method), tags=tags, minimum=2,
            priority=priority, skill="volatility-models",
            caveat="Per-period volatility unless periods_per_year is explicit; EWMA assumes zero mean.")
    for id, name, tags in (("garch", "GARCH", ("volatility_clustering",)),
                            ("egarch", "EGARCH", ("asymmetry",))):
        add(id, name, "volatility", "returns", "volatility", library="arch", minimum=100,
            module="arch", handler=e.arch_volatility(id),
            complexity="medium", tags=tags, skill="lib-arch",
            caveat="Zero-mean one-step forecast; percentage fitting converted to decimal volatility.")
    for method, tags in (("historical", ("nonparametric", "fat_tails")),
                         ("normal", ("gaussian", "simple"))):
        add(f"{method}_var_es", f"{method.title()} VaR and ES", "risk", "returns", "tail_risk",
            handler=r.risk(method), minimum=20, tags=tags, skill="portfolio-and-risk",
            priority=2 if method == "historical" else 0,
            caveat="Positive numbers denote losses. Small tails make empirical estimates unstable.")
    add("momentum", "Time-series momentum", "signal", "returns", "trend_signal",
        handler=r.signal("momentum"), minimum=21, tags=("trend",), priority=2,
        capabilities=("lagged",), skill="trend-following-models",
        caveat="Returns lagged positions, with missing warmup rows; this is not a backtest.")
    add("ma_crossover", "Moving-average crossover", "signal", "prices", "trend_signal",
        handler=r.signal("ma_crossover"), minimum=21, tags=("trend", "simple"),
        capabilities=("lagged",), skill="trend-following-models",
        caveat="Returns lagged positions. Price history must use a consistent adjustment convention.")

    for id, name, task, tags, minimum, priority in (
        ("ridge", "Ridge regression", "regression", ("linear", "regularized", "small_sample"), 10, 3),
        ("random_forest_regression", "Random forest regression", "regression", ("nonlinear",), 100, 1),
        ("logistic", "Logistic regression", "classification", ("linear", "interpretable"), 20, 3),
        ("random_forest_classification", "Random forest classifier", "classification", ("nonlinear",), 100, 1),
    ):
        add(id, name, task, "X y X_predict", "predict", handler=r.supervised(id),
            library="scikit-learn", module="sklearn", tags=tags, minimum=minimum,
            priority=priority, complexity="medium", skill="factor-and-timeseries-research",
            caveat="Training rows only in fit; caller must supply point-in-time features and labels.")
    for library in ("lightgbm", "xgboost"):
        for task in ("regression", "classification"):
            add(f"{library}_{task}", f"{library} {task}", task, "X y X_predict", "predict",
                module=library, handler=r.supervised(f"{library}_{task}"),
                library=library, tags=("nonlinear", "tabular"), minimum=100, complexity="medium",
                skill="factor-and-timeseries-research", caveat="Fixed seeded baseline; no parameter search.")
    add("qlib_alpha", "Qlib linear alpha model", "regression", "X y X_predict", "predict",
        library="pyqlib", module="qlib", handler=r.supervised("qlib_alpha"), tags=("panel",),
        minimum=100, complexity="high", skill="lib-qlib",
        caveat="Qlib ridge on precomputed features; no data provider or Alpha158 generation. No normalization.")

    add("twap", "TWAP schedule", "execution", "shares n_bins", "schedule", handler=r.execution("twap"),
        tags=("simple",), priority=1, skill="execution-algorithms",
        caveat="Returns a schedule only; does not place orders or model liquidity.")
    add("vwap", "Forecast-volume VWAP schedule", "execution", "shares volume_forecast", "schedule",
        handler=r.execution("vwap"), tags=("volume_profile",), priority=3, skill="execution-algorithms",
        caveat="Volume must be a forecast available before scheduling, not future realized volume.")
    for id, name, exercise in (("black_scholes", "Black-Scholes-Merton", "european"),
                                ("american_crr", "American CRR tree", "american")):
        add(id, name, "pricing", "option", "price", handler=r.pricing(id),
            capabilities=(exercise,), tags=("constant_volatility",),
            complexity="medium" if exercise == "american" else "low", priority=1,
            skill="option-pricing-models",
            caveat="Annual decimal rates/volatility and year-fraction expiry; continuous dividends only.")
    add("quantlib_heston", "Heston pricing engine", "pricing", "option heston_parameters", "price",
        module="QuantLib", handler=e.heston,
        library="QuantLib", capabilities=("european",), tags=("stochastic_volatility",),
        complexity="high", skill="lib-quantlib",
        caveat="Requires explicit dates and variance parameters; option.sigma unused. No calibration.")
    add("engle_granger", "Engle-Granger cointegration test", "stat_arb", "X", "cointegration",
        module="statsmodels", handler=e.engle_granger,
        library="statsmodels", minimum=100, tags=("pairs",), complexity="medium",
        skill="stat-arb-cointegration",
        source="https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html",
        caveat="Both series assumed I(1); in-sample test alone does not establish a tradable pair.")
    add("gaussian_hmm", "Gaussian hidden Markov model", "regime", "X", "regime",
        module="hmmlearn", handler=e.gaussian_hmm,
        library="hmmlearn", minimum=100, complexity="high", tags=("latent_states",),
        skill="regime-detection", caveat="Retrospective full-sample states; must not be used as historical signals.")
    add("change_points", "Change-point detection", "regime", "series", "breaks",
        module="ruptures", handler=e.change_points,
        library="ruptures", minimum=30, complexity="medium", tags=("structural_breaks",),
        skill="regime-detection", caveat="Offline PELT L2 break detection is retrospective.")
    for method, name, objective, skill in (
        ("donchian_breakout", "Close-channel breakout", "trend_signal", "trend-following-models"),
        ("bollinger_reversion", "Bollinger mean reversion", "reversion_signal", "trend-following-models"),
        ("rsi_reversion", "Simple rolling RSI reversion", "reversion_signal", "trend-following-models"),
        ("macd", "MACD signal-line crossover", "trend_signal", "trend-following-models"),
        ("vol_target_momentum", "Volatility-targeted momentum", "trend_signal", "trend-following-models"),
    ):
        add(method, name, "signal", "prices", objective, handler=t.signal(method),
            check=t.signal_check(method), minimum=t.warmup(method, t.SIGNAL_DEFAULTS[method]) + 1,
            tags=("mean_reversion",) if objective == "reversion_signal" else ("trend",),
            capabilities=("lagged",), skill=skill, verified_on="2026-09-21",
            source="fin_skills/algorithms/technical.py",
            caveat="Local baseline definition; one-bar lag, explicit warmup, no costs or fills modeled.")
    add("cross_sectional_momentum", "Positive-momentum top-k allocation", "portfolio", "asset_returns",
        "allocation momentum", handler=t.cross_sectional_momentum, check=t.cross_sectional_check,
        minimum=60, tags=("trend",), capabilities=("long_only",), skill="trend-following-models",
        source="fin_skills/algorithms/technical.py", verified_on="2026-09-21",
        caveat="Apply weights to later returns; unallocated balance is cash. Requires a point-in-time universe.")
    add("rolling_market_state", "Causal rolling market state", "regime", "prices", "regime",
        handler=s.regime_adapter, check=lambda data, p: s.regime_settings(p), minimum=1,
        tags=("causal",), skill="regime-detection", source="fin_skills/algorithms/strategy.py",
        verified_on="2026-09-21",
        caveat="Unknown during warmup; heuristic labels at close t, usable for decisions after t only.")
    return registry
