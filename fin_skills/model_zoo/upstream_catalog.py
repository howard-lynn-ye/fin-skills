"""Allowlisted native methods. Names are backend identifiers, not implementations."""
from ._talib_catalog import FUNCTIONS as TA_FUNCTIONS

SKLEARN = {
    "elastic_net": ("linear_model", "ElasticNet", "regression", True),
    "svr": ("svm", "SVR", "regression", True),
    "svc": ("svm", "SVC", "classification", True),
    "knn": ("neighbors", "KNeighborsRegressor", "regression", True),
    "extra_trees": ("ensemble", "ExtraTreesRegressor", "regression", False),
    "hist_gradient_boosting": ("ensemble", "HistGradientBoostingRegressor", "regression", False),
    "gaussian_process": ("gaussian_process", "GaussianProcessRegressor", "regression", True),
    "naive_bayes": ("naive_bayes", "GaussianNB", "classification", False),
    "mlp": ("neural_network", "MLPRegressor", "regression", True),
    "isotonic_calibration": ("isotonic", "IsotonicRegression", "calibration", False),
    "kmeans_regime": ("cluster", "KMeans", "clustering", True),
    "gmm_regime": ("mixture", "GaussianMixture", "clustering", True),
    "dbscan_regime": ("cluster", "DBSCAN", "clustering", True),
    "spectral_regime": ("cluster", "SpectralClustering", "clustering", True),
    "ica_factors": ("decomposition", "FastICA", "decomposition", True),
    "nmf_factors": ("decomposition", "NMF", "decomposition", False),
}
TIME_SERIES = {
    "sarimax": ("forecast", "series"), "var": ("forecast", "X"),
    "vecm": ("forecast", "X"), "unobserved_components": ("forecast", "series"),
    "dynamic_factor": ("forecast", "X"), "exponential_smoothing": ("forecast", "series"),
    "markov_regression": ("regime", "series"),
    "markov_autoregression": ("regime", "series"),
    "stl_decomposition": ("decomposition", "series"), "adf_test": ("diagnostic", "series"),
    "kpss_test": ("diagnostic", "series"), "granger_test": ("diagnostic", "X"),
}
RISK_MEASURES = {
    "mean_mad": "MAD", "mean_semivariance": "MSV", "mean_cvar": "CVaR",
    "mean_evar": "EVaR", "mean_cdar": "CDaR", "mean_edar": "EDaR", "mean_ulcer": "UCI",
}
RISK_SPECIAL = ("worst_case_allocation", "risk_budget_allocation", "factor_risk_budget",
                "owa_allocation")
FRONTIERS = {"semivariance_frontier": "EfficientSemivariance",
             "cvar_frontier": "EfficientCVaR", "cdar_frontier": "EfficientCDaR"}
CVX = {"single_period_optimization": "SinglePeriodOptimization",
       "multi_period_optimization": "MultiPeriodOptimization",
       "periodic_rebalance": "PeriodicRebalance", "proportional_rebalance": "ProportionalRebalance",
       "adaptive_rebalance": "AdaptiveRebalance", "hold_policy": "Hold",
       "uniform_allocation": "Uniform"}
COMPARISONS = {"spa_selection": "SPA", "stepm_selection": "StepM",
               "model_confidence_set": "MCS"}
SPLITS = {"temporal_cv": "TimeSeriesSplit", "grouped_cv": "GroupKFold"}
NEURAL = {"nbeats": "NBEATS", "tft": "TFT", "deepar": "DeepAR",
          "itransformer": "iTransformer", "timemixer": "TimeMixer"}
RL = {"a2c": "A2C", "ddpg": "DDPG", "dqn": "DQN", "td3": "TD3"}


def upstream_cards(available):
    cards = []

    def add(id_, task, library, module, inputs, operations, source, note, **extra):
        cards.append(dict(id=id_, name=id_, task=task, library=library, adapter="upstream",
            inputs=inputs.split(), operations=operations.split(),
            status="ready" if available(module) else "missing_dependency",
            install="pip install 'fin-skills[methods]' (deep/rl extras for neural/RL)",
            license="Core adapter MIT; upstream library and data licenses apply",
            stage="adapter", pretrained=False, source=source, caveat=note,
            verified_on="2026-09-21", json_run="run" in operations.split(), **extra))

    for id_, meta in TA_FUNCTIONS.items():
        add(id_, "technical", "TA-Lib", "talib", "inputs", "run save load",
            "https://ta-lib.github.io/ta-lib-python/abstract.html",
            "Native calculation only, not a strategy. Leading NaN warmup is preserved; "
            "pattern codes are not calibrated probabilities.",
            native_function=meta["function"], input_fields=meta["inputs"],
            output_fields=list(meta["outputs"]), defaults=dict(meta["parameters"]))
    for id_, (_, _, task, _) in SKLEARN.items():
        operations = "fit predict run save load"
        if id_ in ("dbscan_regime", "spectral_regime"):
            operations = "fit run save load"  # transductive; no invented out-of-sample predict
        add(id_, task, "scikit-learn", "sklearn", "X" if task in
            ("clustering", "decomposition") else "X y", operations,
            "https://scikit-learn.org/stable/user_guide.html",
            "Train-only preprocessing. Cluster labels have no bull/bear semantics. "
            "run returns fit diagnostics; use predict on held-out data where supported.")
    for id_, (task, field) in TIME_SERIES.items():
        operations = "fit predict save load" if task == "forecast" else "run save load"
        add(id_, task, "statsmodels", "statsmodels", field, operations,
            "https://www.statsmodels.org/stable/tsa.html",
            "Full-sample decomposition/regime outputs are retrospective. "
            "Forecasts use only supplied training history; future exogenous data must be declared.")
    for id_ in (*RISK_MEASURES, *RISK_SPECIAL):
        add(id_, "portfolio", "Riskfolio-Lib", "riskfolio", "asset_returns", "run save load",
            "https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html",
            "Native optimization on caller's training returns. Explicit budgets/constraints; "
            "solver failure is not replaced by fallback equal weights.")
    for id_ in FRONTIERS:
        add(id_, "portfolio", "PyPortfolioOpt", "pypfopt", "asset_returns", "run save load",
            "https://pyportfolioopt.readthedocs.io/en/latest/GeneralEfficientFrontier.html",
            "Native minimum-risk or efficient-return solution, not an automatically selected frontier point.")
    for id_ in CVX:
        add(id_, "allocation", "cvxportfolio", "cvxportfolio", "holdings market_data t",
            "run save load", "https://www.cvxportfolio.com/en/stable/policies.html",
            "Native policy decision. market_data is a caller-supplied Cvxportfolio object; "
            "orders are returned, never sent. Python API only.")
        cards[-1]["json_run"] = False
    for id_ in COMPARISONS:
        add(id_, "evaluation", "arch", "arch", "losses", "run save load",
            "https://arch.readthedocs.io/en/latest/multiple-comparison/multiple-comparison_examples.html",
            "Caller supplies aligned losses; record all tried candidates. Bootstrap p-values are not profit estimates.")
    for id_ in SPLITS:
        add(id_, "evaluation", "scikit-learn", "sklearn", "X", "run save load",
            "https://scikit-learn.org/stable/modules/cross_validation.html",
            "Returns native split indices; grouped splitting is not temporal, and time splitting is not label purging.")
    for id_, upstream in NEURAL.items():
        add(id_, "forecast", "neuralforecast", "neuralforecast", "unique_id ds y",
            "fit predict save load", "https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html",
            "Native NeuralForecast model; caller supplies calendar/history. No pretrained weights.")
        cards[-1]["adapter"] = "sequence"
    for id_ in RL:
        add(id_, "reinforcement_learning", "stable-baselines3", "stable_baselines3", "env",
            "fit predict save load", "https://stable-baselines3.readthedocs.io/en/master/guide/algos.html",
            "Requires caller's Gymnasium environment with compatible actions. No broker or financial reward assumptions.")
        cards[-1]["adapter"] = "rl"
    return cards
