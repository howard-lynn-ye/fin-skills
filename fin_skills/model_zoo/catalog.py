"""Capability cards augment the existing registry, without changing its routing scores."""
from importlib.util import find_spec


NUMERICAL = {
    "kalman_filter": ("filtering", ["y", "Z", "T", "Q", "H", "a0", "P0"],
        "Fixed-parameter forward filter; no retrospective smoother or parameter fit."),
    "ledoit_wolf_covariance": ("covariance", ["asset_returns"],
        "Training-sample covariance shrinkage, not portfolio weights."),
    "ewma_covariance": ("covariance", ["asset_returns"],
        "One terminal covariance from supplied history, not historical trading signals."),
    "pca_covariance": ("covariance", ["asset_returns"],
        "Training-sample PCA covariance, with caller-selected factor count."),
    "nelson_siegel": ("yield_curve", ["maturities", "yields"],
        "Cross-sectional yield-curve fit; no forecast of future yields."),
    "svensson": ("yield_curve", ["maturities", "yields"],
        "Cross-sectional yield-curve fit; parameters may be weakly identified."),
    "merton_credit": ("credit", ["E", "sigma_E", "D", "r", "T"],
        "Structural credit inversion; probability is risk-neutral, not physical."),
    "svi_surface": ("vol_surface", ["log_moneyness", "total_variance"],
        "One SVI slice; fitting does not certify absence of static arbitrage."),
}


def _available(module):
    try:
        return find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def model_catalog(task=None):
    """Return fresh JSON-compatible cards; ready means discoverable, not validated profit."""
    from fin_skills.algorithms import catalog
    cards = []
    for row in catalog():
        fitted = row["task"] in ("forecast", "regression", "classification")
        cards.append(dict(row, adapter="algorithm", operations=(
            ["run", "fit", "predict", "save", "load"] if fitted else ["run", "save", "load"]),
            pretrained=False, install="Install the backend extra documented in MODEL_USAGE.md",
            license="Core MIT; external backend license applies where used",
            stage="adapter", json_run=True))
    for name, (family, inputs, note) in NUMERICAL.items():
        cards.append(dict(id=name, task=family, adapter="numerical", inputs=inputs,
            operations=["run", "save", "load"], status="ready", library="fin-skills",
            pretrained=False, install="Base dependencies", license="MIT", stage="adapter",
            source="fin_skills/models", caveat=note, json_run=True))
    for name, family, module, operations, install, source, caveat in (
        *((name, "forecast", "neuralforecast", ["fit", "predict", "save", "load"],
           "pip install 'fin-skills[deep]'",
           "https://nixtlaverse.nixtla.io/neuralforecast/models." + upstream + ".html",
           "Native NeuralForecast backend; explicit regular history and horizon; no pretrained weights.")
          for name, upstream in (("lstm_forecast", "lstm"),
              ("transformer_forecast", "vanillatransformer"),
              ("patchtst_forecast", "patchtst"), ("nhits_forecast", "nhits"))),
        ("ppo", "reinforcement_learning", "stable_baselines3", ["fit", "predict", "save", "load"],
         "pip install 'fin-skills[rl]'", "https://stable-baselines3.readthedocs.io/en/stable/modules/ppo.html",
         "Requires a caller-supplied Gymnasium environment; no market simulator or trading orders."),
        ("sac", "reinforcement_learning", "stable_baselines3", ["fit", "predict", "save", "load"],
         "pip install 'fin-skills[rl]'", "https://stable-baselines3.readthedocs.io/en/stable/modules/sac.html",
         "Requires a continuous-action Gymnasium environment; no market simulator or trading orders."),
        ("fly_memory", "associative_memory", "fin_skills_fly", ["predict", "update", "save", "load"],
         "From checkout: pip install ./benchmarks/fly_paper", "benchmarks/fly_paper/README.md",
         "Experimental two-cue memory; explicit circuit parameters; checked episode updates, no autonomous trading policy."),
    ):
        cards.append(dict(id=name, task=family, adapter=("sequence" if module == "neuralforecast" else
            "rl" if module == "stable_baselines3" else "fly"), inputs=(
            ["unique_id", "ds", "y"] if module == "neuralforecast" else ["env"] if module == "stable_baselines3"
            else ["circuit_parameters", "episode_receipt"]), operations=operations,
            status="ready" if _available(module) else "missing_dependency", library=module,
            pretrained=False, install=install, license=("GPL-3.0-or-later optional extension"
            if module == "fin_skills_fly" else "Core MIT; upstream runtime license applies"),
            stage="experimental", source=source, caveat=caveat, json_run=False))
    from .upstream_catalog import upstream_cards
    cards.extend(upstream_cards(_available))
    if task is not None and task not in {c["task"] for c in cards}:
        raise ValueError(f"unknown model task: {task!r}")
    return [c for c in cards if task is None or c["task"] == task]
