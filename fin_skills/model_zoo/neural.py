"""Thin Nixtla adapters; callers supply available history on an explicit calendar."""
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from fin_skills.algorithms.runtime import integer
from .base import ModelArtifact

MODELS = {
    "lstm_forecast": "LSTM",
    "transformer_forecast": "VanillaTransformer",
    "patchtst_forecast": "PatchTST",
    "nhits_forecast": "NHITS",
}
from .upstream_catalog import NEURAL
MODELS.update(NEURAL)


class NeuralForecastModel(ModelArtifact):
    dependencies = ("neuralforecast", "torch", "pytorch-lightning")

    def __init__(self, model_id, *, freq, h=1, input_size=24, max_steps=100,
                 seed=0, **backend_parameters):
        self.model_id = model_id
        self.freq = freq
        pd.tseries.frequencies.to_offset(freq)
        self.h = integer(h, "h", maximum=10000)
        self.input_size = integer(input_size, "input_size", maximum=10000)
        self.max_steps = integer(max_steps, "max_steps", maximum=1000000)
        self.seed = integer(seed, "seed", minimum=0, maximum=2**32-1)
        unsupported = {"futr_exog_list", "hist_exog_list", "stat_exog_list",
                       "random_seed", "alias"}.intersection(backend_parameters)
        if unsupported:
            raise ValueError(f"unsupported adapter parameters: {sorted(unsupported)}")
        self.parameters = dict(backend_parameters)
        self.forecaster = None

    def _history(self, df, minimum):
        if not isinstance(df, pd.DataFrame) or set(df.columns) != {"unique_id", "ds", "y"}:
            raise ValueError("history must be a DataFrame with exactly unique_id, ds, y")
        result = df.copy(deep=True)
        if result.empty or result.isna().any().any():
            raise ValueError("history must be nonempty without missing values")
        if not pd.api.types.is_datetime64_any_dtype(result.ds):
            raise ValueError("ds must be datetime; parse dates explicitly before fitting")
        if result.ds.dt.tz is not None:
            raise ValueError("use an explicit timezone-naive calendar for ds")
        if not pd.api.types.is_numeric_dtype(result.y) or not np.isfinite(result.y).all():
            raise ValueError("y must contain finite numeric observations")
        for _, group in result.groupby("unique_id", sort=False):
            dates = pd.DatetimeIndex(group.ds)
            if len(group) < minimum:
                raise ValueError(f"each series needs at least {minimum} observations")
            if not dates.is_monotonic_increasing or dates.has_duplicates:
                raise ValueError("each series must have ordered, unique dates")
            expected = pd.date_range(dates[0], periods=len(dates), freq=self.freq)
            if not dates.equals(expected):
                raise ValueError("dates must follow freq; do not silently fill market gaps")
        return result

    def fit(self, df):
        from neuralforecast import NeuralForecast, models
        history = self._history(df, self.input_size + self.h)
        parameters = dict(accelerator="cpu", devices=1, logger=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          enable_checkpointing=False)
        parameters.update(self.parameters)
        if self.model_id in ("itransformer", "timemixer"):
            n_series = history.unique_id.nunique()
            if "n_series" in parameters and parameters["n_series"] != n_series:
                raise ValueError("n_series must match the training panel")
            parameters["n_series"] = n_series
        estimator = getattr(models, MODELS[self.model_id])(
            h=self.h, input_size=self.input_size, max_steps=self.max_steps,
            random_seed=self.seed, alias=self.model_id, **parameters)
        forecaster = NeuralForecast(models=[estimator], freq=self.freq)
        forecaster.fit(df=history, val_size=0)
        self.forecaster = forecaster
        return self

    def predict(self, history=None):
        if self.forecaster is None:
            raise ValueError("fit the model before prediction")
        frame = None if history is None else self._history(history, self.input_size)
        result = self.forecaster.predict(df=frame)
        values = result.drop(columns=["unique_id", "ds"])
        if not np.isfinite(values.to_numpy()).all():
            raise ArithmeticError("nonfinite backend predictions")
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        forecaster = state.pop("forecaster")
        state["_native_files"] = None
        if forecaster is not None:
            with TemporaryDirectory(prefix="fin-skills-nf-") as directory:
                forecaster.save(path=directory, overwrite=True, save_dataset=True)
                state["_native_files"] = {
                    p.relative_to(directory).as_posix(): p.read_bytes()
                    for p in Path(directory).rglob("*") if p.is_file()}
        return state

    def __setstate__(self, state):
        from neuralforecast import NeuralForecast
        files = state.pop("_native_files")
        self.__dict__.update(state)
        self.forecaster = None
        if files is not None:
            with TemporaryDirectory(prefix="fin-skills-nf-") as directory:
                for name, data in files.items():
                    path = PurePosixPath(name)
                    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                        raise ValueError("invalid native checkpoint path")
                    target = Path(directory).joinpath(*path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                self.forecaster = NeuralForecast.load(path=directory)
