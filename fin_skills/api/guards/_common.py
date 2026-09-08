"""Input validators shared by the guard modules. Each raises TypeError, never ValueError."""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


def require_frame(x: Any, name: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(x).__name__}")
    return x


def require_series(x: Any, name: str) -> pd.Series:
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(x).__name__}")
    return x


def require_callable(x: Any, name: str) -> Callable[..., Any]:
    if not callable(x):
        raise TypeError(f"{name} must be callable, got {type(x).__name__}")
    return x


def require_str(x: Any, name: str) -> str:
    if not isinstance(x, str) or not x:
        raise TypeError(f"{name} must be a non-empty str, got {x!r}")
    return x


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise TypeError(f"{name} is missing column(s) {missing}; has {list(df.columns)[:12]}")


def require_int(x: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(x).__name__}")
    v = int(x)
    if minimum is not None and v < minimum:
        raise TypeError(f"{name} must be >= {minimum}, got {v}")
    return v


def require_number(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a number, got {type(x).__name__}")
    v = float(x)
    if not np.isfinite(v):
        raise TypeError(f"{name} must be finite, got {v}")
    return v


def as_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x.to_numpy() if isinstance(x, pd.Series) else x, dtype=float)
    if arr.ndim != 1:
        raise TypeError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def as_2d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x.to_numpy() if isinstance(x, pd.DataFrame) else x, dtype=float)
    if arr.ndim != 2:
        raise TypeError(f"{name} must be two-dimensional, got shape {arr.shape}")
    return arr


def as_date(x: Any, name: str):
    """Coerce str / date / datetime / Timestamp to datetime.date."""
    from datetime import date, datetime
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    try:
        return pd.Timestamp(x).date()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a date, got {x!r}") from exc
