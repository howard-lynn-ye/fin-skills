"""Bundle <-> a qlib handler/dataset, with the normalizer's fit window forced disjoint.

THE TRAP (fin_skills.load('lib-qlib'), "The trap that costs you money"): qlib's default
inference processors are

    _DEFAULT_INFER_PROCESSORS = [{"class": "ProcessInf"},
                                 {"class": "ZScoreNorm"},   # fit on [fit_start, fit_end]
                                 {"class": "Fillna"}]

`ZScoreNorm` is a TIME-SERIES normalizer fit over `[fit_start_time, fit_end_time]`. qlib
forces you to pass those (`check_transform_proc` asserts non-None) but does nothing to stop
you passing the whole sample - and every tutorial that sets `fit_end_time` to the end of
the data has leaked test-set moments into the training features, silently. The label side
(`CSZScoreNorm`) is cross-sectional and per-date, so it is leak-free by construction; the
dangerous one is the time-series normalizer on the FEATURES.

What this bridge does, all before qlib is imported:
  * requires `fit_start`/`fit_end` and `infer_start`/`infer_end` explicitly - there is no
    default and none is inferred from the data;
  * asserts `fit_end < infer_start`, strictly;
  * walks every processor you pass and refuses any whose own `fit_start_time`/
    `fit_end_time` range overlaps the inference window, so a hand-built processor list
    cannot smuggle the default back in.

Two packaging facts this bridge relies on: `pip install qlib` is the WRONG package (an
abandoned 2018 upload); Microsoft's is `pyqlib`, imported as `qlib`. Licence MIT - safe as
an extra.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from fin_skills.api import Bundle
from fin_skills.bridges import _lazy

LIBRARY = "pyqlib"
LICENCE = _lazy.info(LIBRARY).licence
VERIFIED_ON = _lazy.info(LIBRARY).verified_on

#: processors whose fitted state is a leak when the fit span reaches the inference window.
FITTED_PROCESSORS: frozenset[str] = frozenset({
    "ZScoreNorm", "RobustZScoreNorm", "MinMaxNorm", "CSZFillna", "TanhProcess",
    "DropnaProcessor", "Fillna",
})


class NormalizerLeakError(ValueError):
    """A processor would be fitted over a span that includes the inference window."""


def _ts(x: Any, name: str) -> pd.Timestamp:
    if x is None:
        raise TypeError(f"{name} is required: qlib's ZScoreNorm is fit over "
                        f"[fit_start, fit_end] and this bridge will not guess either end")
    try:
        return pd.Timestamp(x)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a date, got {x!r}") from exc


def _proc_window(proc: Any) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """(fit_start_time, fit_end_time) declared by a processor dict or object."""
    if isinstance(proc, Mapping):
        kw = proc.get("kwargs", proc)
        a, b = kw.get("fit_start_time"), kw.get("fit_end_time")
    else:
        a = getattr(proc, "fit_start_time", None)
        b = getattr(proc, "fit_end_time", None)
    return (pd.Timestamp(a) if a is not None else None,
            pd.Timestamp(b) if b is not None else None)


def _proc_name(proc: Any) -> str:
    if isinstance(proc, Mapping):
        return str(proc.get("class", proc.get("name", "processor")))
    return type(proc).__name__


def check_processors(processors: Iterable[Any], *, infer_start: pd.Timestamp,
                     infer_end: pd.Timestamp, where: str) -> None:
    """Refuse any processor whose fit span reaches into [infer_start, infer_end]."""
    for proc in processors or ():
        name = _proc_name(proc)
        a, b = _proc_window(proc)
        if a is None and b is None:
            if name in FITTED_PROCESSORS and name != "Fillna":
                raise NormalizerLeakError(
                    f"{where}: {name} declares no fit_start_time/fit_end_time, so qlib will "
                    f"fall back to the handler's - which is exactly how the default leaks. "
                    f"State the fit window on the processor itself.")
            continue
        if b is not None and b >= infer_start:
            raise NormalizerLeakError(
                f"{where}: {name} is fit over [{a}, {b}] which reaches the inference window "
                f"[{infer_start.date()}, {infer_end.date()}]. ZScoreNorm is a TIME-SERIES "
                f"normalizer: its mean and sd become features, so a fit span that includes "
                f"the test set puts test-set moments into training. Set fit_end_time to the "
                f"end of your TRAINING segment only.")


def to_qlib_handler(bundle: Bundle, *, fit_start: Any, fit_end: Any, infer_start: Any,
                    infer_end: Any, instruments: Any = "csi300",
                    handler: str = "Alpha158", **kw: Any):
    """Build a qlib DataHandlerLP whose normalizer cannot see the inference window.

    fit_start / fit_end      the TRAINING segment, and nothing else.
    infer_start / infer_end  the window the model will be asked about.
    handler                  'Alpha158' or 'Alpha360' from qlib.contrib.data.handler, or a
                             handler class.
    **kw                     forwarded to the handler: infer_processors, learn_processors,
                             freq, ... Every processor in either list is checked first.

    Raises `NormalizerLeakError` before qlib is imported when `fit_end >= infer_start` or
    when any processor's own fit span overlaps the inference window.
    """
    fs, fe = _ts(fit_start, "fit_start"), _ts(fit_end, "fit_end")
    is_, ie = _ts(infer_start, "infer_start"), _ts(infer_end, "infer_end")
    if fs > fe:
        raise NormalizerLeakError(f"fit_start {fs.date()} is after fit_end {fe.date()}")
    if is_ > ie:
        raise NormalizerLeakError(f"infer_start {is_.date()} is after infer_end {ie.date()}")
    if fe >= is_:
        raise NormalizerLeakError(
            f"fit_end {fe.date()} >= infer_start {is_.date()}: qlib's ZScoreNorm would be "
            f"fit over a span that includes the inference window, which puts test-set means "
            f"and standard deviations into the training features. The two windows must be "
            f"disjoint, and the gap should be at least your label horizon "
            f"(fin_skills.load('lib-qlib')).")
    for key, label in (("infer_processors", "infer_processors"),
                       ("learn_processors", "learn_processors")):
        check_processors(kw.get(key) or (), infer_start=is_, infer_end=ie, where=label)

    qlib = _lazy.need(LIBRARY, why="to build a data handler")
    from qlib.contrib.data import handler as qhandler                # noqa: PLC0415 - lazy

    cls = getattr(qhandler, handler) if isinstance(handler, str) else handler
    obj = cls(instruments=instruments, start_time=str(min(fs, is_).date()),
              end_time=str(max(fe, ie).date()), fit_start_time=str(fs.date()),
              fit_end_time=str(fe.date()), **kw)
    obj.fin_skills_provenance = _lazy.provenance(
        LIBRARY, qlib, request={"fit": [str(fs.date()), str(fe.date())],
                                "infer": [str(is_.date()), str(ie.date())],
                                "handler": getattr(cls, "__name__", str(cls))})
    return obj


def from_qlib(report: pd.DataFrame, positions: Any = None, *, sessions: Any = None,
              **extra: Any) -> Bundle:
    """qlib's backtest report -> Bundle.

    `report['return']` -> `returns` (qlib reports it GROSS and puts execution in `cost`),
    `report['bench']` -> `benchmark_returns`, `report['turnover']` -> `turnover`. qlib's
    turnover is already one-way traded value over account value, which is the slot's
    convention.
    """
    if not isinstance(report, pd.DataFrame):
        raise TypeError("report must be the DataFrame qlib's backtest returns")
    cols = {c.lower(): c for c in report.columns}
    if "return" not in cols:
        raise TypeError(f"report has no 'return' column; has {list(report.columns)}")
    slots: dict[str, Any] = {"returns": report[cols["return"]].astype(float)}
    if "bench" in cols:
        slots["benchmark_returns"] = report[cols["bench"]].astype(float)
    if "turnover" in cols:
        slots["turnover"] = report[cols["turnover"]].astype(float)
    if positions is not None:
        pos = _positions_series(positions)
        if pos is not None:
            slots["position"] = pos
    ppy = _lazy.periods_per_year(sessions)
    if ppy is not None:
        slots["periods_per_year"] = ppy
    slots.update(extra)
    return Bundle(**slots)


def _positions_series(positions: Any) -> pd.Series | None:
    """qlib hands back {date: Position}; the Bundle wants gross exposure per period."""
    if isinstance(positions, pd.Series):
        return positions.astype(float)
    if isinstance(positions, Mapping):
        rows = {}
        for day, p in positions.items():
            stocks = getattr(p, "position", None)
            if isinstance(stocks, Mapping):
                rows[pd.Timestamp(day)] = float(sum(
                    abs(float(v.get("amount", 0.0)) * float(v.get("price", 0.0)))
                    for v in stocks.values() if isinstance(v, Mapping)))
        if rows:
            s = pd.Series(rows).sort_index()
            return s / s.max() if float(s.max()) else s
    if isinstance(positions, Sequence):
        return None
    return None


__all__ = ["FITTED_PROCESSORS", "LIBRARY", "LICENCE", "NormalizerLeakError", "VERIFIED_ON",
           "check_processors", "from_qlib", "to_qlib_handler"]
