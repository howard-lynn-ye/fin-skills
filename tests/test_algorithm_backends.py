"""Numerical parity with real optional libraries in isolated subprocesses."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

MODULES = {"auto_arima": "statsforecast", "garch": "arch", "egarch": "arch",
    "engle_granger": "statsmodels", "gaussian_hmm": "hmmlearn", "change_points": "ruptures",
    "quantlib_heston": "QuantLib", "qlib_alpha": "qlib",
    "lightgbm_regression": "lightgbm", "lightgbm_classification": "lightgbm",
    "xgboost_regression": "xgboost", "xgboost_classification": "xgboost"}

PRELUDE = '''
import sys, tempfile
import numpy as np
import pandas as pd
from fin_skills.algorithms import run, fit, load_model
rng = np.random.default_rng(17)
X = rng.normal(size=(180, 3))
y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, .1, 180)
R = rng.normal(0, .01, 400)
series = np.cumsum(R)
id = sys.argv[1]
'''

CHECKS = {
    "auto_arima": '''
from statsforecast.models import AutoARIMA
data = {'series': series, 'seasonal_period': 1}
out = run(id, data, horizon=3)
expected = AutoARIMA(season_length=1).fit(y=series).predict(h=3)['mean']
np.testing.assert_allclose(out, expected)
model = fit(id, data)
np.testing.assert_allclose(model.predict(horizon=3), out)
''',
    "arch": '''
from arch import arch_model
out = run(id, {'returns': R}, periods_per_year=252)
model = arch_model(R * 100, mean='Zero', vol=id.upper(), p=1, o=1 if id=='egarch' else 0,
                   q=1, rescale=False).fit(disp='off', show_warning=False)
expected = np.sqrt(model.forecast(horizon=1, reindex=False).variance.iloc[-1, 0] * 252) / 100
np.testing.assert_allclose(out['volatility'], expected, rtol=1e-10)
assert out['converged']
''',
    "engle_granger": '''
from statsmodels.tsa.stattools import coint
pair = np.column_stack([series, series * 2 + rng.normal(0, .001, len(series))])
out = run(id, {'X': pair})
expected = coint(pair[:, 0], pair[:, 1], trend='c', autolag='aic')
np.testing.assert_allclose([out['statistic'], out['pvalue']], expected[:2])
assert out['pvalue'] < .05
''',
    "gaussian_hmm": '''
from hmmlearn.hmm import GaussianHMM
data = np.vstack([rng.normal(-2, .4, (90, 2)), rng.normal(2, .4, (90, 2))])
out = run(id, {'X': data}, seed=2)
expected = GaussianHMM(n_components=2, covariance_type='diag', n_iter=200, random_state=2).fit(data)
np.testing.assert_equal(out['states'], expected.predict(data))
assert 'retrospective' in out['timing']
''',
    "change_points": '''
import ruptures as rpt
data = np.r_[np.zeros(50), np.ones(50) * 5]
out = run(id, {'series': data})
expected = rpt.Pelt(model='l2', min_size=5, jump=1).fit(data).predict(pen=10.)
assert out['segment_ends_exclusive'] == expected == [50, 100]
''',
    "quantlib_heston": '''
import QuantLib as ql
option = dict(S=100., K=100., T=1., r=.03, q=.01, sigma=.2, flag='c', exercise='european')
hp = dict(v0=.04, kappa=2., theta=.04, sigma=.001, rho=-.5,
          valuation_date='2025-01-01', expiry_date='2026-01-01')
before = ql.Settings.instance().evaluationDate
out = run(id, {'option': option, 'heston_parameters': hp})
from fin_skills.algorithms import auto_run
automatic = auto_run('pricing', {'option': option, 'heston_parameters': hp})
assert automatic['selection']['selected'] == 'quantlib_heston'
np.testing.assert_allclose(automatic['result'], out)
assert ql.Settings.instance().evaluationDate == before
bsm = run('black_scholes', {'option': option})
assert abs(out - bsm) < .01, (out, bsm)
assert out > 0
''',
    "qlib_alpha": '''
from sklearn.linear_model import Ridge
out = run(id, {'X': X, 'y': y, 'X_predict': X[:3]})
expected = Ridge(alpha=1., fit_intercept=True).fit(X, y).predict(X[:3])
np.testing.assert_allclose(out, expected, rtol=1e-10)
model = fit(id, {'X': X, 'y': y})
with tempfile.TemporaryDirectory() as temp:
    path = model.save(temp + '/model.zip')
    np.testing.assert_allclose(load_model(path, trusted=True).predict(X[:3]), expected)
''',
    "boosting": '''
from fin_skills.algorithms.models import supervised_estimator
target = (y > 0).astype(int) if id.endswith('classification') else y
out = run(id, {'X': X, 'y': target, 'X_predict': X[:3]}, seed=17)
if id.startswith('lightgbm'):
    from lightgbm import LGBMClassifier, LGBMRegressor
    cls = LGBMClassifier if id.endswith('classification') else LGBMRegressor
    ref = cls(n_estimators=100, max_depth=6, random_state=17, n_jobs=1, verbosity=-1)
else:
    from xgboost import XGBClassifier, XGBRegressor
    cls = XGBClassifier if id.endswith('classification') else XGBRegressor
    ref = cls(n_estimators=100, max_depth=6, random_state=17, n_jobs=1)
expected = ref.fit(X, target).predict(X[:3])
np.testing.assert_allclose(out, expected)
model = fit(id, {'X': X, 'y': target}, seed=17)
with tempfile.TemporaryDirectory() as temp:
    path = model.save(temp + '/model.zip')
    np.testing.assert_allclose(load_model(path, trusted=True).predict(X[:3]), out)
''',
}


@pytest.mark.parametrize("id,module", MODULES.items())
def test_optional_adapter_numerical_parity(id, module):
    if importlib.util.find_spec(module) is None:
        extra = os.environ.get("FIN_SKILLS_REQUIRE_EXTRA")
        if extra == "algorithms" and module != "qlib" or extra == "qlib" and module == "qlib":
            pytest.fail(f"required backend {module} was not installed")
        pytest.skip(f"optional {module} is absent")
    key = "arch" if id in ("garch", "egarch") else "boosting" if module in ("lightgbm", "xgboost") else id
    result = subprocess.run([sys.executable, "-c", PRELUDE + CHECKS[key], id],
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]),
                 OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
        capture_output=True, text=True, errors="replace", timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
