# linearmodels — Fama-MacBeth and panel asset pricing

The reference implementation for panel and asset-pricing econometrics in Python. **statsmodels has
no real Fama-MacBeth and no proper entity/time fixed-effects panel estimator**, so for
cross-sectional asset pricing this is not one option among several — it is the option.

| | |
|---|---|
| pip | `linearmodels` |
| Version | **7.0** (2025-10-21) — a major bump from 6.1 (2024-09-24). 45 releases since 1.0 (2017) |
| Licence | ⚠️ **NCSA** — permissive and BSD-like, but **not** BSD or MIT. Flag it if you have a strict licence allowlist |
| Python | `>=3.10`, classifiers 3.10–3.13 |
| GitHub | `bashtage/linearmodels` — 1,066★, 198 forks, 51 open issues, pushed **2026-08-31** |
| Deps | numpy, pandas, scipy, `statsmodels>=0.13`, **`formulaic>=1.2.1`** (not patsy), `pyhdfe>=0.1` |
| Status | ✅ healthy, actively maintained. Same author as `arch` (Kevin Sheppard) |

✅ Confirmed in source: `PanelOLS` and `FamaMacBeth` in `linearmodels/panel/model.py`, and a
dedicated `linearmodels/asset_pricing/` subpackage holding `LinearFactorModel`,
`LinearFactorModelGMM` and `TradedFactorModel` — the time-series and cross-sectional asset-pricing
tests, including the **J-statistic** for whether pricing errors are jointly zero.

## 🚨 Index order is the opposite of alphalens

```python
# REQUIRED: MultiIndex (entity, time) — entity FIRST, time SECOND.
# alphalens wants (date, asset). Getting this backwards produces a
# valid-looking but meaningless regression, with no error.
data = data.set_index(["permno", "date"])
```

## Fama-MacBeth

```python
from linearmodels.panel import FamaMacBeth

res = FamaMacBeth(dependent=data.ret,
                  exog=data[["const", "beta", "size", "bm"]]).fit(
    cov_type="kernel", bandwidth=None    # <- Newey-West on the lambda series
)
print(res.summary)
```

🚨 **`cov_type="kernel"` is not optional in practice.** Fama-MacBeth averages per-period
cross-sectional slopes, so **cross-sectional correlation within a period is already handled by
construction**. What is *not* handled is **serial correlation of the lambda series** — and with
overlapping or persistent returns that correlation is large. Using the default standard errors is
**the single biggest error in applied Fama-MacBeth work**, and it inflates t-statistics in the
direction that makes a factor look significant.

## Panel with fixed effects

```python
from linearmodels.panel import PanelOLS

res = PanelOLS(data.ret, data[exog_cols],
               entity_effects=True, time_effects=True).fit(
    cov_type="clustered", cluster_entity=True, cluster_time=True
)
# or, with formulaic:
PanelOLS.from_formula("ret ~ 1 + beta + size + EntityEffects + TimeEffects", data)
```

## Choosing standard errors

| `cov_type` | When it's right |
|---|---|
| `"unadjusted"` | Basically never for asset pricing — assumes iid |
| `"robust"` | Heteroskedasticity only. Still wrong under any correlation structure |
| `"clustered"`, `cluster_entity=True` | Firm-level persistence in residuals — the dominant effect in most panels |
| `"clustered"`, `cluster_time=True` | Common shocks across firms within a period |
| **both (two-way)** | **The Petersen (2009) default recommendation for finance panels** |
| `"kernel"` | **Driscoll-Kraay** — robust to cross-sectional *and* serial correlation. The right choice with persistent common time-series shocks, and the right default for Fama-MacBeth |

**Clustering on the wrong dimension is how a t-stat of 2 becomes a t-stat of 5.** Decide it from the
economics, not from which one gives you significance.

## Practical notes

- `pyhdfe` provides high-dimensional fixed-effect absorption, so `PanelOLS` handles many-entity
  panels efficiently.
- **`formulaic`, not patsy**, is the formula engine as of recent versions.
- Unbalanced panels are handled, but 🚨 **`FamaMacBeth` silently drops periods** with too few
  cross-sectional observations. **Check `res.time_info` and the number of periods actually used** —
  a silently shortened sample is common with sparse universes and changes what you are estimating.
- ⚠️ **v7.0 (Oct 2025) is a major bump** — read the changelog before upgrading pinned code.

## Related

- **`pyfixest`** is faster and more modern for high-dimensional fixed effects, but 🚨 **has no
  Fama-MacBeth** — it is complementary, not a replacement.
- For factor *returns* to regress against, see `_event-study-method.md` and the Fama-French section
  of the parent skill — `pandas_datareader.famafrench` works and its parser was fixed in 0.11.0.
- For "is the alpha real after I tried many specifications", the answer is not a t-stat — see
  `../../backtest-validation/SKILL.md`.
