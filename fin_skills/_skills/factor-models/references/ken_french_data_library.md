# Ken French Data Library: what the factor files are

✅ Read 2026-09-09 from `mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/` — the
data-library index, the Fama/French factor description page, and the "6 Portfolios Formed on
Size and Book-to-Market" description page. Paraphrased except where quoted; the pages are
copyright Eugene F. Fama and Kenneth R. French.

## The construction, in the order the calendar runs

```
Dec of t-1        ME used in the BE/ME ratio
fiscal year       BE used in the BE/ME ratio: the last fiscal year end in t-1
  end in t-1
end of June t     portfolios FORMED; size measured here (ME at end of June t)
Jul t .. Jun t+1  the twelve monthly returns those portfolios earn
end of June t+1   re-formed
```

Consequences a re-implementation usually misses:

1. **Two different market equities are in play.** ME in the BE/ME ratio is December of t-1; ME
   for the size sort is the end of June of t. Using one ME for both is a different factor.
2. **The accounting number is 6 to 18 months stale** when it earns its return, which is the
   delay that makes it public information.
3. **The re-form is annual**, at the end of June. A monthly re-sort satisfies the same formulas
   and produces a different series with different turnover.
4. **Breakpoints are NYSE-only, applied to a NYSE + AMEX + NASDAQ universe.** ✅ The size
   breakpoint is quoted as "the median NYSE market equity at the end of June of year t" and the
   value breakpoints as "the 30th and 70th NYSE percentiles". A full-universe breakpoint puts
   far more names in the small bucket, because NASDAQ microcaps dominate the count.
5. **The six portfolios are value-weighted already.** ✅ The factor page describes them as
   "6 value-weight portfolios formed on size and book-to-market". Do not re-weight them.

## The two averaging formulas (verbatim from the factor page)

```
SMB = 1/3 (Small Value + Small Neutral + Small Growth)
    - 1/3 (Big Value   + Big Neutral   + Big Growth)

HML = 1/2 (Small Value + Big Value) - 1/2 (Small Growth + Big Growth)
```

`scripts/factor_regression.py::fama_french_2x3` implements exactly these, and
`tests/test_models_factor_regression.py` asserts the identity against the six columns it
returns.

`Rm-Rf` is the value-weighted return of all CRSP firms incorporated in the US and listed on
NYSE, AMEX or NASDAQ (with the usual share-code and data filters), minus the one-month Treasury
bill rate. It is already an excess return. SMB and HML are long-short, so they are excess
returns by construction and **must not** have the risk-free rate subtracted again — the single
most common unit error after the percent/decimal one.

## Frequencies

✅ On 2026-09-09 the site listed the Fama/French factors at **daily, weekly, monthly and annual**
frequency; the first three began July 1926 and ran to July 2026, the annual file 1927 to 2025.
The 5-factor set is offered daily and monthly; the weekly file is 3-factor only.

**Match the frequency on both sides of the regression.** A monthly row is a whole-month return
and belongs with a whole-month strategy return. Forward-filling monthly factors onto daily
strategy returns produces a beta that is an artefact of the fill, not an estimate.

## Reading the files

- Monthly files stamp each row with a six-digit `YYYYMM`; daily files with `YYYYMMDD`.
  ⚠️ This was not verified against the archive here — check the first column of your own
  download. What *was* verified is the consequence: ✅ `pd.to_datetime("202607", format="%Y%m")`
  is `2026-07-01`, a first-of-month timestamp, so an inner join against month-end strategy
  dates keeps zero rows. Convert both sides to `PeriodIndex(freq="M")` and join on the period
  (`french_period_index` / `align_monthly`).
- ⚠️ The units of the archive files were not verified here. ✅ What the script measures is the
  cost of guessing wrong: the same decimal strategy regressed on the same factor column gives
  beta `-0.004243` read as percent and `-0.4243` read as a decimal, a factor of exactly 100.
  Print `df.describe()` on the factor column before you divide by anything — a monthly market
  excess return has a standard deviation near 5 in percent units and near 0.05 in decimals.
- Missing values: this skill did **not** verify how the archive codes them. Do the defensive
  check regardless — `df.min()`, `df.max()` and a count of repeated extreme values — because a
  numeric missing-value sentinel read as a return is an observation many standard deviations
  from the mean, and one of them will dominate every coefficient in the regression.

## Reproduction checklist

Before concluding "my HML does not match Ken French", confirm in this order:

1. Frequency and period alignment (a silent empty or partial join).
2. Units (percent vs decimal).
3. Re-form calendar: annual at end of June, not monthly.
4. Breakpoint universe: NYSE-only, not the full universe.
5. The two market equities: December t-1 in the ratio, June t for size.
6. The delay: BE from the last fiscal year ending in t-1.
7. Only then, the code.
