"""fin_skills.crypto.perp_mechanics - 365 rows a year, funding on notional, liquidation is not a stop.

The module is arithmetic on constants pulled from venue APIs plus seven printing
sections; the helpers are tested exactly and each section is run under capsys.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _helpers import is_ascii
from fin_skills.crypto import perp_mechanics as pm


def test_liquidation_price_for_a_10x_long_is_entry_times_1_minus_1_over_L_plus_mmr():
    mmr = pm.OKX_TIER1["mmr"]
    assert pm.liq_ratio(10, mmr, "long", exact=False) == pytest.approx(1 - 1 / 10 + mmr)
    assert pm.liq_ratio(10, mmr, "short", exact=False) == pytest.approx(1 + 1 / 10 - mmr)
    exact = pm.liq_ratio(10, mmr, "long")
    assert exact == pytest.approx((1 - 1 / 10) / (1 - mmr))
    assert abs(exact - pm.liq_ratio(10, mmr, "long", exact=False)) < 1e-3   # "a few bps"
    assert pm.liq_ratio(3, mmr, "long") < pm.liq_ratio(5, mmr, "long") < exact < 1.0
    assert pm.liq_ratio(10, mmr, "short") > 1.0


def test_annualisation_helpers_scale_by_the_square_root_of_rows_per_year():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, 730), index=pd.date_range("2022-01-01", periods=730, freq="D"))
    assert pm.sharpe(r, 365) / pm.sharpe(r, 252) == pytest.approx(np.sqrt(365 / 252))
    assert pm.vol(r, 365) / pm.vol(r, 252) == pytest.approx(np.sqrt(365 / 252))
    flat = pd.Series(0.001, index=pd.date_range("2022-01-01", periods=365, freq="D"))
    assert pm.cagr_calendar(flat) == pytest.approx(1.001 ** 365 - 1)


def test_embedded_venue_data_has_the_documented_shape():
    assert len(pm.DERIBIT_FUND_BPS) == len(pm.DERIBIT_INDEX_DAILY) == len(pm.DERIBIT_DAYS) == 365
    assert len(pm.OKX_FUND_BPS) == 277                      # 277 eight-hour settlements
    assert len(pm.COINBASE60) == len(pm.KRAKEN60) == len(pm.GEMINI60) == len(pm.CLOSE_DAYS) == 60
    assert sum(pm.OKX_INTERVALS.values()) == 644
    assert set(pm.VENUE_HTTP.values()) <= {200, 403, 451}
    for name, (mark, expiry) in pm.DERIBIT_FUTURES.items():
        assert pd.Timestamp(expiry) > pm.AS_OF.tz_localize(None) and mark > 0


@pytest.mark.parametrize("section", ["section_annualisation", "section_funding", "section_basis",
                                     "section_liquidation", "section_inverse", "section_venue",
                                     "section_data"])
def test_each_section_prints_ascii_and_its_heading(capsys, section):
    getattr(pm, section)()
    out = capsys.readouterr().out
    assert out.count("=" * 98) == 2 and is_ascii(out)
    assert section.split("_", 1)[1].upper()[:4] in out.upper()


def test_annualisation_section_is_deterministic(capsys):
    pm.section_annualisation()
    first = capsys.readouterr().out
    pm.section_annualisation()
    assert capsys.readouterr().out == first


def test_demo_ends_with_the_rules(run_main):
    out = run_main("fin_skills.crypto.perp_mechanics")
    assert "Nothing here touches the network." in out
    assert "Rules: 365 rows a year -> sqrt(365)." in out
    assert is_ascii(out)
