"""fin_skills.core.identifier_checks - the arithmetic, against published identifiers.

Every identifier asserted here was read out of a filing on sec.gov on 2026-09-10, so the
check digits are validated against what an issuer actually printed rather than against
another implementation of the same formula. The corrupted variants are the other half:
a validator that accepts everything passes the first test and none of the second.
"""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.core.identifier_checks import (AAPL_FORMER_NAMES, DEMO_TICKERS,
                                               NPX_PAIRS, PUBLISHED, WMI_CURRENT_NAME,
                                               WMI_FORMER_NAMES, Window,
                                               check_identifier, cusip_check_digit,
                                               cusip_from_isin, embedding_rate,
                                               figi_check_digit, isin_check_digit,
                                               label_as_of, multi_ticker_rate,
                                               name_windows, sedol_check_digit,
                                               window_gaps, window_overlaps)


# --------------------------------------------------------------------- valid identifiers
@pytest.mark.parametrize("scheme,identifier,what", PUBLISHED)
def test_published_identifiers_validate(scheme, identifier, what):
    result = check_identifier(identifier, scheme)
    assert result.ok, f"{what}: {result.reason}"
    assert result.identifier == identifier.upper()


def test_check_digits_reproduce_the_published_last_character():
    # Apple, IBM, Amazon and BAE, all read from SEC filings on 2026-09-10
    assert isin_check_digit("US037833100") == "5"           # US0378331005
    assert isin_check_digit("GB000263494") == "6"           # GB0002634946
    assert cusip_check_digit("02313510") == "6"             # 023135106, Amazon
    assert cusip_check_digit("G0694010") == "3"             # G06940103, a CINS
    assert sedol_check_digit("026349") == "4"               # 0263494, inside GB0002634946
    assert sedol_check_digit("BH4HKS") == "3"               # BH4HKS3, Vodafone
    assert figi_check_digit("BBG000BLNNH") == "6"           # BBG000BLNNH6, IBM


def test_the_sedol_inside_the_uk_isin_is_the_same_number():
    # GB0002634946 -> country GB, NSIN 000263494, check 6. Strip the padding and the NSIN
    # is BAE's SEDOL, whose own weighted-modulus digit validates independently.
    isin = "GB0002634946"
    assert check_identifier(isin, "isin").ok
    nsin = isin[2:11]
    assert nsin == "000263494"
    assert check_identifier(nsin.lstrip("0").zfill(7), "sedol").ok


# ------------------------------------------------------------------ corrupted identifiers
@pytest.mark.parametrize("scheme,identifier,_what", PUBLISHED)
def test_a_transposed_pair_is_caught_unless_it_is_luhn_s_blind_spot(scheme, identifier,
                                                                    _what):
    """Every adjacent digit swap is caught EXCEPT 0<->9, which Luhn cannot see."""
    body = identifier[:-1]
    for i in range(len(body) - 1):
        if body[i] != body[i + 1] and body[i].isdigit() and body[i + 1].isdigit():
            swapped = body[:i] + body[i + 1] + body[i] + body[i + 2:] + identifier[-1]
            blind = abs(int(body[i]) - int(body[i + 1])) == 9
            break
    else:                                                    # pragma: no cover
        pytest.skip("no swappable digit pair in this identifier")
    caught = not check_identifier(swapped, scheme).ok
    assert caught or blind, f"{swapped} slipped through and is not a 0<->9 swap"


def test_the_luhn_blind_spot_is_real_on_a_real_isin():
    """NXP's own ISIN has a 0/9 pair, so the typo passes the check digit.

    Both strings validate. That is not a defect in this implementation - it is what a
    Luhn-family check digit is: it catches every single-character error and every adjacent
    transposition except 0<->9. Treat a passing check as "not obviously mistyped", never
    as "this identifier exists".
    """
    assert check_identifier("NL0009538784", "isin").ok         # NXP, read from a filing
    assert check_identifier("NL0090538784", "isin").ok         # a 0<->9 swap of the above
    assert check_identifier("NL0009538784", "isin").identifier != "NL0090538784"


@pytest.mark.parametrize("scheme,identifier", [
    ("isin", "US0378331000"), ("isin", "US0378331009"),
    ("cusip", "037833101"), ("cusip", "037833109"),
    ("sedol", "BH4HKS0"), ("sedol", "0263490"),
    ("figi", "BBG000BLNNH0"), ("figi", "BBG000BLNNH9"),
])
def test_a_wrong_check_digit_fails_with_the_computed_one_in_the_reason(scheme, identifier):
    r = check_identifier(identifier, scheme)
    assert not r.ok
    assert "computed" in r.reason


def test_shape_errors_are_reported_before_the_check_digit():
    assert "length" in check_identifier("US03783310", "isin").reason
    assert "country" in check_identifier("120378331005", "isin").reason
    assert "vowel" in check_identifier("BAEHKS3", "sedol").reason      # SEDOLs have none
    assert "third character" in check_identifier("BBB000BLNNH6", "figi").reason
    assert "illegal" in check_identifier("US03783$1005", "isin").reason
    assert "unknown scheme" in check_identifier("US0378331005", "permno").reason


def test_check_identifier_never_raises_on_junk():
    for junk in ("", "   ", "-", "0", "x" * 40, "US0378331005 "):
        assert isinstance(check_identifier(junk, "isin").ok, bool)


# ------------------------------------------------------------- deriving one from another
def test_cusip_from_isin_returns_none_rather_than_a_plausible_slice():
    assert cusip_from_isin("US0378331005") == "037833100"
    assert cusip_from_isin("US0231351067") == "023135106"
    # NXP: the ISIN is Dutch, the CUSIP is a CINS, and the slice is neither
    assert "NL0009538784"[2:11] == "000953878"
    assert cusip_from_isin("NL0009538784") is None
    assert cusip_from_isin("not-an-isin") is None


def test_embedding_rate_over_the_npx_fixture():
    r = embedding_rate(NPX_PAIRS)
    assert r["pairs"] == 5 and r["embeds"] == 4 and r["does_not"] == 1
    assert r["misses"] == [("N6596X109", "NL0009538784")]
    assert embedding_rate([])["pairs"] == 0


# ------------------------------------------------------------------------- name windows
def test_sec_to_is_inclusive_so_a_zero_width_record_survives_conversion():
    ws = name_windows(AAPL_FORMER_NAMES, "Apple Inc.", identifier="320193")
    fa = [w for w in ws if w.label.endswith("/ FA")][0]
    assert fa.start == pd.Timestamp("1997-07-28")
    assert fa.end == pd.Timestamp("1997-07-29")             # to + 1 day, not to
    assert fa.covers("1997-07-28") and not fa.covers("1997-07-29")


def test_the_sec_name_history_gives_two_answers_on_one_day():
    ws = name_windows(AAPL_FORMER_NAMES, "Apple Inc.", identifier="320193")
    assert label_as_of(ws, "1997-07-28") == ["APPLE COMPUTER INC",
                                             "APPLE COMPUTER INC/ FA"]
    ov = window_overlaps(ws)
    assert len(ov) == 1                                     # and only that one
    assert set(ov.iloc[0][["a", "b"]]) == {"APPLE COMPUTER INC", "APPLE COMPUTER INC/ FA"}


def test_the_sec_name_history_also_gives_zero_answers():
    ws = name_windows(AAPL_FORMER_NAMES, "Apple Inc.", identifier="320193")
    gaps = window_gaps(ws)
    assert len(gaps) == 1
    assert gaps.iloc[0]["days"] == 5
    assert gaps.iloc[0]["gap_from"] == pd.Timestamp("2007-01-05")
    assert label_as_of(ws, "2007-01-07") == []


def test_one_cik_spans_six_registrants():
    ws = name_windows(WMI_FORMER_NAMES, WMI_CURRENT_NAME, identifier="933136")
    assert len(ws) == 6
    assert label_as_of(ws, "2007-06-29") == ["WASHINGTON MUTUAL, INC"]
    assert label_as_of(ws, "2013-06-28") == ["WMI HOLDINGS CORP."]
    assert label_as_of(ws, "2019-12-31") == ["Mr. Cooper Group Inc."]
    assert label_as_of(ws, "2026-09-10") == [WMI_CURRENT_NAME]
    # the current name's window is inferred: it opens where the newest former one closed
    assert ws[-1].start == pd.Timestamp("2025-10-02") and ws[-1].end is None


def test_a_current_name_with_no_former_names_still_produces_one_window():
    ws = name_windows([], "SOMETHING NEW", identifier="1")
    assert len(ws) == 1 and ws[0].end is None
    assert label_as_of(ws, "2020-01-01") == ["SOMETHING NEW"]


def test_windows_that_merely_touch_do_not_overlap():
    ws = [Window("x", "a", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-01-01")),
          Window("x", "b", pd.Timestamp("2021-01-01"), None)]
    assert len(window_overlaps(ws)) == 0
    assert len(window_gaps(ws)) == 0
    assert label_as_of(ws, "2021-01-01") == ["b"]


# ----------------------------------------------------------------------- ticker -> CIK
def test_multi_ticker_rate_counts_ciks_not_rows():
    m = multi_ticker_rate(DEMO_TICKERS)
    assert m["rows"] == 8 and m["ciks"] == 4
    assert m["multi"] == 2 and m["max_tickers"] == 4
    assert m["pct"] == pytest.approx(50.0)


def test_multi_ticker_rate_normalises_zero_padded_ciks():
    rows = [{"cik_str": "0000320193", "ticker": "AAPL"},
            {"cik_str": 320193, "ticker": "AAPL2"}]
    assert multi_ticker_rate(rows)["ciks"] == 1


def test_multi_ticker_rate_on_an_empty_file_reports_zero_rather_than_dividing():
    m = multi_ticker_rate([])
    assert m["ciks"] == 0 and pd.isna(m["pct"])


# ------------------------------------------------------------------------------ demo
def test_demo_runs_and_stays_ascii(run_main):
    out = run_main("fin_skills.core.identifier_checks")
    assert out.isascii()
    assert "Maverick Merger Sub 2, LLC" in out
    assert "resolve() must return a LIST" in out
