"""Optional real-engine acceptance tests; no substitute or mocked native algorithm."""
import os
from decimal import Decimal
from pathlib import Path

import pytest


def test_hummingbot_native_paper_quotes():
    pytest.importorskip("hummingbot")
    from hummingbot.connector.exchange.paper_trade.paper_trade_exchange import QuantizationParams
    from hummingbot.connector.test_support.mock_paper_exchange import MockPaperExchange
    from hummingbot.core.clock import Clock, ClockMode
    from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
    from fin_skills.bridges.hummingbot import native_strategy

    market = MockPaperExchange()
    market.set_balanced_order_book("HBOT-ETH", mid_price=100, min_price=1, max_price=200,
                                  price_step_size=1, volume_step_size=10)
    market.set_balance("HBOT", 500)
    market.set_balance("ETH", 5000)
    market.set_quantization_param(QuantizationParams("HBOT-ETH", 6, 6, 6, 6))
    strategy = native_strategy("pure_market_making",
        market_info=MarketTradingPairTuple(market, "HBOT-ETH", "HBOT", "ETH"),
        bid_spread=Decimal(".01"), ask_spread=Decimal(".01"), order_amount=Decimal("1"),
        order_refresh_time=5., filled_order_delay=5., order_refresh_tolerance_pct=-1,
        minimum_spread=-1)
    clock = Clock(ClockMode.BACKTEST, 1, 1600000000, 1600000010)
    clock.add_iterator(market)
    clock.add_iterator(strategy)
    clock.backtest_til(1600000001)
    assert [(o.price, o.quantity) for o in strategy.active_buys] == [(Decimal("99"), Decimal("1"))]
    assert [(o.price, o.quantity) for o in strategy.active_sells] == [(Decimal("101"), Decimal("1"))]
    clock.backtest_til(1600000007)
    assert len(strategy.active_buys) == len(strategy.active_sells) == 1


def test_lean_native_sample(tmp_path):
    checkout = os.environ.get("FIN_SKILLS_LEAN_CHECKOUT")
    dotnet = os.environ.get("FIN_SKILLS_DOTNET")
    if not checkout or not dotnet:
        pytest.skip("set native LEAN checkout and dotnet executable")
    from fin_skills.bridges.lean import LeanBacktest
    root = Path(checkout)
    engine = root / "Launcher/bin/Release"
    bridge = LeanBacktest(engine_dir=engine, dotnet=dotnet,
                          template_config=root / "Launcher/config.json")
    result = bridge.run(algorithm_file=engine / "QuantConnect.Algorithm.CSharp.dll",
        algorithm_type="BasicTemplateFrameworkAlgorithm", data_folder=root / "Data",
        work_dir=tmp_path / "backtest")
    assert result["returncode"] == 0 and result["live_mode"] is False
    assert Path(result["summary_path"]).is_file()
