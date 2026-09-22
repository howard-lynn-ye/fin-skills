"""Native Hummingbot strategy factories for caller-owned runtime objects.

Factories do not start the engine or create exchange connections. Native connector,
configuration and clock objects are caller supplied; this is not a JSON execution tool.
"""
import importlib
import importlib.util
from pathlib import Path

STRATEGIES = {
    "pure_market_making": ("pure_market_making.pure_market_making", "PureMarketMakingStrategy"),
    "cross_exchange_market_making": ("cross_exchange_market_making.cross_exchange_market_making",
                                     "CrossExchangeMarketMakingStrategy"),
    "avellaneda_stoikov": ("avellaneda_market_making.avellaneda_market_making",
                           "AvellanedaMarketMakingStrategy"),
    "amm_arbitrage": ("amm_arb.amm_arb", "AmmArbStrategy"),
    "spot_perpetual_basis": ("spot_perpetual_arbitrage.spot_perpetual_arbitrage",
                             "SpotPerpetualArbitrageStrategy"),
}


def native_strategy(method_id, **parameters):
    """Create and initialize one upstream V1 strategy without starting its clock."""
    if method_id not in STRATEGIES:
        raise KeyError(f"unknown Hummingbot V1 strategy: {method_id}")
    module, name = STRATEGIES[method_id]
    cls = getattr(importlib.import_module("hummingbot.strategy." + module), name)
    strategy = cls()
    strategy.init_params(**parameters)
    return strategy


def funding_strategy(*, checkout, connectors, config):
    """Load the official script from an explicitly trusted Hummingbot checkout.

    The upstream distribution does not package its scripts/ directory. A local checkout
    is therefore explicit, not a downloaded or model-provided arbitrary Python path.
    """
    root = Path(checkout).resolve(strict=True)
    script = root / "scripts" / "v2_funding_rate_arb.py"
    if not (root / "hummingbot" / "strategy" / "strategy_v2_base.py").is_file():
        raise ValueError("checkout must be a Hummingbot source tree")
    spec = importlib.util.spec_from_file_location("fin_skills_hummingbot_funding", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if isinstance(config, dict):
        config = module.FundingRateArbitrageConfig(**config)
    return module.FundingRateArbitrage(connectors=connectors, config=config)
