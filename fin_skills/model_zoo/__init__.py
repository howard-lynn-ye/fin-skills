"""Discover and create models without eagerly importing optional runtimes.

    model = create_model("ridge", seed=7).fit({"X": X_train, "y": y_train})
    prediction = model.predict(X_test)

Capabilities are explicit: numerical methods run, estimators fit/predict,
RL policies learn in a caller-supplied environment, and fly memory updates receipts.
"""
from .catalog import model_catalog
from .base import load_model


def create_model(model_id, **parameters):
    """Create a named adapter; never install packages or download weights implicitly."""
    cards = {card["id"]: card for card in model_catalog()}
    if model_id not in cards:
        raise KeyError(f"unknown model: {model_id!r}; see model_catalog()")
    card = cards[model_id]
    if card["status"] != "ready":
        raise ImportError(f"{model_id}: {card['status']}; {card['install']}")
    family = card["adapter"]
    if family == "local_decision":
        if model_id == "kev":
            from .kev import KevModel
            return KevModel(**parameters)
        from .laya import LayaModel
        return LayaModel(**parameters)
    if family == "decision":
        from .jev import JevModel
        return JevModel(**parameters)
    if family == "upstream":
        from .upstream import UpstreamModel
        return UpstreamModel(model_id, parameters)
    if family == "algorithm":
        from .base import AlgorithmModel
        return AlgorithmModel(model_id, parameters)
    if family == "numerical":
        from .numerical import NumericalModel
        return NumericalModel(model_id, parameters)
    if family == "sequence":
        from .neural import NeuralForecastModel
        return NeuralForecastModel(model_id, **parameters)
    if family == "rl":
        from .reinforcement import RLModel
        return RLModel(model_id, parameters)
    if family == "fly":
        from .fly import FlyMemory
        return FlyMemory(**parameters)
    raise ValueError(f"unknown model adapter: {family!r}")


__all__ = ["model_catalog", "create_model", "load_model"]
