"""Model discovery and stateless numeric execution; no arbitrary code or artifact loading."""


def list_models(task=None):
    from fin_skills.model_zoo import model_catalog
    return {"models": model_catalog(task),
            "scope": "ready means adapter and discoverable dependencies, not verified profitability"}


def run_model(model_id, data, parameters=None):
    from fin_skills.model_zoo import model_catalog, create_model
    from fin_skills.tools.algorithms import _decode, _encode
    cards = {c["id"]: c for c in model_catalog()}
    if model_id not in cards:
        raise KeyError(f"unknown model: {model_id}")
    if not cards[model_id]["json_run"]:
        raise ValueError("this model requires a stateful Python lifecycle; see its operations")
    if cards[model_id]["adapter"] == "decision":
        options = dict(parameters or {})
        if set(options) - {"model", "allow_network", "timeout"}:
            raise ValueError("Jev JSON parameters allow only model, allow_network and timeout")
        return {"model_id": model_id, "result": create_model(model_id, **options).run(data)}
    return {"model_id": model_id,
            "result": _encode(create_model(model_id, **dict(parameters or {})).run(_decode(data)))}


FUNCTIONS = {"list_models": list_models, "run_model": run_model}


def definitions():
    return [
        {"name": "list_models", "description": "Discover numerical, prediction, RL, memory and "
         "hosted Jev decision models, supported operations, dependencies and access requirements.",
         "input_schema": {"type": "object", "properties": {"task": {"type": "string"}},
                          "additionalProperties": False}},
        {"name": "run_model", "description": "Execute a stateless model adapter on explicit numeric "
         "inputs, or Jev on explicit state/questions. Hosted Jev requires parameters.allow_network=true "
         "and TYPESAFE_API_KEY; it sends supplied state/questions to TypeSafe. Stateful neural, RL "
         "and memory models require the Python API; no file loading.",
         "input_schema": {"type": "object", "properties": {
             "model_id": {"type": "string"}, "data": {"type": "object"},
             "parameters": {"type": "object"}}, "required": ["model_id", "data"],
             "additionalProperties": False}},
    ]
