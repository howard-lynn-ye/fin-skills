"""AutoGen adapters for model-chosen calls to the existing fin-skills tools.

The caller supplies a model client and explicitly binds datasets/environments.
No new LLM transport, agent loop, broker, file loader or ranking policy lives here.
"""
import asyncio
import copy
import math
from typing import Any

from fin_skills.algorithms.runtime import integer

DEFAULT_TOOLS = ("list_models", "run_model", "list_algorithms", "run_algorithm",
                 "recommend_algorithms", "recommend_trading_strategy", "describe_guard",
                 "search_quant_methods", "get_quant_method", "quant_method_coverage")


class ModelSession:
    """Process-local model handles over caller-bound training/prediction data.

    datasets maps names to (role, data), where role is train, predict or backtest. These labels
are permissions, not proof that the supplied observations are point-in-time safe.
Environment factories are trusted zero-argument Python callables, never model code.
"""
    def __init__(self, datasets=None, *, environment_factories=None, max_models=8,
                 max_training_steps=1000):
        self.datasets = copy.deepcopy(dict(datasets or {}))
        if any(role not in ("train", "predict", "backtest") for role, _ in self.datasets.values()):
            raise ValueError("dataset roles must be train, predict or backtest")
        self.environments = dict(environment_factories or {})
        if set(self.datasets) & set(self.environments) or not all(
                callable(f) for f in self.environments.values()):
            raise ValueError("environment names must be distinct and factories callable")
        self.max_models = integer(max_models, "max_models", maximum=100)
        self.max_training_steps = integer(max_training_steps, "max_training_steps", maximum=100000)
        self.models = {}
        self.receipts = []

    def list_datasets(self) -> dict:
        """List bound data names and permissions without exposing labels or file paths."""
        return {"datasets": [{"id": k, "role": v[0]} for k, v in self.datasets.items()],
                "training_environments": list(self.environments),
                "max_training_steps": self.max_training_steps}

    def fit_model(self, model_id: str, data_id: str, handle: str,
                  parameters: dict[str, Any] | None = None, total_timesteps: int = 128) -> dict:
        """Train a chosen model on a bound train dataset/environment; keep an in-memory handle."""
        from fin_skills.model_zoo import create_model, model_catalog
        if not handle or handle in self.models or len(self.models) >= self.max_models:
            raise ValueError("use a new nonempty handle within the model-count budget")
        cards = {c["id"]: c for c in model_catalog()}
        if model_id not in cards or "fit" not in cards[model_id]["operations"]:
            raise ValueError("model has no fit operation; use run_model for numerical methods")
        parameters = dict(parameters or {})
        if cards[model_id]["adapter"] == "sequence":
            steps = parameters.setdefault("max_steps", min(100, self.max_training_steps))
            integer(steps, "max_steps", maximum=self.max_training_steps)
            # Keep agent-driven training bounded; advanced Python callers may use the backend directly.
            permitted = {"freq", "h", "input_size", "max_steps", "seed", "learning_rate"}
            if set(parameters) - permitted:
                raise ValueError("agent neural parameters exceed the supported bounded interface")
            integer(parameters.get("input_size", 24), "input_size", maximum=512)
            integer(parameters.get("h", 1), "h", maximum=128)
        model = create_model(model_id, **parameters)
        if cards[model_id]["adapter"] == "rl":
            if data_id not in self.environments:
                raise ValueError("RL requires a caller-bound training environment")
            steps = integer(total_timesteps, "total_timesteps", maximum=self.max_training_steps)
            permitted = {"seed", "learning_rate", "n_steps", "batch_size", "n_epochs",
                         "buffer_size", "learning_starts"}
            if set(parameters) - permitted:
                raise ValueError("agent RL parameters exceed the supported bounded interface")
            # SB3 PPO rounds learn() up to a full rollout. Bound the rollout as well.
            if model_id in ("ppo", "a2c"):
                rollout = integer(parameters.get("n_steps", min(128, self.max_training_steps)),
                                  "n_steps", minimum=2)
                if math.ceil(steps / rollout) * rollout > self.max_training_steps:
                    raise ValueError("PPO rollout exceeds training budget")
                model.parameters.setdefault("n_steps", rollout)
                if model_id == "ppo":
                    model.parameters.setdefault("batch_size", min(64, rollout))
                    integer(parameters.get("n_epochs", 10), "n_epochs", maximum=10)
                elif set(parameters) & {"batch_size", "n_epochs"}:
                    raise ValueError("A2C does not accept batch_size or n_epochs")
            else:
                integer(parameters.get("buffer_size", 10000), "buffer_size", maximum=100000)
                model.parameters.setdefault("buffer_size", 10000)
            env = self.environments[data_id]()
            try:
                if getattr(env, "num_envs", 1) != 1:
                    raise ValueError("agent training budget currently supports one environment")
                model.fit(env, total_timesteps=steps)
            finally:
                env.close()
        else:
            if data_id not in self.datasets or self.datasets[data_id][0] != "train":
                raise ValueError("fit requires a bound train dataset; prediction data cannot train")
            model.fit(copy.deepcopy(self.datasets[data_id][1]))
        self.models[handle] = model
        receipt = {"operation": "fit", "model_id": model_id, "data_id": data_id, "handle": handle}
        self.receipts.append(receipt)
        return dict(receipt)

    def predict_model(self, handle: str, data_id: str | None = None, horizon: int = 1) -> dict:
        """Predict using an existing handle and optional caller-bound prediction data."""
        from fin_skills.tools.algorithms import _encode
        if handle not in self.models:
            raise ValueError("unknown model handle")
        data = None
        if data_id is not None:
            if data_id not in self.datasets or self.datasets[data_id][0] != "predict":
                raise ValueError("prediction inputs require a bound predict dataset")
            data = copy.deepcopy(self.datasets[data_id][1])
        model = self.models[handle]
        if hasattr(model, "forecaster"):
            result = model.predict(data)
        elif model.model_id in ("ppo", "sac", "a2c", "ddpg", "dqn", "td3"):
            if data is None:
                raise ValueError("policy prediction needs a bound observation")
            result = model.predict(data)
        else:
            result = model.predict(data, horizon=horizon)
        receipt = {"operation": "predict", "model_id": model.model_id, "handle": handle,
                   "data_id": data_id, "result": _encode(result)}
        self.receipts.append(receipt)
        return copy.deepcopy(receipt)

    def backtest_strategy(self, data_id: str, topk: int, n_drop: int, account: float,
                          open_cost: float, close_cost: float, min_cost: float) -> dict:
        """Run native Qlib TopkDropout on bound scores/clocks/start/end after caller initializes Qlib."""
        from fin_skills.bridges.qlib_strategy import qlib_backtest
        from fin_skills.tools.algorithms import _encode
        if data_id not in self.datasets or self.datasets[data_id][0] != "backtest":
            raise ValueError("backtest requires a bound backtest dataset")
        data = copy.deepcopy(self.datasets[data_id][1])
        if not isinstance(data, dict) or set(data) != {"scores", "available_at", "start", "end"}:
            raise ValueError("backtest dataset must contain scores, available_at, start, end")
        result = qlib_backtest(**data, topk=topk, n_drop=n_drop, account=account,
                              open_cost=open_cost, close_cost=close_cost, min_cost=min_cost)
        receipt = {"operation": "backtest", "data_id": data_id, "result": _encode(
            {k: result[k] for k in ("report", "net_returns", "nav", "provenance")})}
        self.receipts.append(receipt)
        return copy.deepcopy(receipt)


def make_tool_agent(model_client, *, session=None, allowed_tools=DEFAULT_TOOLS,
                    max_tool_iterations=8):
    """Return upstream AssistantAgent; model_client decides which tools to call.

Use a fresh agent/session for independent research runs. The supplied client's
provider receives task text, tool arguments and results; bind only intended data.
"""
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import BaseTool, FunctionTool
    from pydantic import BaseModel
    import jsonschema
    from fin_skills.tools import list_tools, call_tool
    definitions = {d["name"]: d for d in list_tools()}
    allowed = tuple(allowed_tools)
    if len(set(allowed)) != len(allowed) or set(allowed) - definitions.keys():
        raise ValueError("allowed_tools must be distinct registered tool names")
    limit = integer(max_tool_iterations, "max_tool_iterations", maximum=100)
    lock = asyncio.Lock()

    class RegistryTool(BaseTool):
        def __init__(self, definition):
            self.definition = copy.deepcopy(definition)
            super().__init__(BaseModel, dict, definition["name"], definition["description"])

        @property
        def schema(self):
            return {"name": self.name, "description": self.description,
                    "parameters": copy.deepcopy(self.definition["input_schema"])}

        async def run(self, args, cancellation_token):
            return call_tool(self.name, args.model_dump())

        async def run_json(self, args, cancellation_token, call_id=None):
            jsonschema.validate(args, self.definition["input_schema"])
            async with lock:
                if cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()
                return call_tool(self.name, args)

    class SessionTool(FunctionTool):
        async def run(self, args, cancellation_token):
            async with lock:
                if cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()
                return await super().run(args, cancellation_token)

    tools = [RegistryTool(definitions[n]) for n in allowed]
    if session is not None:
        tools.extend(SessionTool(fn, description=fn.__doc__) for fn in
                     (session.list_datasets, session.fit_model, session.predict_model,
                      session.backtest_strategy))
    return AssistantAgent("fin_research", model_client=model_client, tools=tools,
        max_tool_iterations=limit, reflect_on_tool_use=True,
        system_message="You are a quantitative research assistant. Discover available models and "
        "datasets, choose tools appropriate to the user's task, execute them, and inspect results. "
        "Recommendations are advisory; you decide which supported method to use. Respect data "
        "roles and timestamps. Report actual tool results and limitations. A tool failure is not "
        "a successful result. Do not infer profitability from interface tests.")
