"""Stable-Baselines3 policies for explicit caller-supplied Gymnasium environments."""
import io

from fin_skills.algorithms.runtime import integer
from .base import ModelArtifact
from .upstream_catalog import RL

CLASSES = {"ppo": "PPO", "sac": "SAC", **RL}


class RLModel(ModelArtifact):
    dependencies = ("stable-baselines3", "gymnasium", "torch")

    def __init__(self, model_id, parameters):
        self.model_id, self.parameters = model_id, dict(parameters)
        self.estimator = None

    def fit(self, env, *, total_timesteps=1000):
        import gymnasium as gym
        import stable_baselines3 as sb3
        from stable_baselines3.common.vec_env import VecEnv
        if not isinstance(env, (gym.Env, VecEnv)):
            raise TypeError("fit requires a live Gymnasium Env or SB3 VecEnv")
        if self.model_id in ("sac", "td3", "ddpg") and not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("SAC/TD3/DDPG require a continuous Box action space")
        if self.model_id == "dqn" and not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DQN requires a Discrete action space")
        steps = integer(total_timesteps, "total_timesteps", maximum=10_000_000)
        p = dict(device="cpu", seed=0, verbose=0)
        p.update(self.parameters)
        policy = p.pop("policy", "MlpPolicy")
        cls = getattr(sb3, CLASSES[self.model_id])
        if self.model_id in ("sac", "td3", "ddpg", "dqn"):
            p.setdefault("train_freq", (1, "step"))
        model = cls(policy, env, **p)
        model.learn(total_timesteps=steps)
        self.estimator = model
        return self

    def predict(self, observation, *, deterministic=True):
        if self.estimator is None:
            raise ValueError("fit or load the policy before prediction")
        return self.estimator.predict(observation, deterministic=deterministic)[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("estimator")
        if model is not None:
            stream = io.BytesIO()
            model.save(stream)
            state["policy_bytes"] = stream.getvalue()
        return state

    def __setstate__(self, state):
        import stable_baselines3 as sb3
        blob = state.pop("policy_bytes", None)
        self.__dict__.update(state)
        cls = getattr(sb3, CLASSES[self.model_id])
        self.estimator = None if blob is None else cls.load(io.BytesIO(blob), device="cpu")
