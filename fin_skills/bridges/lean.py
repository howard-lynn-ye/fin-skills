"""Run a caller-owned LEAN algorithm through the real local engine.

This bridge is Python-only: algorithm files execute trusted user code. It never
downloads data, installs an engine, sends orders or submits a cloud job by itself.
"""
from pathlib import Path
import json
import os
import subprocess


class LeanBacktest:
    def __init__(self, *, engine_dir, dotnet, template_config):
        self.engine_dir = Path(engine_dir).resolve(strict=True)
        self.dotnet = Path(dotnet).resolve(strict=True)
        self.template_config = Path(template_config).resolve(strict=True)
        self.launcher = self.engine_dir / "QuantConnect.Lean.Launcher.dll"
        if not self.launcher.is_file():
            raise ValueError("engine_dir must contain a built LEAN launcher")

    def run(self, *, algorithm_file, algorithm_type, data_folder, work_dir,
            language="CSharp", parameters=None, timeout=300):
        """Return paths and completion metadata; reject runtime errors even with exit code 0."""
        import json5
        if language not in ("CSharp", "Python"):
            raise ValueError("language must be CSharp or Python")
        algorithm = Path(algorithm_file).resolve(strict=True)
        data = Path(data_folder).resolve(strict=True)
        work = Path(work_dir).resolve()
        if work.exists():
            raise FileExistsError("use a new work_dir for each backtest")
        if not algorithm_type or not isinstance(algorithm_type, str):
            raise ValueError("algorithm_type must name a class in the supplied file")
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
            raise ValueError("timeout must be positive and at most one day")
        config = json5.loads(self.template_config.read_text(encoding="utf-8"))
        # Retain upstream engine handlers but remove other environment/provider credentials.
        environment = config.get("environments", {}).get("backtesting")
        if not environment:
            raise ValueError("template must define the native backtesting environment")
        allowed = ("environment", "algorithm-type-name", "algorithm-language", "algorithm-location",
                   "data-folder", "log-handler", "messaging-handler", "job-queue-handler",
                   "api-handler", "map-file-provider", "factor-file-provider", "data-provider",
                   "data-channel-provider", "object-store", "data-aggregator")
        config = {k: v for k, v in config.items() if k in allowed}
        config.update(environment)
        config.update({"environment": "backtesting", "live-mode": False,
                       "algorithm-type-name": algorithm_type, "algorithm-language": language,
                       "algorithm-location": str(algorithm), "data-folder": str(data),
                       "results-destination-folder": str(work), "object-store-root": str(work / "storage"),
                       "log-file": str(work / "engine.log"), "job-user-id": "0",
                       "api-access-token": "", "job-organization-id": "",
                       "composer-dll-directory": str(self.engine_dir),
                       "parameters": dict(parameters or {}), "debugging": False,
                       "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider"})
        work.mkdir(parents=True)
        config_path = work / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        env = os.environ.copy()
        env.update(DOTNET_CLI_HOME=str(work / "dotnet-cache"),
                   DOTNET_CLI_TELEMETRY_OPTOUT="1", XDG_CACHE_HOME=str(work / "cache"))
        with (work / "stdout.log").open("w", encoding="utf-8") as output:
            result = subprocess.run([str(self.dotnet), str(self.launcher), "--config", str(config_path)],
                cwd=self.engine_dir, env=env, stdout=output, stderr=subprocess.STDOUT,
                timeout=timeout, check=False)
        summaries = sorted(work.glob("*-summary.json"))
        if result.returncode != 0 or not summaries:
            raise RuntimeError(f"LEAN backtest did not complete; see {work / 'stdout.log'}")
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
        state = summary.get("state", {})
        error = summary.get("runtimeError") or (state.get("runtimeError") if isinstance(state, dict) else None)
        if error:
            raise RuntimeError(f"LEAN runtime error: {error}; see {work / 'stdout.log'}")
        return dict(engine="QuantConnect LEAN", returncode=result.returncode,
                    summary_path=str(summaries[-1]), output_dir=str(work),
                    algorithm_file=str(algorithm), live_mode=False)
