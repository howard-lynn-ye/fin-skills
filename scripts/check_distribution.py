"""Build/check wheel and sdist, then install each in a fresh environment outside the repo."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SMOKE = r'''
import json
from pathlib import Path
import fin_skills
from fin_skills.algorithms import auto_run, fit, load_model, research, catalog
assert fin_skills.load('backtest-validation')
assert len(catalog()) >= 1
assert auto_run('forecast', {'series': [1., 2., 3.]})['result'][0] == 3
r = research('forecast', {'series': list(range(40))}, initial_train=10, horizon=5,
             holdout=10, candidates=('naive', 'drift'))
assert r.selected == 'drift' and r.validation['holdout']['score'] == 0
model = fit('drift', {'series': [1., 2., 3.]})
path = model.save('model.zip')
assert load_model(path, trusted=True).predict(horizon=1)[0] == 4
r.save('research.json')
assert json.loads(Path('research.json').read_text())['schema_version'] == 1
print('INSTALLED_SMOKE_OK', fin_skills.__file__)
'''


def run(args, cwd):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run(args, cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fin-skills-dist-") as temp:
        root = Path(temp)
        out = args.output.resolve() if args.output else root / "dist"
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise ValueError("distribution output must be empty to avoid checking stale artifacts")
        run([sys.executable, "-m", "build", "--outdir", str(out), str(ROOT)], root)
        artifacts = sorted([*out.glob("*.whl"), *out.glob("*.tar.gz")])
        if len(artifacts) != 2:
            raise ValueError("expected exactly one wheel and one sdist")
        run([sys.executable, "-m", "twine", "check", "--strict", *map(str, artifacts)], root)
        with zipfile.ZipFile(next(out.glob("*.whl"))) as archive:
            names = archive.namelist()
            assert "fin_skills/algorithms/research.py" in names
            assert "fin_skills/_skills/index.json" in names
            assert not any(n.startswith(("tests/", ".venv", "benchmarks/")) for n in names)
        for i, artifact in enumerate(artifacts):
            env_dir = root / f"env{i}"
            venv.EnvBuilder(with_pip=True).create(env_dir)
            python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            work = root / f"work{i}"
            work.mkdir()
            run([str(python), "-m", "pip", "install", str(artifact)], work)
            run([str(python), "-I", "-c", SMOKE], work)
        print("DISTRIBUTION_OK: wheel and sdist installed and exercised outside repository")


if __name__ == "__main__":
    main()
