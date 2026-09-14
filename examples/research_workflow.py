"""Run a complete forecast study and model roundtrip without network access."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from fin_skills.algorithms import fit, load_model, research
from fin_skills.data.provenance import make


def main():
    series = pd.Series(np.arange(100.) + np.random.default_rng(17).normal(0, .2, 100),
                       index=pd.date_range("2020-01-01", periods=100))
    provenance = make("synthetic-example", library_version=np.__version__, request={"seed": 17},
                      content=series, redistributable=True)
    with tempfile.TemporaryDirectory(prefix="fin-skills-example-") as directory:
        directory = Path(directory)
        result = research("forecast", {"series": series}, initial_train=40, horizon=10,
                          holdout=20, gap=1, candidates=("naive", "mean", "drift"),
                          provenance=provenance, ledger_path=directory / "trials.jsonl")
        assert result.status == "completed" and result.validation["holdout"] is not None
        result.save(directory / "research.json")
        # This is a separate deployment fit AFTER the evaluation; it does not replace test results.
        model = fit(result.selected, {"series": series})
        saved = model.save(directory / "model.zip")
        restored = load_model(saved, trusted=True)
        np.testing.assert_allclose(restored.predict(horizon=3), model.predict(horizon=3))
        print(result.render())
        print("MODEL_ROUNDTRIP_OK")
        print("TAKEAWAY\n  Selection uses development folds only.\n"
              "  Holdout scores assess the frozen choice.\n"
              "  Saved model predictions survive a trusted roundtrip.")


if __name__ == "__main__":
    main()
