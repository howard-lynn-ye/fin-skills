"""Base-dependency model discovery and execution; no network or file writes."""
import numpy as np

from fin_skills.model_zoo import create_model, model_catalog


def main():
    print([(c["id"], c["operations"], c["status"])
           for c in model_catalog("forecast") if c["adapter"] == "sequence"])
    model = create_model("naive").fit({"series": np.arange(20.)})
    print("forecast", model.predict(horizon=3))
    returns = np.random.default_rng(7).normal(0, .01, (100, 3))
    risk = create_model("ledoit_wolf_covariance").run({"asset_returns": returns})
    print("covariance", risk["covariance"])
    print("TAKEAWAY")
    print("Discovery reports optional dependencies without installing them.")
    print("Forecast adapters fit history; numerical adapters run explicit inputs.")
    print("Interface output alone does not establish trading profitability.")


if __name__ == "__main__":
    main()
