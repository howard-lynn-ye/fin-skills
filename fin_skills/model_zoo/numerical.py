"""Validated adapters for existing scientific model functions; no new model equations."""
import numpy as np

from fin_skills.algorithms.runtime import _array, params, number, integer
from .base import ModelArtifact
from .catalog import NUMERICAL


class NumericalModel(ModelArtifact):
    def __init__(self, model_id, parameters):
        self.model_id, self.parameters = model_id, dict(parameters)

    def run(self, data):
        required = set(NUMERICAL[self.model_id][1])
        if set(data) != required:
            raise ValueError(f"{self.model_id} requires exactly {sorted(required)}")
        name, p = self.model_id, self.parameters
        if name.endswith("covariance"):
            from fin_skills.models import risk_model as r
            X = np.asarray(_array(data["asset_returns"], "asset_returns", 2))
            if len(X) < 2 or (X < -1).any() or np.max(np.abs(X.mean(axis=0))) > .1:
                raise ValueError("at least two rows of decimal simple returns required")
            if name == "ledoit_wolf_covariance":
                p = params(p, {"target": "identity"})
                cov, shrinkage = r.ledoit_wolf(X, target=p["target"])
                return {"covariance": cov, "shrinkage": shrinkage}
            if name == "ewma_covariance":
                p = params(p, {"decay": .94})
                decay = number(p["decay"], "decay", minimum=0, maximum=1)
                if not 0 < decay < 1:
                    raise ValueError("decay must be strictly between 0 and 1")
                return {"covariance": r.ewma_cov(X, lam=decay)}
            p = params(p, {"factors": 1})
            k = integer(p["factors"], "factors", maximum=min(X.shape))
            cov, loadings, eigenvalues = r.pca_factor_cov(X, k)
            return {"covariance": cov, "loadings": loadings, "eigenvalues": eigenvalues}
        if name in ("nelson_siegel", "svensson"):
            from fin_skills.models import term_structure as t
            x = np.asarray(_array(data["maturities"], "maturities", 1))
            y = np.asarray(_array(data["yields"], "yields", 1))
            if x.shape != y.shape or (x <= 0).any() or (np.diff(x) <= 0).any():
                raise ValueError("positive increasing maturities and matching decimal yields required")
            if name == "nelson_siegel":
                p = params(p, {"lam": None})
                if p["lam"] is not None:
                    number(p["lam"], "lam", minimum=1e-8)
                if len(x) < (3 if p["lam"] is not None else 4):
                    raise ValueError("insufficient curve points")
                return t.fit_nelson_siegel(x, y, lam=p["lam"])
            params(p, {})
            if len(x) < 6:
                raise ValueError("Svensson requires at least six points")
            return t.fit_svensson(x, y)
        if name == "merton_credit":
            from fin_skills.models.credit_models import merton_solve
            params(p, {})
            clean = {k: number(v, k) for k, v in data.items()}
            return merton_solve(**clean).as_dict()
        if name == "svi_surface":
            from fin_skills.models.vol_surface import fit_svi
            x = np.asarray(_array(data["log_moneyness"], "log_moneyness", 1))
            w = np.asarray(_array(data["total_variance"], "total_variance", 1))
            if x.shape != w.shape or len(x) < 5 or (np.diff(x) <= 0).any() or (w <= 0).any():
                raise ValueError("need >= 5 ordered distinct strikes and positive total variances")
            p = params(p, {"starts": 6})
            return fit_svi(x, w, starts=integer(p["starts"], "starts", maximum=20))
        from fin_skills.models.kalman_models import kalman_filter
        params(p, {})
        clean = {k: np.asarray(_array(data[k], k, 1 if k in ("y", "a0") else 2))
                 for k in required - {"H"}}
        H = number(data["H"], "H", minimum=1e-12)
        n, k = clean["Z"].shape
        if (clean["y"].shape != (n,) or clean["a0"].shape != (k,) or
                any(clean[key].shape != (k, k) for key in ("T", "Q", "P0"))):
            raise ValueError("inconsistent state-space dimensions")
        for key in ("Q", "P0"):
            a = clean[key]
            if not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() < -1e-12:
                raise ValueError(f"{key} must be symmetric positive semidefinite")
        return kalman_filter(**clean, H=H)
