"""Algorithm inventory, explainable selection, execution and temporal forecast comparison.

    from fin_skills.algorithms import auto_run, catalog, recommend, Request
    report = recommend(Request("portfolio", ("asset_returns",), n_observations=250))
    result = auto_run("portfolio", {"asset_returns": returns}, objective="min_variance")

Catalog entries distinguish ready adapters, missing dependencies and catalog-only methods.
Routing ranks compatibility, not expected profit. Use walk_forward for forecast validation.
"""
from .core import Algorithm, Candidate, Registry, Request, Selection, TASKS
from .catalog import default_registry

_DEFAULT = default_registry()
catalog = _DEFAULT.catalog
recommend = _DEFAULT.recommend
run = _DEFAULT.run
auto_run = _DEFAULT.auto_run

# Imported last so evaluation can lazily use the default registry without import cycles.
from .evaluation import walk_forward
from .models import FittedModel, fit, load_model
from .research import ResearchResult, profile_data, research
from .diagnostics import doctor

__all__ = ["Algorithm", "Candidate", "Registry", "Request", "Selection", "TASKS",
           "auto_run", "catalog", "default_registry", "recommend", "run", "walk_forward",
           "FittedModel", "fit", "load_model", "ResearchResult", "profile_data", "research", "doctor"]
