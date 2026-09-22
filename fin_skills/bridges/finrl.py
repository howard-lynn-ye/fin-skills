"""Lazy access to the optional, next-open FinRL environment adapter."""


def make_finrl_env(frame, **parameters):
    """Reuse installed FinRL with next-open fills and independently checked NAV."""
    from fin_skills.model_zoo._finrl import make_finrl_env as create
    return create(frame, **parameters)
