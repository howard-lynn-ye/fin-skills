"""GPL-3.0-or-later optional extension. No pretrained weights are distributed.

Original equations: Copyright 2024 Junjie Luo, Cheng Huang, Mark J. Schnitzer.
See COPYING.txt, model.py, and online.py for provenance and adaptation details.
"""
from .online import CausalMemory


def load_parameters(path):
    """Read the caller's trusted published .mat parameter file; no network download."""
    from scipy.io import loadmat
    from .model import parameter_matrices, reset_valence
    data = loadmat(path)
    return reset_valence(parameter_matrices(data["para_mu"].T, data["mat_lu_cell"]), 0.)


__all__ = ["CausalMemory", "load_parameters"]
