"""Optional GPL memory extension accessed through a checked-feedback adapter.

No biological equations or fitted parameters are bundled into the MIT core.
This is an experimental associative memory interface, not a trading policy.
"""
import copy

import numpy as np
import pandas as pd

from fin_skills.algorithms.runtime import number
from .base import ModelArtifact
from .feedback import CreditGate


class FlyMemory(ModelArtifact):
    model_id = "fly_memory"
    dependencies = ("fin-skills-fly",)

    def __init__(self, *, circuit_parameters, return_scale=.01):
        from fin_skills_fly import CausalMemory
        self.return_scale = number(return_scale, "return_scale", minimum=1e-12)
        if not isinstance(circuit_parameters, (list, tuple)) or len(circuit_parameters) != 6:
            raise ValueError("six explicit circuit parameter matrices required")
        arrays = [np.asarray(a, dtype=float).copy() for a in circuit_parameters]
        shapes = {0: (1, 1, 6), 3: (1, 6, 6), 4: (1, 1, 3), 5: (1, 1, 1)}
        if any(a.ndim != 3 or a.shape[0] != 1 or not a.size or not np.isfinite(a).all()
               for a in arrays) or any(arrays[i].shape != s for i, s in shapes.items()):
            raise ValueError("invalid finite single-circuit parameter shapes")
        if (arrays[4] <= 0).any() or (arrays[5] <= 0).any():
            raise ValueError("retention/adaptation time constants must be positive")
        self.memory = CausalMemory(arrays)
        self.gate = CreditGate()
        self.last_update = None

    def predict(self):
        return self.memory.cue_scores().copy()

    def update(self, receipt, *, now, cue=0):
        if type(cue) is not int or cue not in (0, 1):
            raise ValueError("cue must identify channel 0 or 1")
        now = pd.Timestamp(now)
        if pd.isna(now) or self.last_update is not None and now < self.last_update:
            raise ValueError("update clock must be valid and nondecreasing")
        # A failed numerical update must not consume an otherwise valid receipt.
        gate, memory = copy.deepcopy(self.gate), copy.deepcopy(self.memory)
        result = gate.consume(receipt, now)
        if result["status"] == "ready":
            memory.reinforce_return(result["target"], scale=self.return_scale, cue=cue)
            if not np.isfinite(memory.cue_scores()).all():
                raise ArithmeticError("nonfinite memory readout")
            self.gate, self.memory, self.last_update = gate, memory, now
        return result
