"""Causal stateful adaptation of the published memory equations, not a reproduction.

SPDX-License-Identifier: GPL-3.0-or-later
Original equations copyright 2024 Junjie Luo, Cheng Huang, Mark J. Schnitzer.
Adaptation 2026-09-21: a past-only training clock and non-mutating readout.
Signed financial reinforcement and readout choice are engineering hypotheses.
Biological seconds here are assay units, not a claimed conversion to market time.
"""
import numpy as np


class CausalMemory:
    def __init__(self, params):
        if params[0].shape != (1,1,6):
            raise ValueError("One fitted circuit and one six-neuron cue vector required")
        self.initial = np.repeat(params[0][0],2,axis=0)
        self.weights = self.initial.copy()
        self.recurrent = params[3][0].copy()
        self.inverse = np.linalg.inv(np.eye(6)-self.recurrent)
        self.fw0 = float(params[1][0,0,0])
        self.fw_dt = float(params[2][0,0,0])
        self.tau = params[4][0,0].copy()
        self.adapt_tau = float(params[5][0,0,0])
        self.odor_state = np.ones(2)
        self.delta_weights = np.zeros((2,3))
        self.clock = 0.
        self.last_conditioning_end = None
        self.baseline = np.array([0.,0.,0.,35.2,9.,11.2])
        self.ceiling = self.baseline[3:]+[36.46,8.9,19.96]
        self.punishment_weights = np.array([27.85,0.,11.38,0.,0.,0.])

    def activity(self, kc, punishment=0.):
        external = self.weights.T@kc + self.punishment_weights*punishment
        x = self.inverse@external
        for _ in range(10):
            rhs = external+self.baseline+self.recurrent.T@x
            rhs[3:] = np.clip(rhs[3:],0.,self.ceiling)
            x = (rhs-self.baseline-x)+x
        return x

    def step(self, duration, odor=(0.,0.), punishment=0., *, conditioning=False):
        """Consume only the current event, without a future experimental schedule.

        Each observed conditioning event resets the retention clock at its end.
        Before/within a conditioning bout use the early retention constants.
        This explicit change from the native last-training schedule is logged.
        """
        odor = np.asarray(odor,dtype=float)
        if (duration < 0 or not np.isfinite(duration) or not np.isfinite(punishment)
                or odor.shape != (2,) or not np.isfinite(odor).all() or (odor<0).any()):
            raise ValueError("Invalid current event")
        end = self.odor_state*np.exp(-.05*duration*odor)
        end = 1-(1-end)*np.exp(-duration/self.adapt_tau)
        kc = (self.odor_state+end)/2*odor
        self.odor_state = end
        x = self.activity(kc,punishment)
        dopamine = self.weights[:,:3].T@kc+self.recurrent[3:,:3].T@x[3:]
        delta = (self.fw0*dopamine + self.fw_dt*self.punishment_weights[:3]*punishment)*duration/90.
        self.delta_weights += kc[:,None]*delta
        age = 0. if self.last_conditioning_end is None else self.clock-self.last_conditioning_end
        early = duration if conditioning else min(duration,max(0.,10800.-age))
        late = duration-early
        self.delta_weights *= np.exp(-early/self.tau[[0,1,1]])
        self.delta_weights *= np.exp(-late/self.tau[[0,2,2]])
        self.weights = self.initial.copy()
        self.weights[:,3:] += self.delta_weights
        self.clock += duration
        if conditioning:
            self.last_conditioning_end = self.clock
        if not np.isfinite(self.weights).all():
            raise ArithmeticError("Nonfinite memory weights")
        return x

    def cue_scores(self):
        """Engineering probe: mean normalized MBON response, all three modules.

        A larger response is treated as approach relative to the unconditioned
        comparison cue. This is not a calibrated expected financial return.
        Probe responses do not themselves alter memory or simulate ingestion.
        """
        return np.array([np.mean(self.activity(np.eye(2)[i])[3:]/self.ceiling)
                         for i in range(2)])

    def reinforce_return(self, log_return, *, scale=.01, cue=0):
        if not np.isfinite(log_return) or scale <= 0 or not np.isfinite(scale):
            raise ValueError("Invalid return or normalization scale")
        if cue not in (0,1):
            raise ValueError("Expected one of the two odor/action channels")
        # Positive financial return -> negative punishment drive. This signed
        # extension is a hypothesis, not a published reward-trading circuit.
        self.step(30.,np.eye(2)[cue],-log_return/scale,conditioning=True)
        self.step(135.)
