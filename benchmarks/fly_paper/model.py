"""Vectorized translation of Luo/Huang's published recurrent equation model.

SPDX-License-Identifier: GPL-3.0-or-later
Original: Copyright (C) 2024 Junjie Luo, Cheng Huang, Mark J. Schnitzer.
Python translation: 2026-09-21; source commit 5d7c08a9a88f923169a0c3008aca68af421e9a7f.
Source: https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model
Preserves the native solver initialization and ten fixed iterations, including
its matrix orientation. No new plasticity, reward, clipping, or trading rule.
This is an offline experimental-protocol simulator, NOT an online trading agent.
"""
from dataclasses import dataclass

import numpy as np

UPSTREAM_COMMIT = "5d7c08a9a88f923169a0c3008aca68af421e9a7f"


@dataclass(frozen=True)
class Bout:
    name: str
    duration: float
    odor: tuple[float, float]
    punishment: float = 0.0
    imaging: int = 0


def session(name, n=1, isi=120.0, cs=5.0):
    return [Bout(name, duration, odor, punishment, imaging)
            for _ in range(n)
            for duration, odor, punishment, imaging in (
                (cs, (1., 0.), float(name == "training"), 1 if name == "imaging" else 0),
                (isi, (0., 0.), 0., 0),
                (cs, (0., 1.), 0., 2 if name == "imaging" else 0),
                (isi, (0., 0.), 0., 0))]


def with_duration(bouts, index, seconds):
    b = bouts[index]
    bouts[index] = Bout(b.name, seconds, b.odor, b.punishment, b.imaging)


def figure5c_protocol():
    bouts = session("imaging")
    for _ in range(2):
        bouts += session("training", n=3, isi=135., cs=30.) + session("imaging")
    for rest in (3600-250-300, 7200-250, 75600-250):
        bouts += [Bout("rest", rest, (0., 0.))] + session("imaging")
    for index in (3, 15, 19, 31):
        with_duration(bouts, index, 300.)
    return bouts


def figure5e_protocol(training_count, rest_seconds):
    bouts = (session("imaging") + session("training", training_count, 135., 30.)
             + [Bout("rest", rest_seconds, (0., 0.))] + session("imaging"))
    with_duration(bouts, 3, 300.)
    return bouts


def parameter_matrices(vectors, bounds):
    """Native parameter_vec2mat, with a leading independent-sample dimension."""
    vectors = np.atleast_2d(vectors)
    matrices = []
    cursor = 0
    for limits in bounds.ravel(order="F"):
        lo, hi = limits[:, :, 0], limits[:, :, 1]
        indices = np.flatnonzero((lo != hi).ravel(order="F"))
        out = np.broadcast_to(lo.ravel(order="F"), (len(vectors), lo.size)).copy()
        out[:, indices] = vectors[:, cursor:cursor+len(indices)]
        matrices.append(out.reshape((len(vectors), *lo.shape), order="F"))
        cursor += len(indices)
    if cursor != vectors.shape[1]:
        raise ValueError("Parameter vector length disagrees with native bounds")
    return matrices


def reset_valence(params, valence=0.):
    """Native reset_parameter_by_valance; does not modify the caller's arrays."""
    params = [p.copy() for p in params]
    firing = np.array([0., 0., 0., 31.6, 5.8, 17.8])
    firing[:3] = valence
    params[0] = np.einsum("nij,j->ni", np.eye(6)-params[3].transpose(0,2,1), firing)[:,None,:]
    params[0] *= 2/(1+np.exp(-.05*5))
    return params


def simulate(params, protocol, *, trace=False):
    """Return native-shaped responses (samples, neuron, odor, imaging time).

    The author's three-hour retention switch is measured after the LAST named
    training bout, known from the entire experimental protocol. Preserve this
    here; a causal online adapter must explicitly replace that scheduling rule.
    """
    if any(b.duration < 0 for b in protocol):
        raise ValueError("Negative duration")
    training = [i for i,b in enumerate(protocol) if b.name == "training"]
    if not training:
        raise ValueError("Native retention schedule requires a named training bout")
    last_training = training[-1]
    size = len(params[0])
    base_weights = np.repeat(params[0], 2, axis=1)
    weights = base_weights.copy()
    recurrent = params[3]
    tau = params[4][:,0,:]
    adaptation_tau = params[5][:,0,0]
    fw0, fw_dt = params[1][:,0,0], params[2][:,0,0]
    punishment_weights = np.array([27.85, 0., 11.38, 0., 0., 0.])
    baseline = np.array([0., 0., 0., 35.2, 9., 11.2])
    ceiling = baseline[3:] + [36.46, 8.9, 19.96]
    inverse = np.linalg.inv(np.eye(6) - recurrent)  # native initialization, not W.T
    dw = np.zeros((size,2,3))
    odor_start = np.ones((size,2))
    responses = np.zeros((size,6,2,sum(b.imaging == 1 for b in protocol)))
    traces = {k: [] for k in ("w_odor_mean", "w_odor_end", "W_KDKM_KC",
                              "Dx_KC_mean", "Dx_DAN_MBON")} if trace else None
    image_index, elapsed = 0, 0.
    for index, bout in enumerate(protocol):
        odor = np.asarray(bout.odor)
        odor_end = odor_start * np.exp(-.05*bout.duration*odor)
        odor_end = 1 - (1-odor_end)*np.exp(-bout.duration/adaptation_tau[:,None])
        odor_mean = (odor_start+odor_end)/2
        odor_start = odor_end
        kc = odor_mean * odor
        external = np.einsum("noi,no->ni", weights, kc) + punishment_weights*bout.punishment
        activity = np.einsum("nij,nj->ni", inverse, external)
        for _ in range(10):
            rhs = external + baseline + np.einsum("nji,nj->ni", recurrent, activity)
            rhs[:,3:] = np.clip(rhs[:,3:], 0., ceiling)
            # Native fun_Dx returns rhs-baseline-activity; then adds activity.
            # Keep that order to test floating-point agreement with MATLAB.
            activity = (rhs-baseline-activity) + activity
        dopamine_input = (np.einsum("noj,no->nj", weights[:,:,:3], kc)
            + np.einsum("nmd,nm->nd", recurrent[:,3:,:3], activity[:,3:]))
        delta = (fw0[:,None]*dopamine_input
                 + fw_dt[:,None]*punishment_weights[:3]*bout.punishment)*bout.duration/90.
        dw += kc[:,:,None]*delta[:,None,:]
        if index > last_training:
            elapsed += bout.duration
        if elapsed <= 10800.:
            early, late = bout.duration, 0.
        elif 10800. > elapsed-bout.duration:
            early = 10800. - (elapsed-bout.duration)
            late = bout.duration-early
        else:
            early, late = 0., bout.duration
        dw *= np.exp(-early/tau[:,[0,1,1]])[:,None,:]
        dw *= np.exp(-late/tau[:,[0,2,2]])[:,None,:]
        weights = base_weights.copy()
        weights[:,:,3:6] += dw  # native effective weights may be negative
        if bout.imaging:
            responses[:,:,bout.imaging-1,image_index] = activity
            image_index += bout.imaging == 2
        if trace:
            for key, value in zip(traces, (odor_mean,odor_end,weights,kc,activity)):
                traces[key].append(value.copy())
    if trace:
        traces = {k: np.stack(v,axis=-1) for k,v in traces.items()}
    return responses, traces
