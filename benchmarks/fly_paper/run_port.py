"""Replay published fitted parameters, compare to native functions, plot evidence.

SPDX-License-Identifier: GPL-3.0-or-later
Original model copyright 2024 Junjie Luo, Cheng Huang, Mark J. Schnitzer.
Reproduction harness and Python translation added 2026-09-21.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.fly_paper.model import (
    UPSTREAM_COMMIT, figure5c_protocol, figure5e_protocol, parameter_matrices,
    reset_valence, simulate,
)


def original_data(upstream):
    """Match xlsread's numeric bounding rectangle and native load_original_data."""
    values = []
    for sheet in ("ACVvsETA","OCTvsBEN"):
        frame = pd.read_excel(upstream/"data_and_parameters/Imaging_24hr_data.xlsx",
                              sheet_name=sheet,header=None)
        numeric = frame.apply(pd.to_numeric,errors="coerce").to_numpy()
        rows,cols = np.where(np.isfinite(numeric))
        numeric = numeric[rows.min():rows.max()+1,cols.min():cols.max()+1]
        if numeric.shape != (17,14):
            raise ValueError(f"Unexpected published spreadsheet layout: {numeric.shape}")
        values.append(numeric)
    raw = np.stack(values,axis=-1)
    mean = np.stack([raw[0::3,:6,:],raw[0::3,8:14,:]],axis=1)
    sem = np.stack([raw[1::3,:6,:],raw[1::3,8:14,:]],axis=1)
    return mean,sem


def compare(actual, expected, label, checks):
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: {actual.shape} != {expected.shape}")
    finite = np.isfinite(actual)&np.isfinite(expected)
    same_nonfinite = np.array_equal(np.isnan(actual),np.isnan(expected)) and np.array_equal(
        np.isinf(actual),np.isinf(expected))
    difference = np.abs(actual[finite]-expected[finite])
    passed = same_nonfinite and np.allclose(actual,expected,rtol=1e-9,atol=1e-8,equal_nan=True)
    checks.append({"label":label,"scalars":actual.size,"finite_scalars":int(finite.sum()),
                   "max_abs_error":float(difference.max()) if len(difference) else None,
                   "passed":bool(passed)})
    return passed


def plot_results(output, arrays, observed, sem, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["DAN gamma1","DAN alpha2","DAN alpha3", "MBON gamma1","MBON alpha2","MBON alpha3"]
    labels = ["Pre","3x","6x","1 h","3 h","24 h"]
    x = np.arange(6)
    fig,axes = plt.subplots(3,2,figsize=(11,9),sharex=True)
    bands = arrays[3]["bands"]
    for i,ax in enumerate(axes.flat):
        for odor,color in ((0,"#185fa5"),(1,"#b26b19")):
            med,lo,hi = bands[i,odor,:,0,:].T
            ax.fill_between(x,lo,hi,color=color,alpha=.15)
            ax.plot(x,med,color=color,label="CS+" if odor==0 else "CS-")
            ax.errorbar(x,observed[i,odor,:,0],yerr=sem[i,odor,:,0],fmt="o",
                        color=color,ms=3,capsize=2)
        ax.set_title(names[i]); ax.axhline(0,color=".75",lw=.6)
        ax.set_xticks(x,labels); ax.set_ylabel("Response change (Hz)")
        ax.spines[["top","right"]].set_visible(False)
    axes[0,0].legend(frameon=False)
    fig.suptitle("Published three-module model: replay of Fig. 5c (ACV / EtA)\n"
                 "Lines/bands: supplied parameter ensemble; points: source means +/- SEM",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    for extension in ("png","svg"):
        fig.savefig(output/f"figure5c_replay.{extension}",dpi=160)
    plt.close(fig)
    # Original Fig. 5e uses CS+ minus CS- final responses, not financial utility.
    values = grid[5,0,1]-grid[5,1,1]  # rest, training count, feedback arm
    limit = max(1.,float(np.abs(values).max()))
    fig,axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for arm,ax in enumerate(axes):
        im=ax.imshow(values[:,:,arm].T,origin="lower",aspect="auto",cmap="RdBu_r",
                     vmin=-limit,vmax=limit,extent=(.25,3,1.5,16.5))
        ax.set_title("Intact feedback" if arm==0 else "Feedback removed")
        ax.set_xlabel("Post-training time (h)"); ax.set_ylabel("Training bouts")
        ax.set_yticks([3,6,9,12,15])
    fig.colorbar(im,ax=axes,label="MBON alpha3 CS+ minus CS- (Hz)")
    fig.suptitle("Published Fig. 5e point-estimate grid; no parameter refitting")
    for extension in ("png","svg"):
        fig.savefig(output/f"figure5e_feedback.{extension}",dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path,required=True)
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    checks,arrays,fit,mechanisms = [],{},[],{}
    observed,sem = original_data(args.upstream)
    for modules in (2,3):
        source = args.upstream/f"data_and_parameters/Dx_steady_state_nonlinear_3_27-Mar-2023_{modules}modules.mat"
        d = loadmat(source)
        native = loadmat(args.native/f"native_{modules}modules.mat")
        vectors = np.concatenate([d["para_mu"],d["para_rand"]],axis=1).T
        params = parameter_matrices(vectors,d["mat_lu_cell"])
        pairs,central_traces = [],[]
        for indices in ([0,1,2,3,4,5],[6,7,8,3,4,5]):
            p = list(params); p[0] = params[0][:,:,indices]
            responses,_ = simulate(p,figure5c_protocol())
            pairs.append(responses)
            _,trace = simulate([a[:1] for a in p],figure5c_protocol(),trace=True)
            central_traces.append(trace)
        all_responses = np.stack(pairs,axis=-1) # sample, neuron, odor, time, pair
        central = all_responses[0]
        ensemble = all_responses[1:].transpose(1,2,3,4,0)
        # MATLAB prctile uses midpoint plotting positions (Hyndman-Fan type 5).
        probs = native["probabilities"].ravel()
        bands = np.quantile(ensemble,probs,axis=-1,method="hazen").transpose(1,2,3,4,0)
        compare(central,native["central"],f"{modules}mod central responses",checks)
        compare(ensemble,native["responses"],f"{modules}mod all supplied parameter samples",checks)
        compare(bands,native["bands"],f"{modules}mod median and interval",checks)
        for pair,trace in enumerate(central_traces):
            nt = native["central_trace"][pair,0][0,0]
            for key,actual in trace.items():
                expected = nt[key]
                if key != "W_KDKM_KC": expected = expected[:,0,:]
                compare(actual[0],expected,f"{modules}mod pair{pair} trace {key}",checks)
        mask = np.isfinite(observed)&np.isfinite(sem)&(sem>0)
        if modules==2: mask[[1,4]] = False
        error = (central-observed)[mask]
        fit.append({"modules":modules,"observed_scalars":int(mask.sum()),
                    "rmse_hz":float(np.sqrt(np.mean(error**2))),
                    "sem_weighted_rmse":float(np.sqrt(np.mean((error/sem[mask])**2))),
                    "interpretation":"replay on original fitting data, not held-out prediction"})
        arrays[modules] = {"central":central,"bands":bands}
        np.savez_compressed(args.output/f"replay_{modules}modules.npz",central=central,
                            bands=bands,observed=observed,sem=sem)
        if modules==3:
            weights = central_traces[0]["W_KDKM_KC"][0]
            changes = np.diff(weights[:,3:,:],axis=-1)
            odor_on = np.array([any(b.odor) for b in figure5c_protocol()[1:]])
            active_changes = changes[:,:,odor_on]
            mechanisms["net_effective_weight_changes_during_odor_bouts"] = {
                "positive":int((active_changes>1e-10).sum()),
                "negative":int((active_changes<-1e-10).sum()),
                "note":"net updates include plasticity and retention; not an isolated plasticity test"}
            mechanisms["cs_plus_minus_mbon_traces_hz"] = (central[3:,0,:,:]-central[3:,1,:,:]).tolist()
        print(f"Replayed {modules}-module model, {len(vectors)-1} supplied samples",flush=True)
    d = loadmat(args.upstream/"data_and_parameters/Dx_steady_state_nonlinear_3_27-Mar-2023_3modules.mat")
    grid = np.zeros((6,2,2,12,5,2))
    for arm in range(2):
        v = d["para_mu"].ravel().copy()
        if arm: v[np.array([12,13,15,18,19])-1]=0
        params = reset_valence(parameter_matrices(v,d["mat_lu_cell"]))
        for j,n in enumerate(range(3,16,3)):
            for k,rest in enumerate(range(765,10666,900)):
                y,_ = simulate(params,figure5e_protocol(n,rest))
                grid[:,:,:,k,j,arm] = y[0]
    native_grid = loadmat(args.native/"native_feedback.mat")["feedback_grid"]
    compare(grid,native_grid,"Fig5e feedback grid",checks)
    np.savez_compressed(args.output/"feedback_grid.npz",responses=grid)
    mechanisms["feedback_grid_max_abs_arm_difference_hz"] = float(np.max(np.abs(grid[...,0]-grid[...,1])))
    completion_file = args.native/"native_complete_matlab.mat"
    if not completion_file.exists():
        completion_file = args.native/"native_complete.mat"
    completed = loadmat(completion_file,simplify_cells=True)
    results = {"upstream_commit":UPSTREAM_COMMIT,"protocol":"Fig5c/ED10j supplied 10000 samples + Fig5e point estimates",
        "native_runtime":str(completed["runtime"]),"python_runtime":platform.python_version(),
        "native_elapsed_seconds":float(completed["elapsed_seconds"]),
        "port_elapsed_seconds":time.monotonic()-start,
        "tolerance":{"atol":1e-8,"rtol":1e-9},"checks":checks,
        "all_checks_passed":all(c["passed"] for c in checks),
        "fit_on_published_training_data":fit,"mechanism_descriptives":mechanisms,
        "limitations":["Octave executes original MATLAB functions; MATLAB executable failed to start locally",
          "supplied fitted parameters replayed, not independently refitted",
          "not all paper figures reproduced","no financial adaptation or profitability result",
          "native retention switch depends on complete experimental schedule; not a causal online API"],
        "native_file_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.native.glob('*.mat')}}
    (args.output/"results.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    plot_results(args.output,arrays,observed,sem,grid)
    print(json.dumps({"all_checks_passed":results["all_checks_passed"],
                      "comparisons":len(checks),"output":str(args.output)}),flush=True)
    if not results["all_checks_passed"]: raise SystemExit(1)


if __name__=="__main__":main()
