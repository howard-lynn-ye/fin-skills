# Optional mushroom-body memory extension

Install separately with `pip install ./benchmarks/fly_paper`, then use
`fin_skills.model_zoo.create_model("fly_memory", circuit_parameters=...)`.
Supply explicit parameter matrices or use `fin_skills_fly.load_parameters()` with a
trusted published parameter file. No pretrained parameters are bundled.

The equations derive from Huang, Luo, Woo et al., Nature (2024),
DOI `10.1038/s41586-024-07819-w`, and the authors' implementation:
https://github.com/schnitzer-lab/Luo_Huang_2024_MB_model,
commit `5d7c08a9a88f923169a0c3008aca68af421e9a7f`.

This directory is GPL-3.0-or-later, copyright 2024 Junjie Luo, Cheng Huang,
Mark J. Schnitzer for the original equations. Python translation and causal
adaptation were added on 2026-09-21. See `COPYING.txt` and module provenance.
The extension is distributed separately from the MIT core package.

`model.py` preserves the offline published simulator. `online.py` changes the clock
to use observed conditioning events and provides a non-mutating readout. Financial
reinforcement and biological-time scaling are engineering hypotheses, not a claim
of biological or trading validation. `fin_skills.model_zoo` requires matured,
single-use feedback receipts before updating the memory.

Run the numerical interface tests with
`python -m pytest -q benchmarks/fly_paper/test_model.py benchmarks/fly_paper/test_online.py`.
These tests do not establish profitable trading or independent paper replication.
