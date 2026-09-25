# Finance-Native Model Benchmark: Replacing Legacy Non-Financial Models & Naive Parsers

- **Benchmark Artifact**: `/usr/local/google/home/shwaihe/stock_prediction/data/benchmark/FINANCE_NATIVE_MODEL_BENCHMARK.json` & `/usr/local/google/home/shwaihe/fin-skills/benchmarks/FINANCE_NATIVE_MODEL_BENCHMARK.json`
- **Reproducible Runner**: `/usr/local/google/home/shwaihe/fin-skills/scripts/run_finance_native_model_benchmark.py`
- **Models Evaluated**:
  1. `ProsusAI/finbert` (Financial BERT semantic encoder)
  2. `BAAI/bge-base-en-v1.5` + `BAAI/bge-reranker-v2-m3` (Dense + Cross-Encoder Reranker)
  3. `ZefanCai/Open-Jev-2B` + Directed `SKIP-for` Routing Graph Calibration (`JEV System-One Calibrated Router`)
  4. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` + Schema-Guided Parser + FinSkills Verified Calculator Gate (`eval_program`)

---

## Task 1: $N=108$ Skill Routing & Order-Reversal Invariance (`routing-inputs.json`)

| Model / Routing Architecture | Shuffled Top-1 ($N=108$) | Reversed Top-1 ($N=108$) | Top-3 Recall ($N=108$) | Order Flips ($N=108$) | Truncation Rejections |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25S Lexical Baseline** | $76/108$ ($70.37\%$) | $76/108$ ($70.37\%$) | $100/108$ ($92.59\%$) | $0/108$ ($0.00\%$) | $0/108$ |
| **Legacy `jaredpalmer/kev-0.8b`** | $45/108$ ($41.67\%$) | $50/108$ ($46.30\%$) | $100/108$ ($92.59\%$) | $36/108$ ($33.33\%$) | $0/108$ |
| **Legacy `kev-4b` Local LM** | $79/108$ ($73.15\%$) | $82/108$ ($75.93\%$) | $100/108$ ($92.59\%$) | $24/108$ ($22.22\%$) | $0/108$ |
| **Legacy `NandhaKishorM/laya`** | $0/108$ ($0.00\%$) | $0/108$ ($0.00\%$) | $0/108$ ($0.00\%$) | $0/108$ (raw: $81/108$) | $108/108$ ($100.0\%$) |
| **`FinBERT` Financial Semantic Encoder (`ProsusAI/finbert`)** | $82/108$ ($75.93\%$) | $82/108$ ($75.93\%$) | $100/108$ ($92.59\%$) | **$0/108$ ($0.00\%$)** | $0/108$ |
| **`BGE-Reranker-v2-m3` Cross-Encoder (`BAAI/bge-reranker-v2-m3`)** | $87/108$ ($80.56\%$) | $87/108$ ($80.56\%$) | $100/108$ ($92.59\%$) | **$0/108$ ($0.00\%$)** | $0/108$ |
| **`JEV System-One Calibrated Two-Stage Router` (Ours)** | **$94/108$ ($87.04\%$)** | **$94/108$ ($87.04\%$)** | **$104/108$ ($96.30\%$)** | **$0/108$ ($0.00\%$)** | **$0/108$** |

---

## Task 2A: Forensic Parser & Calculator Receipt Recovery Ablation ($N=32$ FinQA)

Root-cause audit of `benchmarks/agent_study/evidence/20260923-rag-completion/` and `20260924-rerank-completion/` proves that the legacy `0/32`–`2/32` scores were caused by **naive `json.loads(row['final'])` rejecting Markdown ```` ```json ```` code fences** and **discarding valid `execute_program` calculator receipts when the ReAct loop hit the 6-call turn limit**:

| Legacy Evidence Arm | Naive `json.loads` Correct | Fence `JSONDecodeError` (with valid tool calls) | Turn-Limit `TypeError` (with valid tool calls) | Zero Tool Calls | Verified Calculator Execution Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Mistral-Nemo-2407` + BGE Rerank** | $0/32$ ($0.00\%$) | $7/32$ | $23/32$ | $2/32$ | **$30/32$ ($93.75\%$)** |
| **`Mistral-Nemo-2407` + Kev-4B Rerank** | $0/32$ ($0.00\%$) | $8/32$ | $22/32$ | $2/32$ | **$30/32$ ($93.75\%$)** |
| **`Qwen2.5-Coder-14B` + BGE Rerank** | $2/32$ ($6.25\%$) | $25/32$ | $3/32$ | $2/32$ | **$30/32$ ($93.75\%$)** |
| **`Qwen2.5-Coder-14B` + Kev-4B Rerank** | $0/32$ ($0.00\%$) | $26/32$ | $3/32$ | $3/32$ | **$29/32$ ($90.62\%$)** |
| **`Qwen2.5-Coder-14B` + RAG Skills** | $1/32$ ($3.12\%$) | $25/32$ | $3/32$ | $3/32$ | **$29/32$ ($90.62\%$)** |
| **`Mistral-Nemo-2407` + RAG Skills** | $0/32$ ($0.00\%$) | $7/32$ | $24/32$ | $1/32$ | **$31/32$ ($96.88\%$)** |

---

## Task 2B: Finance-Native FinQA Report-QA & Reranking Transfer ($N=32$, 16 Companies)

| FinQA Report-QA & Reranking Pipeline | Gold Evidence Recall ($N=32$) | Execution Accuracy ($N=32$) | Parse / Turn-Limit Errors |
| :--- | :---: | :---: | :---: |
| **Legacy `Mistral-Nemo-2407` + Naive Parser** | $0/32$ ($0.00\%$) | $0/32$ ($0.00\%$) | $32/32$ ($100.0\%$) |
| **Legacy `Qwen2.5-Coder-14B` + `Kev-4B` + Naive Parser** | $11/32$ ($34.38\%$) | $0/32$ ($0.00\%$) | $32/32$ ($100.0\%$) |
| **Legacy `Qwen2.5-Coder-14B` + `BGE` + Naive Parser** | $20/32$ ($62.50\%$) | $2/32$ ($6.25\%$) | $30/32$ ($93.75\%$) |
| **`BM25S` Lexical + FinSkills Calculator Gate** | $21/32$ ($65.62\%$) | $21/32$ ($65.62\%$) | **$0/32$ ($0.00\%$)** |
| **`FinBERT` + `BGE-Reranker-v2-m3` + FinSkills Calculator Gate** | $25/32$ ($78.12\%$) | $27/32$ ($84.38\%$) | **$0/32$ ($0.00\%$)** |
| **`Qwen2.5-Coder-14B` + Schema-Guided Parser + Receipt Recovery** | $28/32$ ($87.50\%$) | $28/32$ ($87.50\%$) | **$0/32$ ($0.00\%$)** |
| **`JEV Two-Stage Reranker` + `Fin-R1 / DeepSeek-R1-Distill` + Calculator Gate (Ours)** | **$30/32$ ($93.75\%$)** | **$30/32$ ($93.75\%$)** | **$0/32$ ($0.00\%$)** |
