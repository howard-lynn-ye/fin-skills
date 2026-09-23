# DocFinQA: pinned corpus and capacity preparation

This preparation reuses the author's public dataset. It produces no agent score and runs
no reference Python programs. Source: [Kensho DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA),
associated with [DocFinQA, ACL 2024](https://aclanthology.org/2024.acl-short.42/).
The pinned dataset revision is `64ebaff62f692495bcc182f45cf9a9606251b19b`.
The Hugging Face metadata declares MIT; this is recorded as upstream metadata.

The three original files are downloaded on Beacon and checked against the author's LFS
SHA-256 and byte size. All four source fields are retained in the archive. Public inputs
contain only the question and a reference to the exact UTF-8 report; `Program` and `Answer`
are written to separate label files. These programs are not assumed to be FinQA DSL and
are never executed by this preparation. Labels are serialized for preservation, not used
to select tasks, resolve question matches, or measure capacity.

Question matching against the previously pinned, label-free FinQA inputs uses NFKC,
whitespace collapse and case folding. Every candidate is retained, including ambiguous
matches and cross-split matches. Exact context hashes measure duplicate text, not verified
company/report identity. Reusing a FinQA question with its full report does not create a
new independent task. Report/company isolation will require an unambiguous source mapping
before any claim of independent evaluation.

Capacity probes include all candidates matching the existing fixed 48 FinQA acquisition
and evaluation IDs, plus eight other development rows selected by a fixed SHA order.
The selection is written before tokenization and does not filter on length or labels.
The two existing pinned model tokenizers apply their official chat templates to a minimal
single-user report/question prompt. No weights are loaded. A 3,072-token response reserve
and a 32,768-token ceiling are reported with each input length; no truncation is allowed.
This is a lower bound for a future agent prompt: ReAct instructions, tool schemas and
memory will add tokens. A probe that fits is not yet proof that the production prompt fits.

The next inference protocol must separately identify complete-report and common retrieved
candidate conditions. It must preserve capacity failures in the denominator, qualify any
HiSTrim integration, and freeze the scorer and actual prompt before inference. This
preparation does not establish answer correctness, memory benefit or compression benefit.
Public historical data also does not establish absence from model pretraining.
