"""Local Kev adapter; caller supplies both trusted adapter and base checkpoints."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

from .jev import _json_copy, _questions
from .laya import LayaModel, canonical_response


class KevModel:
    """No hosted calls, automatic downloads, server or date-facts preprocessing.

    Upstream loads head.pt from the trusted checkpoint. Install Kev in its own
    compatible environment. Weight identity is declared by the caller and recorded.
    """
    model_id = "kev"

    def __init__(self, *, checkpoint, revision, base_checkpoint, base_revision,
                 device="cpu", dtype="float32"):
        if not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{40}", v)
                   for v in (revision, base_revision)):
            raise ValueError("checkpoint and base revisions must be full commits")
        if device != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", device):
            raise ValueError("device must be cpu, cuda or cuda:N")
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        self.checkpoint, self.base_checkpoint = Path(checkpoint).resolve(), Path(base_checkpoint).resolve()
        for directory, files in ((self.checkpoint, ("head.pt", "adapter_config.json", "adapter_model.safetensors")),
                                 (self.base_checkpoint, ("config.json", "tokenizer.json", "tokenizer_config.json"))):
            for name in files:
                if not (directory / name).is_file():
                    raise FileNotFoundError(f"incomplete local Kev checkpoint: {name}")
        if not list(self.base_checkpoint.glob("*.safetensors")):
            raise FileNotFoundError("base checkpoint has no local safetensors weights")
        self.revision, self.base_revision = revision, base_revision
        self.device, self.dtype = device, dtype
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            import kev
            from kev.checkpoint import Checkpoint, LoadOptions
            ck = Checkpoint(str(self.checkpoint))
            if ck.meta.base_revision != self.base_revision:
                raise ValueError("base revision differs from the adapter head metadata")
            self._base_name = ck.meta.base
            # Local path substitution only; no changes to saved checkpoint metadata or weights.
            ck.meta.base, ck.meta.base_revision = str(self.base_checkpoint), None
            self._tok, self._model = ck.load(self.device, LoadOptions(
                dtype=getattr(torch, self.dtype), backend="torch", merge=True))
            self._implementation_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(kev.__file__).parent.glob("*.py")}
        return self._tok, self._model

    def predict(self, state, *, questions):
        if not isinstance(state, (str, dict, list)):
            raise TypeError("state must be text, a JSON object or a JSON array")
        checked, state = _questions(questions), _json_copy(state, "state")
        tok, model = self._load()
        from kev.api import SystemOneRequest, to_record, to_answers, output_tokens
        from kev.model import SERVE_MAX_STATE, SERVE_MAX_BRANCH
        import torch
        request = SystemOneRequest(state=state, model="kev-latest", questions=checked)
        record, meta = to_record(request)
        encoded = model.encode(tok, record, max_state=SERVE_MAX_STATE,
                               max_branch=SERVE_MAX_BRANCH, strict=True)
        if encoded.get("state_truncated"):
            raise ValueError("Kev truncated the state")
        with torch.inference_mode():
            probabilities = [p.tolist() for p in model.probs(encoded)]
        raw = dict(model="kev-local", answers=to_answers(probabilities, meta),
                   usage=dict(input_tokens=len(encoded["ids"]), output_tokens=0))
        result = canonical_response(raw, checked)
        result["provider_model"], result["model"] = result["model"], self.model_id
        result["provenance"] = dict(checkpoint=str(self.checkpoint), declared_revision=self.revision,
            base_checkpoint=str(self.base_checkpoint), declared_base_revision=self.base_revision,
            base_model=self._base_name, device=self.device, dtype=self.dtype,
            implementation_sha256=self._implementation_hashes,
            temperature=float(model.head.temperature), date_facts=False, prefix_cache=False,
            input_ids_sha256=hashlib.sha256(str(encoded["ids"]).encode("ascii")).hexdigest(),
            serialized_output_tokens=output_tokens(tok, raw["answers"]),
            output_tokens_interpretation="No autoregressive decoding; serialization tokens reported separately.")
        return result

    def run(self, data):
        if not isinstance(data, dict) or set(data) != {"state", "questions"}:
            raise ValueError("Kev data requires exactly state and questions")
        return self.predict(data["state"], questions=data["questions"])

    def rerank(self, query, passages, *, top_k=None):
        return LayaModel.rerank(self, query, passages, top_k=top_k)
