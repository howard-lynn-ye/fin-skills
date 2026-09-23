"""Stage one pinned second-family model on Beacon; never infer on a login node."""
import hashlib
import json
import os
from pathlib import Path

MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"


def stage(output, cache):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    output, cache = Path(output), Path(cache)
    if not str(output.resolve()).startswith("/beacon-projects/radfm/wy891/"):
        raise ValueError("RADFM output required")
    if not str(cache.resolve()).startswith("/beacon-projects/radfm/wy891/"):
        raise ValueError("RADFM cache required")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=cache,
        allow_patterns=["config.json", "generation_config.json", "model*.safetensors",
            "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "merges.txt", "vocab.json"], max_workers=4))
    files = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files[path.name] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    messages = [{"role": "system", "content": "Reply using one JSON tool call."},
                {"role": "user", "content": "Read the task."},
                {"role": "assistant", "content": '{"tool":"read_file","arguments":{}}'},
                {"role": "user", "content": '{"text":"example task"}'}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not all(x["content"] in rendered for x in messages):
        raise RuntimeError("official template lost an input message")
    receipt = dict(model=MODEL, revision=REVISION, snapshot=str(snapshot), files=files,
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        conversation_template_passed=True, inference_performed=False)
    with (output / "receipt.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(dict(model=MODEL, revision=REVISION, files=len(files), staged=True)))
