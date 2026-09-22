"""Resolve revisions once, then download exact public snapshots inside RADFM."""
import argparse
import json
from pathlib import Path
from huggingface_hub import model_info, snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "models.json"
    if manifest.exists():
        models = json.loads(manifest.read_text())
    else:
        models = [{"model": f"Qwen/Qwen2.5-Coder-{size}B-Instruct",
                   "revision": model_info(f"Qwen/Qwen2.5-Coder-{size}B-Instruct").sha}
                  for size in (7, 14, 32)]
        manifest.write_text(json.dumps(models, indent=2))
    for item in models:
        path = snapshot_download(item["model"], revision=item["revision"],
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*"],
                                 max_workers=4)
        print(json.dumps({**item, "snapshot": path}), flush=True)


if __name__ == "__main__":
    main()
