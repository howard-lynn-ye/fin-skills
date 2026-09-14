"""Ensure distribution, runtime and (when supplied) release tag versions agree."""
import argparse
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fin_skills


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["version"] != fin_skills.__version__:
        raise ValueError("runtime and distribution versions differ")
    if args.tag is not None and args.tag != "v" + project["version"]:
        raise ValueError(f"tag must be v{project['version']}; update both versions before tagging")
    print("RELEASE_VERSION_OK", project["version"])


if __name__ == "__main__":
    main()
