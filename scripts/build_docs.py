#!/usr/bin/env python3
"""Build the API reference for the fin_skills package and, optionally, publish it.

    python scripts/build_docs.py             # pdoc -> site/  (gitignored)
    python scripts/build_docs.py --publish   # also push site/ to the gh-pages branch and
                                             # make sure GitHub Pages serves it

Publishing goes through a plain branch push, so it needs only the `repo` scope - no
workflow file, no Actions. The site is regenerated from whatever is installed as
`fin_skills`, so run `pip install -e .` (or `scripts/build_package.py`) first.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO = "howard-lynn-ye/fin-skills"


def run(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), cwd=cwd or ROOT, text=True, capture_output=True, check=check)


def build() -> None:
    try:
        import pdoc  # noqa: F401
    except ImportError:
        sys.exit("pdoc is not installed: pip install pdoc")
    if SITE.exists():
        shutil.rmtree(SITE)
    # One page per module, plus the package index. Google-style docstrings; the generated
    # modules carry their own docstrings, and the api/ layer documents the owning skill.
    # The top-level __init__ restricts __all__ to the text API, which makes pdoc skip the
    # namespace subpackages; name them explicitly so every generated module gets a page.
    pkg = ROOT / "fin_skills"
    modules = ["fin_skills"]
    for p in sorted(pkg.rglob("*.py")):
        rel = p.relative_to(pkg)
        if "__pycache__" in rel.parts or any(part.startswith("_") and part != "__init__.py"
                                             for part in rel.parts):
            continue                       # _skills/ data, _common.py helpers
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.append("fin_skills." + ".".join(parts))
    modules = sorted(set(modules))
    r = run(sys.executable, "-m", "pdoc", *modules, "-o", str(SITE), "--docformat", "google",
            "--no-show-source", check=False)
    if r.returncode != 0:
        sys.exit("pdoc failed:\n" + r.stderr[-2000:])
    # A landing note: what the two halves of the package are, so a reader who arrives at the
    # API reference knows the skills text is reachable from the same install.
    idx = ROOT / "catalog" / "index.json"
    n_skills = len(json.loads(idx.read_text(encoding="utf-8")).get("skills", [])) if idx.exists() else "?"
    (SITE / "README.md").write_text(
        f"fin-skills API reference. {n_skills} skills as package data (fin_skills.load(name)); the\n"
        "executable guards and conventions are the modules listed in index.html. Source of truth:\n"
        f"https://github.com/{REPO}\n", encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")
    pages = sorted(p.relative_to(SITE).as_posix() for p in SITE.rglob("*.html"))
    print(f"site/ built: {len(pages)} pages")


def publish() -> None:
    if not (SITE / "index.html").exists():
        sys.exit("site/ is empty - run without --publish first")
    with tempfile.TemporaryDirectory() as td:
        wt = Path(td) / "gh-pages"
        # Does the branch exist on the remote?
        exists = run("git", "ls-remote", "--exit-code", "--heads", "origin", "gh-pages", check=False).returncode == 0
        if exists:
            run("git", "worktree", "add", "--detach", str(wt), "origin/gh-pages")
            run("git", "checkout", "-B", "gh-pages", "origin/gh-pages", cwd=wt)
        else:
            run("git", "worktree", "add", "--detach", str(wt))
            run("git", "checkout", "--orphan", "gh-pages", cwd=wt)
            run("git", "rm", "-rfq", ".", cwd=wt, check=False)
        for p in list(wt.iterdir()):
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        for p in SITE.iterdir():
            (shutil.copytree if p.is_dir() else shutil.copy2)(p, wt / p.name)
        run("git", "add", "-A", cwd=wt)
        head = run("git", "rev-parse", "--short", "HEAD").stdout.strip()
        r = run("git", "commit", "-q", "-m", f"Docs: API reference built from {head}", cwd=wt, check=False)
        if r.returncode != 0 and "nothing to commit" in (r.stdout + r.stderr):
            print("gh-pages: nothing changed")
        else:
            run("git", "push", "-q", "origin", "gh-pages", cwd=wt)
            print("gh-pages: pushed")
        run("git", "worktree", "remove", "--force", str(wt), check=False)
    # Make sure Pages serves the branch. Needs only repo scope.
    r = run("gh", "api", f"repos/{REPO}/pages", check=False)
    if r.returncode != 0:
        r = run("gh", "api", "--method", "POST", f"repos/{REPO}/pages",
                "-f", "source[branch]=gh-pages", "-f", "source[path]=/", check=False)
        ok = r.returncode == 0 or "already enabled" in r.stderr   # a gh-pages push auto-enables it
        print("pages: enabled" if ok else "pages: could not enable - " + r.stderr[-300:])
    else:
        url = json.loads(r.stdout).get("html_url", "?")
        print(f"pages: already enabled at {url}")


if __name__ == "__main__":
    build()
    if "--publish" in sys.argv:
        publish()
