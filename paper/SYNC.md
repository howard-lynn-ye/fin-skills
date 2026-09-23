# Manuscript delivery: GitHub and Overleaf

Updated 2026-09-22. A manuscript revision is delivered only when its source is committed
to GitHub **and** the corresponding files are verified in the existing Overleaf project.
A local commit, a ZIP export or a successful GitHub push alone does not meet both conditions.

## Source locations

- GitHub repository: `howard-lynn-ye/fin-skills`; current default branch: `master`.
- Manuscript sources: `paper/latex_naacl/`; entry point: `main.tex`.
- Writing outline: `paper/OUTLINE.md`; experiment plan: `paper/EXPERIMENTS_NEXT_ZH.md`.
- Existing Overleaf project: <https://www.overleaf.com/project/6aad9a03f27c3c07a182965d>.

The GitHub repository stores the library, experiments, evidence and article together.
Overleaf receives the article's TeX, bibliography, required style files and referenced
figures. Keep experimental datasets, private labels, credentials, generated package trees
and unrelated research archives out of the manuscript project. Do not replace coauthor
edits with an unreviewed copy of the local manuscript.

## Revision workflow

1. Read the current Overleaf files and compare them with the last synchronized revision.
   Incorporate coauthor edits before changing the same manuscript files locally. Resolve
   disagreements in the diff; do not force-push over them.
2. Make a bounded revision. Keep descriptions of the current library separate from frozen
   experimental snapshots. Regenerate numerical tables from their recorded evidence when
   the underlying evidence changes; do not hand-fill missing experimental results.
3. Run repository checks and compile the manuscript. Review the resulting PDF for missing
   references, clipped tables and layout problems. Commit the source and reviewed PDF to
   a `codex/` branch, then follow the repository's review/merge requirements.
4. Synchronize the manuscript files to the existing Overleaf project. Pull first, inspect
   the exact changed paths and push without force. Recompile in Overleaf and verify the
   actual remote contents, rather than assuming GitHub also updated Overleaf.
5. Record the GitHub commit, the Overleaf revision or history label, the source-file hashes
   and both build outcomes in the delivery note. These systems need not share commit IDs;
   matching manuscript contents are the comparison that matters.

## Connection and authentication

Overleaf offers Git integration and a separate GitHub synchronization feature. A GitHub
push alone is not a completed Overleaf sync. Avoid linking the entire software repository
to a manuscript-only project without reviewing its file layout.

For direct Git access, obtain the Git URL from the existing project's Integrations menu.
The documented URL form for this project is:

```text
https://git@git.overleaf.com/6aad9a03f27c3c07a182965d
```

Authenticate locally with username `git` and an Overleaf Git authentication token stored
in the user's credential manager. Never put the token in this document, a commit, a remote
URL, or a chat message. The account must have project access and the project must support
Git integration. Inspect the remote's advertised branch: current Overleaf documentation
uses `main`, while older clones may use `master`. This is independent of this repository's
default branch name.

After authentication, use a separate local clone of the manuscript project. Preserve its
history and compare its layout and existing files before copying changes into it. Do not
push the root of `fin-skills` directly into the Overleaf remote.

Git synchronization may displace Overleaf comments or tracked changes, so coordinate the
editing window and review unresolved comments before applying manuscript updates.

## Status checked on 2026-09-22

The local software repository had only the GitHub `origin` remote. A noninteractive
`git ls-remote` check of the documented Overleaf project failed because Git could not
obtain authentication. The existing authenticated browser tab was readable: the visible
history showed Shwai He's September 22, 9:35 am uploads, and the selected appendix source
was visible. This is a partial inspection, not a downloaded or merged source snapshot.

Browser write actions were blocked by automatic safety review, which reported that it
could not determine the request's safety status. On resumption, even cancelling the
unsubmitted Add label dialog was blocked. No new label, upload, Overleaf commit or remote
compile was verified. Coauthor files remain untouched by this synchronization attempt.
The project URL is known; completing delivery requires a permitted synchronization path
and a comparison of the complete current sources before applying manuscript changes.

## Merge synchronization completed on 2026-09-22

The later user request authorized merging the local manuscript with Shwai He's current
shared project. A full source ZIP was downloaded and preserved before editing. The merged
working draft was uploaded to the same project through the authenticated browser. Its
root `main.tex` was read back and its normalized SHA-256 matched the local source:
`725e80407cfbbfdb32764e05e386e16d62f4adbb9323ee5e56abaa34d35f4322`.

Overleaf successfully compiled the merged title into 13 pages, with Errors 0 and
Warnings 0; its log retained underfull-box typesetting notices. The verified history
label is `Merged local + Overleaf — 2026-09-22` (upload shown at 9:20 pm). The original
HiSTrim entry point is preserved as `main_histrim_before_merge.tex`, and its sections,
figures and custom bibliography remain in the project. Detailed merge decisions and
local/remote verification are under `paper/merge_20260922/`.

This supersedes the earlier browser-write blocker for Overleaf synchronization. It does
not establish GitHub delivery: no GitHub push was performed during this merge.

## Primary documentation

- [Overleaf Git integration](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration)
- [Overleaf Git authentication](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/git-integration/git-integration-authentication-tokens)
- [GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization)

These links were consulted for the synchronization workflow on 2026-09-22; they are
service documentation, not evidence that this project's connection has been configured.
