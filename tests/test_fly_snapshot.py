"""Archive deployment checks reject changed or newly injected Python sources."""
import hashlib
import io
import tarfile

import pytest

from benchmarks.fly_reuse.verify_snapshot import verify_archive


def test_archive_source_matches_then_detects_mutation_and_extra_code(tmp_path):
    source = tmp_path / "upstream/repo/module.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    archive = tmp_path / "upstream.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        data = source.read_bytes()
        entry = tarfile.TarInfo("upstream/repo/module.py")
        entry.size = len(data)
        stream.addfile(entry, io.BytesIO(data))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert verify_archive(tmp_path, sha)["files_verified"] == 1
    source.write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="differs from archive"):
        verify_archive(tmp_path, sha)
    source.write_bytes(data)
    (source.parent / "extra.py").write_text("VALUE = 3")
    with pytest.raises(ValueError, match="unlisted Python"):
        verify_archive(tmp_path, sha)
