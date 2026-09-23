"""Kev's local-only lifecycle and declared base identity boundary."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fin_skills.model_zoo.kev import KevModel


def local_files(tmp_path):
    adapter, base = tmp_path / "adapter", tmp_path / "base"
    for root, names in ((adapter, ("head.pt", "adapter_config.json", "adapter_model.safetensors")),
                        (base, ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"))):
        root.mkdir()
        for name in names:
            (root / name).write_text("{}")
    return dict(checkpoint=adapter, revision="a" * 40, base_checkpoint=base, base_revision="b" * 40)


def test_lazy_factory_and_no_stateless_file_loading(tmp_path, monkeypatch):
    from fin_skills.model_zoo import catalog, create_model
    monkeypatch.setattr(catalog, "_available", lambda name: True)
    model = create_model("kev", **local_files(tmp_path))
    assert model._model is None
    card = next(c for c in catalog.model_catalog() if c["id"] == "kev")
    assert card["deployment"] == "local" and not card["json_run"]
    from fin_skills.tools.models import run_model
    with pytest.raises(ValueError, match="Python lifecycle"):
        run_model("kev", {})


def test_missing_local_base_refused_before_loading(tmp_path):
    options = local_files(tmp_path)
    (options["base_checkpoint"] / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="no local"):
        KevModel(**options)


def test_mismatched_base_revision_does_not_load_model(tmp_path, monkeypatch):
    model = KevModel(**local_files(tmp_path))
    module = ModuleType("kev.checkpoint")
    module.Checkpoint = lambda path: SimpleNamespace(meta=SimpleNamespace(base_revision="c" * 40))
    module.LoadOptions = lambda **kw: pytest.fail("must reject before model loading")
    monkeypatch.setitem(sys.modules, "kev", ModuleType("kev"))
    monkeypatch.setitem(sys.modules, "kev.checkpoint", module)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    with pytest.raises(ValueError, match="base revision differs"):
        model._load()
