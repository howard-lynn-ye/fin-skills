"""Shared trusted-artifact persistence and compatibility with existing algorithm models."""
import hashlib
import json
import pickle
from pathlib import Path
import zipfile

from fin_skills.algorithms.models import environment


class ModelArtifact:
    def save(self, path):
        """Save exclusively. Files contain pickle; load only trusted artifacts."""
        payload = pickle.dumps(self, protocol=5)
        versions = environment()
        for library in getattr(self, "dependencies", ()):
            versions.update(environment(library))
        digest = hashlib.sha256()
        for file in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(file.name.encode() + b"\0" + file.read_bytes())
        versions["model_zoo_sha256"] = digest.hexdigest()
        manifest = dict(schema=1, model_id=self.model_id, environment=versions,
                        sha256=hashlib.sha256(payload).hexdigest())
        with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, allow_nan=False))
            archive.writestr("model.pkl", payload)
        return Path(path)


def load_model(path, *, trusted=False, strict_versions=True):
    if trusted is not True:
        raise ValueError("model contains pickle; load only a trusted artifact with trusted=True")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") != 1:
            raise ValueError("unsupported model-zoo artifact schema")
        expected = manifest["environment"]
        current = environment()
        digest = hashlib.sha256()
        for file in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(file.name.encode() + b"\0" + file.read_bytes())
        current["model_zoo_sha256"] = digest.hexdigest()
        for name in expected:
            if name not in current:
                current.update(environment(name))
        if strict_versions and any(current.get(k) != v for k, v in expected.items()):
            raise ValueError("model environment or implementation differs")
        payload = archive.read("model.pkl")
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("model checksum mismatch")
    model = pickle.loads(payload)
    if not isinstance(model, ModelArtifact) or model.model_id != manifest["model_id"]:
        raise ValueError("unexpected model artifact content")
    return model


class AlgorithmModel(ModelArtifact):
    def __init__(self, model_id, parameters):
        from fin_skills.algorithms import _DEFAULT
        spec = _DEFAULT.get(model_id)
        self.model_id, self.parameters = model_id, dict(parameters)
        self.dependencies = (spec.library,)
        self.fitted = None

    def run(self, data):
        from fin_skills.algorithms import run
        return run(self.model_id, data, **self.parameters)

    def fit(self, data):
        from fin_skills.algorithms import fit
        self.fitted = fit(self.model_id, data, **self.parameters)
        return self

    def predict(self, X=None, *, horizon=1):
        if self.fitted is None:
            raise ValueError("fit the model before prediction")
        return self.fitted.predict(X, horizon=horizon)
