"""Adversarial protocol tests, not a model capability benchmark."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from fin_skills.api import sealed_eval as se

MODEL = se.MODELS[0]
ROOT = Path(__file__).resolve().parents[1]


def response(cid, pick="alpha", **message):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps({"id": cid, "pick": pick}), **message}}]}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "TEST_API_KEY_NEVER_EXPORT")
    pub, key, out = tmp_path / "public", tmp_path / "grader" / "key.json", tmp_path / "answers"
    cases = [{"q": "choose alpha", "expect": "alpha", "note": "SECRET_AUTHOR_NOTE"},
             {"q": "choose beta", "expect": "beta"}]
    se.prepare(cases, listing="alpha or beta", allowed=["alpha", "beta", "none"],
               public_dir=pub, private_key=key, repository=tmp_path / "repo",
               exposure="public-fixture")
    return pub, key, out


def fake_model(body, _):
    question = json.loads(body["messages"][1]["content"])["question"]
    return response(question["id"], "beta" if "beta" in question["q"] else "alpha")


def finish(bundle, monkeypatch, post=fake_model):
    pub, _, out = bundle
    monkeypatch.setattr(se, "_post", post)
    return se.run(pub, out, model=MODEL)


def grade(bundle, receipt):
    pub, key, out = bundle
    return se.score(pub, key, out, receipt_sha256=receipt)


def change(path, transform):
    obj = se.read_json(path)
    transform(obj)
    path.write_bytes(se.canonical(obj))


def test_model_input_excludes_private_key_notes_history_and_other_questions(bundle, monkeypatch):
    pub, key, out = bundle
    secret = se.read_json(key)["salt"]
    captured = []
    real_read = se.read_json

    def public_only(path):
        assert Path(path).resolve() != key.resolve(), "runner tried to read answer key"
        return real_read(path)

    def model(body, token):
        assert token == "TEST_API_KEY_NEVER_EXPORT"
        captured.append(copy.deepcopy(body))
        return fake_model(body, token)

    monkeypatch.setattr(se, "read_json", public_only)
    receipt = finish(bundle, monkeypatch, model)
    monkeypatch.setattr(se, "read_json", real_read)
    serialized = se.canonical(captured).decode()
    for forbidden in (secret, "SECRET_AUTHOR_NOTE", "TEST_API_KEY_NEVER_EXPORT", '"expect"',
                      str(key), '"tools"', '"functions"', '"previous_response_id"'):
        assert forbidden not in serialized
    assert len(captured) == 2
    assert "choose beta" not in captured[0]["messages"][1]["content"]
    assert "choose alpha" not in captured[1]["messages"][1]["content"]
    assert all([m["role"] for m in r["messages"]] == ["system", "user"] for r in captured)
    result = grade(bundle, receipt)
    assert result["hits"] == result["total"] == 2
    assert not result["training_contamination_excluded"]
    assert result["exposure"] == "public-fixture"
    assert "labels" not in result


@pytest.mark.parametrize("where", ["repo", "public", "parent"])
def test_private_key_cannot_be_exported_with_public_files(tmp_path, where):
    pub, repo = tmp_path / "public", tmp_path / "repo"
    key = {"repo": repo / "key.json", "public": pub / "key.json",
           "parent": tmp_path / "key.json"}[where]
    with pytest.raises(se.IntegrityError, match="outside"):
        se.prepare([{"q": "q", "expect": "alpha"}], listing="listing",
                   allowed=["alpha", "none"], public_dir=pub, private_key=key,
                   repository=repo, exposure="public-fixture")
    assert not pub.exists()


def test_key_commitment_is_salted_and_labels_not_exported(bundle):
    pub, key, _ = bundle
    private = se.read_json(key)
    manifest = se.read_json(pub / "manifest.json")
    assert len(private["salt"]) == 64
    assert manifest["key_commitment"] == se.digest(private)
    assert manifest["key_commitment"] != se.digest(private["labels"])
    assert private["salt"] not in (pub / "manifest.json").read_text()
    assert {p.name for p in pub.iterdir()} == {"packet.json", "manifest.json"}


@pytest.mark.parametrize("git_marker", ["directory", "worktree-file"])
def test_private_key_rejected_in_another_git_checkout(tmp_path, git_marker):
    other = tmp_path / "other-repo"
    other.mkdir()
    if git_marker == "directory":
        (other / ".git").mkdir()
    else:
        (other / ".git").write_text("gitdir: /elsewhere/worktrees/other")
    with pytest.raises(se.IntegrityError, match="outside"):
        se.prepare([{"q": "q", "expect": "none"}], listing="", allowed=["none"],
                   public_dir=tmp_path / "public", private_key=other / "hidden" / "key.json",
                   repository=tmp_path / "current-repo", exposure="public-fixture")


@pytest.mark.parametrize("target", ["packet", "manifest", "answers", "response", "request",
                                   "receipt", "key", "attempt"])
def test_post_submission_tampering_is_rejected(bundle, monkeypatch, target):
    pub, key, out = bundle
    receipt = finish(bundle, monkeypatch)
    paths = {"packet": pub / "packet.json", "manifest": pub / "manifest.json",
             "answers": out / "answers.json", "response": out / "response-0.json",
             "request": out / "request-0.json", "receipt": out / "receipt.json",
             "key": key, "attempt": pub / "attempt.json"}
    def mutate(obj):
        if isinstance(obj, list):
            obj.pop()
        else:
            obj["tampered"] = True
    change(paths[target], mutate)
    with pytest.raises(se.IntegrityError):
        grade(bundle, receipt)
    assert not (out / "score.json").exists()


def test_rewriting_receipt_cannot_bypass_independently_kept_hash(bundle, monkeypatch):
    _, _, out = bundle
    receipt = finish(bundle, monkeypatch)
    change(out / "answers.json", lambda a: a.pop())
    change(out / "receipt.json", lambda r: r["artifacts"].update(
        {"answers.json": se.digest(se.read_json(out / "answers.json"))}))
    with pytest.raises(se.IntegrityError, match="receipt changed"):
        grade(bundle, receipt)


@pytest.mark.parametrize("mode", ["invalid-json", "wrong-id", "unknown-pick", "empty",
                                  "extra-field", "duplicate-key", "refusal", "truncated"])
def test_bad_answers_remain_in_denominator(bundle, monkeypatch, mode):
    def model(body, token):
        r = fake_model(body, token)
        q = json.loads(body["messages"][1]["content"])["question"]
        if "beta" in q["q"]:
            choice, msg = r["choices"][0], r["choices"][0]["message"]
            if mode == "invalid-json":
                msg["content"] = "not JSON"
            elif mode == "wrong-id":
                msg["content"] = json.dumps({"id": "wrong", "pick": "beta"})
            elif mode == "unknown-pick":
                msg["content"] = json.dumps({"id": q["id"], "pick": "unlisted"})
            elif mode == "empty":
                msg["content"] = None
            elif mode == "extra-field":
                msg["content"] = json.dumps({"id": q["id"], "pick": "beta", "tool": "read"})
            elif mode == "duplicate-key":
                msg["content"] = '{"id":"x","id":"y","pick":"beta"}'
            elif mode == "refusal":
                msg["refusal"] = "cannot answer"
            else:
                choice["finish_reason"] = "length"
        return r
    result = grade(bundle, finish(bundle, monkeypatch, model))
    assert (result["hits"], result["total"], result["invalid"], result["accuracy"]) == (1, 2, 1, .5)


@pytest.mark.parametrize("attack", ["tool_calls", "function_call"])
def test_tool_attempt_fails_closed_and_never_scores(bundle, monkeypatch, attack):
    pub, _, out = bundle
    def malicious(body, token):
        r = fake_model(body, token)
        r["choices"][0]["message"][attack] = [{"name": "read_file", "path": "../key.json"}]
        return r
    with pytest.raises(se.IntegrityError, match="tool request rejected"):
        finish(bundle, monkeypatch, malicious)
    assert (out / "failed.json").exists()
    assert not (out / "receipt.json").exists()
    assert len(list(out.glob("request-*.json"))) == 1
    with pytest.raises(FileExistsError):
        se.run(pub, out.parent / "retry", model=MODEL)


def test_failure_cannot_be_retried_under_a_new_output_path(bundle, monkeypatch):
    pub, _, out = bundle
    calls = []
    def fail(body, token):
        calls.append(body)
        raise se.IntegrityError("transport unavailable")
    with pytest.raises(se.IntegrityError):
        finish(bundle, monkeypatch, fail)
    with pytest.raises(FileExistsError):
        se.run(pub, out.parent / "second-attempt", model=MODEL)
    assert len(calls) == 1


def test_success_cannot_be_rerun_or_rescored(bundle, monkeypatch):
    pub, _, out = bundle
    receipt = finish(bundle, monkeypatch)
    grade(bundle, receipt)
    with pytest.raises(FileExistsError):
        grade(bundle, receipt)
    with pytest.raises(FileExistsError):
        se.run(pub, out.parent / "best-of-two", model=MODEL)


def test_wrong_key_from_another_run_is_rejected(bundle, monkeypatch, tmp_path):
    receipt = finish(bundle, monkeypatch)
    wrong = tmp_path / "another-key.json"
    wrong.write_text("{}")
    with pytest.raises(se.IntegrityError):
        se.score(bundle[0], wrong, bundle[2], receipt_sha256=receipt)


@pytest.mark.parametrize("model", ["gpt-4o-search-preview", "o3-deep-research", "unknown", ""])
def test_implicit_tool_models_and_unreviewed_models_rejected(bundle, monkeypatch, model):
    pub, _, out = bundle
    with pytest.raises(se.IntegrityError, match="reviewed"):
        se.run(pub, out, model=model)
    assert not (pub / "attempt.json").exists()


def test_missing_credentials_fails_before_consuming_attempt(bundle, monkeypatch):
    pub, _, out = bundle
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(se.IntegrityError, match="OPENAI_API_KEY"):
        se.run(pub, out, model=MODEL)
    assert not (pub / "attempt.json").exists()


def test_redirects_refused():
    with pytest.raises(se.IntegrityError, match="redirects"):
        se._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid")


def test_transport_uses_fixed_endpoint_no_proxy_and_redacts_errors(monkeypatch):
    def opener(*handlers):
        assert any(getattr(h, "proxies", None) == {} for h in handlers)
        assert any(isinstance(h, se._NoRedirect) for h in handlers)
        class Client:
            def open(self, req, timeout):
                assert req.full_url == se.ENDPOINT
                assert timeout == 60
                assert req.get_header("Authorization") == "Bearer SECRET"
                raise HTTPError(req.full_url, 401, "SECRET", {}, None)
        return Client()
    monkeypatch.setattr(se, "build_opener", opener)
    with pytest.raises(se.IntegrityError, match="HTTP 401") as error:
        se._post({}, "SECRET")
    assert "SECRET" not in str(error.value)


def pit_source():
    return {
        "feature_names": ["price"], "selection_end": "2026-01-01T00:00:00Z",
        "selection_labels_end": "2026-01-02T00:00:00Z",
        "holdout_start": "2026-01-03T00:00:00Z",
        "decisions": [{"at": "2026-01-03T12:00:00Z", "label_end_at": "2026-01-04T12:00:00Z",
                       "q": "Predict up or down", "expect": "up"},
                      {"at": "2026-01-04T12:00:00Z", "label_end_at": "2026-01-05T12:00:00Z",
                       "q": "Predict up or down", "expect": "down"}],
        "observations": [
            {"observed_at": "2026-01-03T10:00:00Z", "available_at": "2026-01-03T10:01:00Z",
             "features": {"price": 10, "future_return": 999}},
            {"observed_at": "2026-01-03T11:00:00Z", "available_at": "2026-01-04T00:00:00Z",
             "features": {"price": 999, "future_return": 999}},
            {"observed_at": "2026-01-04T10:00:00Z", "available_at": "2026-01-04T10:01:00Z",
             "features": {"price": 111, "future_return": 999}},
        ]}


def test_pit_excludes_future_and_late_publication_even_for_historical_observations():
    cases = se.point_in_time_cases(**pit_source())
    first = cases[0]["context"]["observations"]
    assert len(first) == 1 and first[0]["features"] == {"price": 10}
    assert "future_return" not in se.canonical(cases).decode()
    assert len(cases[1]["context"]["observations"]) == 3
    mutated = pit_source()
    mutated["observations"][1]["features"]["price"] = -777
    mutated["observations"][2]["features"]["price"] = -888
    assert se.point_in_time_cases(**mutated)[0] == cases[0]


def test_later_pit_cases_never_enter_earlier_requests(tmp_path, monkeypatch):
    pub, key, out = tmp_path / "pub", tmp_path / "private" / "key.json", tmp_path / "out"
    se.prepare(se.point_in_time_cases(**pit_source()), listing="", allowed=["up", "down", "none"],
               public_dir=pub, private_key=key, repository=tmp_path / "repo",
               exposure="public-fixture")
    bodies = []
    def model(body, token):
        bodies.append(body)
        q = json.loads(body["messages"][1]["content"])["question"]
        return response(q["id"], "none")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    finish((pub, key, out), monkeypatch, model)
    assert "999" not in se.canonical(bodies[0]).decode()
    assert "111" not in se.canonical(bodies[0]).decode()
    assert "999" in se.canonical(bodies[1]).decode()


@pytest.mark.parametrize("change_kind", ["overlap", "unsettled-label", "naive-time", "early-decision",
                                       "early-outcome", "target-feature", "bad-availability",
                                       "nan-feature", "duplicate-decision"])
def test_pit_rejects_invalid_boundaries(change_kind):
    source = pit_source()
    if change_kind == "overlap":
        source["selection_end"] = source["holdout_start"]
    elif change_kind == "unsettled-label":
        source["selection_labels_end"] = source["holdout_start"]
    elif change_kind == "naive-time":
        source["holdout_start"] = "2026-01-03"
    elif change_kind == "early-decision":
        source["decisions"][0]["at"] = source["selection_end"]
    elif change_kind == "early-outcome":
        source["decisions"][0]["label_end_at"] = source["decisions"][0]["at"]
    elif change_kind == "target-feature":
        source["feature_names"] = ["future_return"]
    elif change_kind == "bad-availability":
        source["observations"][0]["available_at"] = "2026-01-01T00:00:00Z"
    elif change_kind == "nan-feature":
        source["observations"][0]["features"]["price"] = float("nan")
    else:
        source["decisions"].append(source["decisions"][0])
    with pytest.raises(se.IntegrityError):
        se.point_in_time_cases(**source)


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_refuses_private_holdout_in_repository(tmp_path, capsys):
    cli = load_cli("eval_sealed")
    code = cli.main(["prepare", "--source", str(ROOT / "evals" / "queries.jsonl"),
                     "--public-dir", str(tmp_path / "pub"), "--private-key",
                     str(tmp_path / "private" / "key.json"), "--exposure", "private-holdout"])
    assert code == 1
    assert "outside the repository" in capsys.readouterr().err


def test_cli_prepare_strips_notes_and_context(tmp_path):
    cli = load_cli("eval_sealed")
    source, pub = tmp_path / "queries.jsonl", tmp_path / "pub"
    source.write_text(json.dumps({"q": "download prices", "expect": "lib-yfinance",
                                  "context": {"answer": "NEVER_SEND"}, "note": "NEVER_SEND"}))
    assert cli.main(["prepare", "--source", str(source), "--public-dir", str(pub),
                     "--private-key", str(tmp_path / "private" / "key.json"),
                     "--exposure", "public-fixture"]) == 0
    assert "NEVER_SEND" not in (pub / "packet.json").read_text(encoding="utf-8")
    assert len(se.read_json(pub / "packet.json")["cases"]) == 1
