from benchmarks.agent_study.audit_validity import CASES, policy_rows, source
from benchmarks.agent_study.open_agent_runner import PUBLIC_CHECKS


def test_policy_lattice_retains_false_alarms_and_bad_acceptance():
    rows = [dict(case="clean", label="clean", execution=dict(passed=True),
                 checks={c: dict(passed=c!="assert_causal") for c in PUBLIC_CHECKS}),
            dict(case="bad", label="defect", execution=dict(passed=True),
                 checks={c: dict(passed=c!="accounting") for c in PUBLIC_CHECKS})]
    policies = policy_rows(rows)
    assert len(policies) == 16
    assert policies[0]["defect_accepted"] == 1
    assert policies[-1]["clean_rejected"] == 1
    assert policies[-1]["defect_accepted"] == 0


def test_invalid_execution_is_never_a_success():
    policies = policy_rows([dict(case="missing", label="invalid", execution=dict(status="TIMEOUT"),
        checks={c: dict(passed=True) for c in PUBLIC_CHECKS})])
    assert all(p["defect_accepted"] == 0 for p in policies)


def test_all_declared_candidates_compile():
    for case in CASES:
        compile(source(case), "submission.py", "exec")


def test_report_reference_agrees_with_public_accounting(tmp_path):
    import json
    from benchmarks.agent_study.build_task import export
    from benchmarks.agent_study.audit_validity import reference_report
    from benchmarks.agent_study.submission_audit import accounting
    export(11, tmp_path)
    (tmp_path / "submission.py").write_text(source("clean_news"))
    value = reference_report(tmp_path, 10.0)
    (tmp_path / "report.json").write_text(json.dumps(dict(reported_sharpe=value,
                                                        cost_bps_per_side=10.0)))
    assert accounting(tmp_path)["absolute_gap"] < 1e-10
