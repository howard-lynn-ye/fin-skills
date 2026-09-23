import numpy as np
import pytest

from benchmarks.agent_study.decision_tasks import cases, execute, reference, grade
from benchmarks.agent_study.decision_memory import ExperienceMemory
from benchmarks.agent_study.histrim import DualRouter


def test_reference_calibration_and_method_parameter_attribution():
    for case in cases(13):
        receipt=execute(case,case['method'],case['parameters'])
        assert np.allclose(receipt['values'],reference(case),rtol=1e-10,atol=1e-12)
        action=dict(method=case['method'],parameters=case['parameters'])
        final=dict(values=receipt['values'],receipt_sha256=receipt['sha256'])
        assert grade(case,action,final,receipt)['correct']
        assert not grade(case,dict(method='wrong',parameters={}),final,receipt)['correct']
        assert not grade(case,action,dict(final,receipt_sha256='wrong'),receipt)['correct']


def test_memory_causal_read_no_duplicate_and_no_update_on_read():
    first,next_case=cases()[:2]
    memory=ExperienceMemory('retrieval')
    memory.update(first,{},dict(correct=False),'Use the requested annualization.')
    assert memory.read(first)[0] == []
    items,read=memory.read(next_case)
    assert items and read['experience_ids'] == [first['id']]
    assert memory.read(next_case)[1]['state_sha256'] == read['state_sha256']
    with pytest.raises(ValueError,match='duplicate'):
        memory.update(first,{},dict(correct=True),'')
    none=ExperienceMemory('none')
    none.update(first,{},dict(correct=False),'anything')
    assert none.read(next_case)[0] == []


def test_router_sign_and_unit_directions():
    x=np.array([[1.,0.],[-1.,0.],[1.,.1],[-1.,.1]])
    router=DualRouter.fit(x,[1.,-1.,1.,-1.])
    assert np.isclose(np.linalg.norm(router.gate),1.)
    assert np.isclose(np.linalg.norm(router.utility),1.)
    utility=8*(x/np.linalg.norm(x,axis=1,keepdims=True))@router.utility+router.utility_bias
    assert np.array_equal(np.sign(router.scores(x)),np.sign(utility))


def test_time_blocks_do_not_overlap_and_seed_is_not_new_family():
    a,b=cases(101),cases(202)
    assert len(a)==32 and {c['family'] for c in a} == {c['family'] for c in b}
    for client in {c['client'] for c in a}:
        group=[c for c in a if c['client']==client]
        for previous,current in zip(group,group[1:]):
            assert previous['feedback_available_day'] < current['observed_block'][0]
        assert [c['phase'] for c in group] == ['acquisition']*2+['later_evaluation']*2
