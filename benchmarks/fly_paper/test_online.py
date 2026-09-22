import numpy as np

from benchmarks.fly_paper.model import Bout, simulate
from benchmarks.fly_paper.online import CausalMemory
from benchmarks.fly_paper.test_model import simple_params


def test_single_conditioning_native_protocol_agrees_with_causal_event_clock():
    p = simple_params()
    p[2][0,0,0] = -1.
    # Last native training bout is the actually observed conditioning event.
    events = [Bout("training",30.,(1.,0.),1.), Bout("rest",11000.,(0.,0.)),
              Bout("imaging",5.,(1.,0.),imaging=1),
              Bout("imaging",5.,(0.,1.),imaging=2)]
    _,native = simulate(p,events,trace=True)
    online = CausalMemory(p)
    for i,b in enumerate(events):
        x = online.step(b.duration,b.odor,b.punishment,conditioning=b.name=="training")
        np.testing.assert_allclose(x,native["Dx_DAN_MBON"][0,:,i],atol=1e-12)
        np.testing.assert_allclose(online.weights,native["W_KDKM_KC"][0,:,:,i],atol=1e-12)


def test_scoring_does_not_write_memory_or_advance_clock():
    m = CausalMemory(simple_params())
    m.reinforce_return(.01)
    before = (m.weights.copy(),m.odor_state.copy(),m.clock)
    m.cue_scores(); m.cue_scores()
    np.testing.assert_array_equal(m.weights,before[0])
    np.testing.assert_array_equal(m.odor_state,before[1])
    assert m.clock == before[2]


def test_common_prefix_has_identical_state_without_future_inputs():
    a,b = CausalMemory(simple_params()),CausalMemory(simple_params())
    for r in (.005,-.01,.004):
        a.reinforce_return(r); b.reinforce_return(r)
    before = a.weights.copy()
    np.testing.assert_array_equal(a.weights,b.weights)
    b.reinforce_return(-.5)
    np.testing.assert_array_equal(a.weights,before)
