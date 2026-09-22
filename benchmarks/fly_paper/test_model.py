"""Small numerical guardrails; native replay is the primary validation.

SPDX-License-Identifier: GPL-3.0-or-later
"""
import numpy as np
import pytest

from benchmarks.fly_paper.model import Bout, parameter_matrices, simulate


def simple_params(samples=1):
    return [np.tile([[[0.,0.,0.,10.,2.,3.]]],(samples,1,1)),
            np.full((samples,1,1),-1.), np.zeros((samples,6,2)),
            np.zeros((samples,6,6)), np.full((samples,1,3),1e6),
            np.full((samples,1,1),100.)]


def test_parameter_mapping_preserves_matlab_column_major_order_and_sample_axis():
    lo = np.array([[0.,10.],[20.,30.]])
    hi = np.array([[1.,10.],[21.,31.]])
    bounds = np.empty((1,1),dtype=object)
    bounds[0,0] = np.stack([lo,hi],axis=-1)
    out = parameter_matrices([[1,2,3],[4,5,6]],bounds)[0]
    np.testing.assert_array_equal(out,[[[1,10],[2,3]],[[4,10],[5,6]]])


def test_zero_input_has_zero_response_not_baseline_reward():
    p = simple_params()
    protocol = [Bout("training",30.,(0.,0.)),Bout("imaging",5.,(0.,0.),imaging=1),
                Bout("imaging",5.,(0.,0.),imaging=2)]
    y,trace = simulate(p,protocol,trace=True)
    np.testing.assert_array_equal(y,0.)
    np.testing.assert_array_equal(trace["W_KDKM_KC"][...,0],np.repeat(p[0],2,axis=1))


def test_dopamine_sign_reverses_plasticity_in_isolated_no_feedback_case():
    protocol = [Bout("training",30.,(1.,0.))]
    p = simple_params(2)
    p[0][:,0,0] = [1.,-1.]
    _,trace = simulate(p,protocol,trace=True)
    delta = trace["W_KDKM_KC"][:,0,3,0]-p[0][:,0,3]
    assert delta[0] < 0 < delta[1]
    np.testing.assert_allclose(delta[0],-delta[1],rtol=1e-12)
    np.testing.assert_array_equal(trace["W_KDKM_KC"][:,1,:,0],p[0][:,0,:])


def test_batch_samples_are_independent():
    p = simple_params(2)
    p[0][1,0,0] = 1.
    protocol = [Bout("training",30.,(1.,0.)),Bout("imaging",5.,(1.,0.),imaging=1),
                Bout("imaging",5.,(0.,1.),imaging=2)]
    batched,_ = simulate(p,protocol)
    for i in range(2):
        individual,_ = simulate([x[i:i+1] for x in p],protocol)
        np.testing.assert_allclose(batched[i:i+1],individual,rtol=0,atol=0)


def test_native_retention_schedule_rejects_undefined_training_anchor():
    with pytest.raises(ValueError,match="training"):
        simulate(simple_params(),[Bout("rest",100.,(0.,0.))])
    with pytest.raises(ValueError,match="Negative"):
        simulate(simple_params(),[Bout("training",-1.,(1.,0.))])
