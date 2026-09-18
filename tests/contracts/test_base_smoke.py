from dataclasses import replace
import pytest

from cross_event_verifier import CrossEventVerifier, FeatureBundle, Observation, DecisionKind
from tests.test_core import good_quality


def test_base_register_verify_batch_and_controlled_promotion():
    v = CrossEventVerifier()
    try:
        observation = Observation(features=FeatureBundle([1, 0], [0, 1]),
                                  quality=good_quality())
        v.register_identity('P1', observation.features)
        assert v.formal_identities == ('P1',)
        assert v.verify(observation).identity_id == 'P1'
        assert len(v.verify_batch([replace(observation, event_id='batch')])) == 1
        unknown = Observation(features=FeatureBundle([0, 1], [1, 0]),
                              quality=good_quality())
        v.verify(unknown, candidate_id='candidate')
        assert v.get_candidate('candidate') is not None
        assert 'candidate' not in v.formal_identities
        with pytest.raises(ValueError):
            v.promote_candidate('candidate')
        result = v.promote_candidate('candidate', identity_id='P2', force=True)
        assert result.identity_id == 'P2'
        assert 'P2' in v.formal_identities
    finally:
        v.close()


def test_other_encoder_cannot_match_same_dimension_vector():
    v = CrossEventVerifier()
    try:
        v.register_identity('P1', FeatureBundle(gait=[1, 0]), model_version='GaitGraph2')
        result = v.verify(Observation(features=FeatureBundle(gait=[1, 0]),
                                      quality=good_quality(), model_version='HyperGait'))
        assert result.kind != DecisionKind.FORMAL_MATCH
    finally:
        v.close()
