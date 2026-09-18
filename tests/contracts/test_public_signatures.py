from inspect import signature

import pytest
from cross_event_verifier import CrossEventVerifier, FeatureBundle, Observation, VerifierConfig
from cross_event_verifier.smart_home import ActionPublisher
from cross_event_verifier.vision import VisionAdapter


@pytest.mark.parametrize('name,args,kwargs', [
    ('__init__', (None,), {}),
    ('__init__', (None, VerifierConfig()), {'store': None, 'memory': None}),
    ('verify', (None, Observation()), {}),
    ('verify', (None, Observation()), {'candidate_id': 'c', 'open_set_guard': True}),
    ('verify_batch', (None, []), {}),
    ('verify_batch', (None, []), {'candidate_ids': [], 'open_set_guard': False}),
    ('register_identity', (None, 'P1', FeatureBundle(gait=[1, 0])), {}),
    ('register_identity', (None, 'P1', FeatureBundle()), {
        'metadata': {}, 'camera_id': 'cam', 'view_angle': None, 'clothing_tag': None,
        'quality': 1., 'source_event_id': 'e', 'model_version': 'm', 'feature_schema': 's',
        'artifact_sha256': 'a', 'preprocess_version': 'p', 'joint_format': 'j',
        'sequence_length': 60, 'tta_mode': 'off', 'coordinate_contract': 'c',
        'embedding_dimensions': {'gait': 2}}),
    ('promote_candidate', (None, 'c'), {}),
    ('promote_candidate', (None, 'c'), {'identity_id': 'P1', 'force': True}),
    ('get_candidate', (None, 'c'), {}), ('list_candidates', (None,), {}),
    ('rollback', (None, 'P1', 1), {}), ('maintenance', (None,), {'now': 1.}),
    ('close', (None,), {}), ('clear_gallery', (None,), {}),
])
def test_old_call_shapes_bind_without_freezing_complete_signatures(name, args, kwargs):
    signature(getattr(CrossEventVerifier, name)).bind(*args, **kwargs)


def test_vision_and_publisher_call_shapes():
    signature(VisionAdapter.process).bind(None, object())
    signature(VisionAdapter.reset).bind(None)
    signature(ActionPublisher.publish).bind(None, object())
    signature(ActionPublisher.close).bind(None)
