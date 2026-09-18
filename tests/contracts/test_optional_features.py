from cross_event_verifier import CrossEventVerifier, FeatureBundle, Observation, TrackQuality
from cross_event_verifier.contexts import ContextComposer
from cross_event_verifier.interaction import ContextRule, InteractionEvent
from cross_event_verifier.spatial import VisionMetadataSpatialProvider
from cross_event_verifier.vision import VisionTrack


def test_empty_composer_and_policy_are_neutral():
    verifier = CrossEventVerifier()
    try:
        decision = verifier.verify(Observation())
        context = ContextComposer(None).compose(decision)
        assert context.identity_id is None and context.errors == ()
        assert ContextRule().evaluate(InteractionEvent(context=context)) == ()
        assert verifier.formal_identities == ()
    finally:
        verifier.close()


def test_spatial_is_optional_and_reuses_calibrated_facts():
    provider = VisionMetadataSpatialProvider()
    track = VisionTrack(1, (0, 0, 100, 200), .9, FeatureBundle(), TrackQuality())
    assert provider.locate(track, camera_id='c', timestamp=1.) is None
    from dataclasses import replace
    track = replace(track, metadata={'ground_calibrated': True, 'world_position': (1., 2.),
        'ground_coordinate_frame': 'site-m', 'ground_calibration_id': 'cal1',
        'camera_zone': 'left', 'world_direction_degrees': 45.})
    spatial = provider.locate(track, camera_id='c', timestamp=1.)
    assert spatial.world_position == (1., 2.)
    assert spatial.zone_id is None  # image bin cannot be promoted to bedroom
    assert spatial.facing_direction is None  # motion direction is not body facing


def test_replacement_vision_runtime_needs_no_identity_logic():
    class Runtime:
        def process(self, frame_bgr):
            return (VisionTrack(1, (0, 0, 100, 200), .9, FeatureBundle(), TrackQuality()),)
        def reset(self):
            pass
    vision = Runtime()
    verifier = CrossEventVerifier()
    try:
        track, = vision.process(None)
        observation = Observation(features=track.features, quality=track.quality,
                                  track_id=str(track.track_id))
        assert verifier.verify(observation).identity_id is None
        vision.reset()
    finally:
        verifier.close()
