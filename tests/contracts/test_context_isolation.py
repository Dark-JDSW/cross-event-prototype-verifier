from dataclasses import FrozenInstanceError, replace
import json

import pytest
from cross_event_verifier import (ActionEvent, ActionType, CrossEventVerifier,
    Decision, DecisionKind, FeatureBundle, Observation, VerificationState)
from cross_event_verifier.contexts import (ContextComposer, ContextContribution,
    DeviceContext, SpatialContext)
from cross_event_verifier.interaction import ContextRule, InteractionEvent
from cross_event_verifier.spatial import SpatialContextProvider
from tests.test_core import good_quality


def known():
    return Decision(DecisionKind.FORMAL_MATCH, VerificationState.CONFIRMED_IDENTITY, 'P1')


def action():
    return ActionEvent('P1', 1, ActionType.WAVE, .95, 10., 'cam')


def spatial(**changes):
    return replace(SpatialContext(track_id=1, camera_id='cam', timestamp=10.,
                                  zone_id='bedroom', world_position=(1., 2.)), **changes)


def rule():
    return ContextRule(identity_id='P1', action=ActionType.WAVE, zone_id='bedroom',
                       device_id='bedroom_light', operation='toggle')


def test_composition_policy_and_failure_never_write_identity_memory():
    verifier = CrossEventVerifier()
    try:
        observation = Observation(features=FeatureBundle([1, 0], [0, 1]), quality=good_quality())
        verifier.register_identity('P1', observation.features)
        decision = verifier.verify(observation)
        before_db = '\n'.join(verifier.store.connection.iterdump())
        before_vectors = [p.vector.copy() for p in verifier.memory.formal_prototypes('P1')]
        before_readiness = verifier.evaluate_gait_readiness('P1')
        device = DeviceContext(track_id=1, camera_id='cam', timestamp=10.,
                               nearest_device_id='bedroom_light', distance_to_device=.5)
        context = ContextComposer([SpatialContextProvider(spatial())]).compose(
            decision, action(), device=device)
        commands = rule().evaluate(InteractionEvent(context=context))
        assert len(commands) == 1 and commands[0].operation == 'toggle'
        payload = json.loads(json.dumps(InteractionEvent(context=context).to_dict()))
        assert payload['protocol'] == 'cevp.interaction.v1'
        assert payload['action']['protocol'] == 'cevp.action.v1'

        class Broken:
            def provide(self, request):
                assert not hasattr(request, 'verifier')
                raise RuntimeError('offline')
        failed = ContextComposer([Broken()]).compose(decision, action(), spatial=spatial(), device=device)
        assert failed.errors and rule().evaluate(InteractionEvent(context=failed)) == ()
        assert '\n'.join(verifier.store.connection.iterdump()) == before_db
        assert verifier.evaluate_gait_readiness('P1') == before_readiness
        for before, after in zip(before_vectors, verifier.memory.formal_prototypes('P1')):
            assert (before == after.vector).all()
        assert verifier.formal_identities == ('P1',)
        assert verifier.verify(replace(observation, event_id='still-alive')).identity_id == 'P1'
    finally:
        verifier.close()


@pytest.mark.parametrize('change', [dict(track_id=2), dict(camera_id='other'),
    dict(timestamp=8.), dict(timestamp=11.), dict(timestamp=float('nan')),
    dict(capture_session_id='old-session')])
def test_mismatched_spatial_facts_are_rejected_atomically(change):
    context = ContextComposer().compose(known(), action(), spatial=spatial(**change))
    assert context.errors and context.world_position is None and context.zone_id is None


@pytest.mark.parametrize('decision,event', [
    (Decision(DecisionKind.UNKNOWN, VerificationState.UNKNOWN), action()),
    (known(), replace(action(), identity_id='P2')),
    (known(), replace(action(), timestamp=float('nan'))),
])
def test_action_cannot_assign_or_override_identity(decision, event):
    context = ContextComposer().compose(decision, event, timestamp=10.)
    assert context.action is None and context.errors
    assert rule().evaluate(InteractionEvent(context=context)) == ()


def test_mutable_metadata_is_copied_and_nested_facts_are_read_only():
    source = {'occupancy': {'people': ['P1']}}
    contribution = spatial(attributes=source)
    source['occupancy']['people'].append('P2')
    context = ContextComposer().compose(known(), action(), spatial=contribution)
    assert context.attributes['occupancy']['people'] == ('P1',)
    with pytest.raises(TypeError):
        context.attributes['occupancy']['people'] = ()
    with pytest.raises(FrozenInstanceError):
        context.identity_id = 'P2'
    with pytest.raises(TypeError):
        ContextContribution(attributes={'client': object()})


def test_conflicts_do_not_overwrite_or_partially_merge():
    class Conflict:
        def provide(self, request):
            return spatial(zone_id='kitchen', nearest_device_id='wrong')
    context = ContextComposer([Conflict()]).compose(known(), action(), spatial=spatial())
    assert context.zone_id == 'bedroom' and context.nearest_device_id is None
    assert context.errors


def test_identity_workflow_decision_construction_preserves_existing_values():
    verifier = CrossEventVerifier()
    try:
        assert verifier._visual_identity_decision('P1') == Decision(
            DecisionKind.FORMAL_MATCH, VerificationState.CONFIRMED_IDENTITY,
            'P1', reasons=('visual_identity_binding_continues',))
        assert verifier._visual_identity_decision('P1', created=True, score=.9) == Decision(
            DecisionKind.VISUAL_IDENTITY_CREATED, VerificationState.CONFIRMED_IDENTITY,
            'P1', score=.9, reasons=('osnet_visual_identity_created',))
    finally:
        verifier.close()


def test_context_modules_do_not_import_core_implementations():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / 'cross_event_verifier'
    for name in ('contexts', 'interaction', 'spatial'):
        tree = ast.parse((root / (name + '.py')).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[-1] not in {
                    'engine', 'memory', 'storage', 'appearance_first', 'automation',
                    'production_vision', 'ground_control', 'pipeline'}
    # Optional extensions must not become imports in the identity dependency graph.
    for name in ('engine', 'memory', 'storage', 'types', 'config'):
        tree = ast.parse((root / (name + '.py')).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[-1] not in {'contexts', 'interaction', 'spatial'}
