from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from cross_event_verifier import (
    AutomationPolicy,
    CrossEventVerifier,
    Decision,
    DecisionKind,
    FeatureBundle,
    Observation,
    VerifierConfig,
    VerificationState,
)
from cross_event_verifier.appearance_first import (
    AppearanceFirstGaitEnrollmentController,
)
from cross_event_verifier.automation import AutomaticVerificationController
from cross_event_verifier.gui_presenter import GuiPresenter
from cross_event_verifier.storage import SqliteStore
from tests.test_appearance_first import _observation
from tests.test_registration_regressions import good_quality


def _automation_policy() -> AutomationPolicy:
    return AutomationPolicy(
        minimum_track_frames=1,
        minimum_stable_gait_samples=2,
        gait_sample_window=3,
        minimum_independent_gait_events=1,
        minimum_sample_similarity=0.80,
        minimum_gait_stability=0.80,
    )


def test_shared_helpers_do_not_import_identity_core_or_optional_runtimes():
    package = Path(__file__).resolve().parents[1] / 'cross_event_verifier'
    forbidden = {
        'engine', 'memory', 'storage', 'pipeline', 'contexts', 'interaction',
        'spatial', 'cv2', 'torch', 'PyQt6', 'paho', 'bleak', 'tensorrt',
    }
    for module_name in ('orchestration', 'gui_presenter'):
        tree = ast.parse((package / f'{module_name}.py').read_text(encoding='utf-8'))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or '').split('.')[-1])
        assert not imported.intersection(forbidden), module_name


def test_gait_first_keeps_using_the_legacy_core_enrollment_api():
    verifier = CrossEventVerifier()
    controller = AutomaticVerificationController(verifier, _automation_policy())
    try:
        with (
            patch.object(
                verifier,
                'enroll_gait_identity',
                wraps=verifier.enroll_gait_identity,
            ) as gait_first_enrollment,
            patch.object(
                verifier,
                'register_identity',
                wraps=verifier.register_identity,
            ) as trusted_registration,
        ):
            for timestamp in (1.0, 2.0):
                decision, _ = controller.verify(
                    Observation(
                        event_id=f'gait-first-{timestamp}',
                        camera_id='cam-a',
                        capture_session_id='gait-session',
                        track_id='track-1',
                        timestamp=timestamp,
                        features=FeatureBundle(gait=[0.0, 1.0, 0.0]),
                        quality=good_quality(),
                    ),
                    candidate_id='gait-candidate',
                )

        assert gait_first_enrollment.call_count == 1
        # The legacy Core method performs the trusted registration internally;
        # the controller must enter through enroll_gait_identity first.
        assert trusted_registration.call_count == 1
        assert decision.identity_id == 'P1'
        assert verifier.formal_identities == ('P1',)
    finally:
        verifier.close()


def test_appearance_first_creates_visual_identity_then_enrolls_walk_gait():
    verifier = CrossEventVerifier(
        VerifierConfig(
            appearance_identity_min_samples=2,
            gait_provisional_min_events=2,
            gait_ready_min_events=3,
            gait_ready_min_coverage=1,
            gait_event_min_similarity=0.70,
            gait_holdout_min_similarity=0.70,
        )
    )
    controller = AppearanceFirstGaitEnrollmentController(
        verifier,
        _automation_policy(),
    )
    try:
        with (
            patch.object(
                verifier,
                'register_identity',
                wraps=verifier.register_identity,
            ) as visual_registration,
            patch.object(
                verifier,
                'enroll_gait_identity',
                wraps=verifier.enroll_gait_identity,
            ) as gait_first_enrollment,
            patch.object(
                verifier,
                'enroll_gait_prototype',
                wraps=verifier.enroll_gait_prototype,
            ) as gait_prototype_enrollment,
        ):
            decisions = []
            for timestamp in (1.0, 2.0, 3.0):
                decision, _ = controller.verify(
                    _observation(
                        'appearance-session',
                        np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        timestamp,
                    ),
                    candidate_id='appearance-track',
                )
                decisions.append(decision)

        assert visual_registration.call_count == 1
        assert gait_first_enrollment.call_count == 0
        assert gait_prototype_enrollment.call_count == 1
        assert decisions[1].kind == DecisionKind.VISUAL_IDENTITY_CREATED
        assert verifier.memory.formal_prototypes('P1', 'appearance')
        assert verifier.memory.formal_prototypes('P1', 'gait')
    finally:
        verifier.close()


def test_event_key_precedence_remains_strategy_specific():
    observation = Observation(
        event_id='event-id',
        capture_session_id='session-id',
        challenge_id='challenge-id',
    )
    assert AutomaticVerificationController._gait_event_key(observation) == (
        'challenge:challenge-id'
    )
    assert AppearanceFirstGaitEnrollmentController._event_key(observation) == (
        'session:session-id'
    )


def test_legacy_sqlite_schema_migrates_without_losing_prototypes_or_audit():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'legacy.sqlite3'
        connection = sqlite3.connect(path)
        connection.executescript(
            '''
            CREATE TABLE prototypes (
                prototype_id TEXT PRIMARY KEY,
                identity_id TEXT NOT NULL,
                zone TEXT NOT NULL,
                modality TEXT NOT NULL,
                vector BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                quality REAL NOT NULL,
                camera_id TEXT,
                view_angle TEXT,
                clothing_tag TEXT,
                source_event_id TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                entity_id TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            PRAGMA user_version = 1;
            '''
        )
        vector = np.array([1.0, 0.0], dtype=np.float32)
        connection.execute(
            '''INSERT INTO prototypes(
                prototype_id,identity_id,zone,modality,vector,dimension,quality,
                camera_id,view_angle,clothing_tag,source_event_id,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
            (
                'legacy-prototype', 'P1', 'formal', 'gait', vector.tobytes(),
                2, 0.9, 'cam-a', 'side', None, 'legacy-event', 1.0,
            ),
        )
        connection.execute(
            '''INSERT INTO audit_log(action,entity_id,payload_json,created_at)
               VALUES(?,?,?,?)''',
            ('legacy_registration', 'P1', '{"operator":"alice"}', 1.0),
        )
        connection.commit()
        connection.close()

        store = SqliteStore(str(path))
        try:
            prototypes = store.load_prototypes()
            assert len(prototypes) == 1
            assert prototypes[0].prototype_id == 'legacy-prototype'
            assert prototypes[0].model_version == 'unconfigured'
            assert prototypes[0].feature_schema == 'unconfigured-v1'
            assert prototypes[0].embedding_dimensions == {}
            audit = store.audit_log()
            assert len(audit) == 1
            assert audit[0]['audit_id'] == 1
            assert audit[0]['action'] == 'legacy_registration'
            assert audit[0]['entity_id'] == 'P1'
            assert audit[0]['payload_json'] == '{"operator":"alice"}'
            assert audit[0]['payload'] == {'operator': 'alice'}
            assert audit[0]['created_at'] == 1.0
            assert store.connection.execute('PRAGMA user_version').fetchone()[0] == 4
            assert {
                row[0]
                for row in store.connection.execute(
                    'SELECT version FROM schema_migrations'
                )
            } == {2, 3, 4}
        finally:
            store.close()


@pytest.mark.parametrize('action', ['WAVE', 'SQUAT', 'SIT_DOWN', 'STAND_UP', 'HAND_UP'])
def test_short_actions_never_write_gait_or_change_readiness(action: str):
    verifier = CrossEventVerifier()
    verifier.register_identity('P1', FeatureBundle(appearance=[1.0, 0.0, 0.0]))
    controller = AppearanceFirstGaitEnrollmentController(verifier, _automation_policy())
    state_key = 'bound-track'
    try:
        controller.register_visual_identity(
            'P1',
            _observation(
                'visual-session',
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                0.0,
            ),
            state_key=state_key,
        )
        before = verifier.evaluate_gait_readiness('P1')
        observation = replace(
            _observation(
                'visual-session',
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                1.0,
            ),
            metadata={
                'action_type': action,
                'action_confidence': 0.99,
                'action_quality': 'STRONG',
                'action_completion': 1.0,
                'action_source': 'regression-test',
                'action_model_version': 'action-test-v1',
                'action_segment_id': f'{action}-1',
            },
        )

        with patch.object(
            verifier,
            'enroll_gait_prototype',
            wraps=verifier.enroll_gait_prototype,
        ) as gait_write:
            decision, status = controller.verify(observation, candidate_id=state_key)

        assert gait_write.call_count == 0
        assert verifier.memory.formal_prototypes('P1', 'gait') == ()
        assert verifier.evaluate_gait_readiness('P1') == before
        assert 'action_quarantine' in decision.reasons
        assert status.action_type == action
    finally:
        verifier.close()


def test_gui_presenter_builds_the_same_source_and_frame_models_for_both_toolkits():
    presenter = GuiPresenter()
    camera = presenter.source_selection(
        source_kind='camera',
        camera_index='2',
        camera_id='',
        video_path='',
        video_repeat_count='1',
        candidate_id=' candidate ',
    )
    video = presenter.source_selection(
        source_kind='file',
        camera_index='0',
        camera_id='ignored',
        video_path='clips/walk.mp4',
        video_repeat_count='3',
        candidate_id='',
    )
    assert camera.as_source_args() == ('camera', 2, 'camera-2', 'candidate', 1)
    assert video.as_source_args() == ('file', 'clips/walk.mp4', 'walk.mp4', None, 3)

    decision = Decision(
        DecisionKind.FORMAL_MATCH,
        VerificationState.CONFIRMED_IDENTITY,
        identity_id='P1',
        score=0.9876,
    )
    track = SimpleNamespace(
        track_id=7,
        action_type='WALK',
        decision=decision,
        automation=SimpleNamespace(message='P1：步态学习中'),
    )
    frame = SimpleNamespace(
        tracks=(track,),
        formal_identities=('P1',),
        pending_request_ids=(),
        frame_index=12,
        processing_seconds=0.015,
        transport_errors=(),
    )
    model = presenter.frame(
        frame,
        automatic_registration_enabled=True,
        previous_action_target='',
    )

    assert model.rows[0].values == (
        '7', 'WALK', 'P1', 'formal_match', 'P1：步态学习中', '0.988')
    assert model.rows[0].visual_tag == 'verified'
    assert model.action_targets.targets == {'Track 7 · P1': 7}
    assert model.action_targets.selected_label == 'Track 7 · P1'
    assert model.gallery_text == 'P1'
    assert model.pending_requests_text == '无'
    assert model.status_text == '帧 12 | 目标 1 | 处理 15 ms'


@pytest.mark.parametrize(
    ('source_kind', 'camera_index', 'video_path', 'repeat_count', 'message'),
    [
        ('camera', 'bad', '', '1', '摄像头设备号必须是整数'),
        ('file', '0', '', '1', '请先选择视频文件'),
        ('file', '0', 'walk.mp4', '0', '视频重复学习次数必须是正整数'),
    ],
)
def test_gui_presenter_keeps_legacy_source_validation_messages(
    source_kind: str,
    camera_index: str,
    video_path: str,
    repeat_count: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        GuiPresenter().source_selection(
            source_kind=source_kind,
            camera_index=camera_index,
            camera_id='',
            video_path=video_path,
            video_repeat_count=repeat_count,
            candidate_id='',
        )
