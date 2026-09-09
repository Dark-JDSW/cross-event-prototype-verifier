from __future__ import annotations

import json

import numpy as np

from cross_event_verifier.actions import ActionType
from cross_event_verifier.smart_home import MqttConfig, event_payload, event_topic
from cross_event_verifier.transient_actions import ActionEvent, TransientActionManager


BOX = (0, 0, 100, 180)


def _standing_pose() -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float32)
    points = {
        5: (40, 40),
        6: (60, 40),
        7: (38, 65),
        8: (62, 65),
        9: (36, 85),
        10: (64, 85),
        11: (43, 80),
        12: (57, 80),
        13: (43, 120),
        14: (57, 120),
        15: (43, 160),
        16: (57, 160),
    }
    for index, (x, y) in points.items():
        pose[index] = (x, y, 0.95)
    return pose


def _squat_pose() -> np.ndarray:
    pose = _standing_pose()
    pose[11] = (43, 92, 0.95)
    pose[12] = (57, 92, 0.95)
    pose[13] = (25, 120, 0.95)
    pose[14] = (75, 120, 0.95)
    pose[15] = (43, 148, 0.95)
    pose[16] = (57, 148, 0.95)
    return pose


def _seated_pose() -> np.ndarray:
    pose = _standing_pose()
    pose[11] = (35, 100, 0.95)
    pose[12] = (45, 100, 0.95)
    pose[13] = (70, 102, 0.95)
    pose[14] = (80, 102, 0.95)
    pose[15] = (70, 145, 0.95)
    pose[16] = (80, 145, 0.95)
    return pose


def _authorized(manager: TransientActionManager, timestamp: float) -> tuple[ActionEvent, ...]:
    return manager.authorize(7, "P1", timestamp=timestamp, camera_id="living-room")


def test_completed_squat_is_buffered_until_identity_is_confirmed() -> None:
    manager = TransientActionManager()
    manager.observe(7, _standing_pose(), BOX, timestamp=0.0, segment_id=1)
    manager.observe(7, _squat_pose(), BOX, timestamp=0.3, segment_id=1)
    manager.observe(7, _standing_pose(), BOX, timestamp=0.8, segment_id=1)

    assert manager.authorize(7, None, timestamp=0.8, camera_id="living-room") == ()
    events = _authorized(manager, 0.9)

    assert [event.action for event in events] == [ActionType.SQUAT]
    assert events[0].identity_id == "P1"


def test_sitting_then_rising_emits_distinct_completed_actions() -> None:
    manager = TransientActionManager()
    manager.observe(7, _standing_pose(), BOX, timestamp=0.0)
    manager.observe(7, _seated_pose(), BOX, timestamp=0.2)
    manager.observe(7, _seated_pose(), BOX, timestamp=1.0)
    assert [event.action for event in _authorized(manager, 1.0)] == [ActionType.SIT_DOWN]

    manager.observe(7, _standing_pose(), BOX, timestamp=1.4)
    assert [event.action for event in _authorized(manager, 1.4)] == [ActionType.STAND_UP]


def test_held_raised_hand_emits_once_until_lowered() -> None:
    manager = TransientActionManager()
    raised = _standing_pose()
    raised[9] = (30, 20, 0.95)
    manager.observe(7, raised, BOX, timestamp=0.0)
    manager.observe(7, raised, BOX, timestamp=0.6)
    assert [event.action for event in _authorized(manager, 0.6)] == [ActionType.HAND_UP]

    manager.observe(7, raised, BOX, timestamp=0.9)
    assert _authorized(manager, 0.9) == ()


def test_horizontal_reversals_of_raised_wrist_emit_wave() -> None:
    manager = TransientActionManager()
    for timestamp, x in ((0.0, 20), (0.2, 80), (0.4, 20), (0.6, 80)):
        pose = _standing_pose()
        pose[9] = (x, 20, 0.95)
        manager.observe(7, pose, BOX, timestamp=timestamp)

    assert [event.action for event in _authorized(manager, 0.6)] == [ActionType.WAVE]


def test_identity_switch_discards_unreleased_action() -> None:
    manager = TransientActionManager()
    manager.observe(7, _standing_pose(), BOX, timestamp=0.0)
    manager.authorize(7, "P1", timestamp=0.0, camera_id="living-room")
    manager.observe(7, _squat_pose(), BOX, timestamp=0.2)
    assert manager.authorize(7, "P2", timestamp=0.2, camera_id="living-room") == ()
    manager.observe(7, _standing_pose(), BOX, timestamp=0.5)
    assert manager.authorize(7, "P2", timestamp=0.5, camera_id="living-room") == ()


def test_subtracklet_boundary_cannot_complete_previous_motion() -> None:
    manager = TransientActionManager()
    manager.observe(7, _standing_pose(), BOX, timestamp=0.0, segment_id=1)
    manager.observe(7, _squat_pose(), BOX, timestamp=0.2, segment_id=1)
    manager.observe(7, _standing_pose(), BOX, timestamp=0.5, segment_id=2)

    assert _authorized(manager, 0.5) == ()


def test_mqtt_contract_contains_identity_action_location_and_timestamp() -> None:
    event = ActionEvent("P1", 7, ActionType.WAVE, 0.9, 12.5, "living-room")
    config = MqttConfig(host="broker.local", topic_prefix="vision")

    assert event_topic(config, event) == "vision/P1/action"
    assert json.loads(event_payload(event)) == {
        "identity_id": "P1",
        "track_id": 7,
        "protocol": "cevp.action.v1",
        "action": "WAVE",
        "event_type": "wave",
        "event_id": "living-room:track-7:wave:12500",
        "area": "living-room",
        "completed": True,
        "confidence": 0.9,
        "timestamp": 12.5,
        "camera_id": "living-room",
    }
