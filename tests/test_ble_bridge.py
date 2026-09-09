from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from cross_event_verifier.actions import ActionType
from cross_event_verifier.ble_bridge import (
    BleActionPublisher,
    BleCommandMode,
    BleConfig,
    BleBridgeError,
    DEFAULT_COMMAND_UUID,
    DEFAULT_SERVICE_UUID,
    LIGHT_OFF_OPCODE,
    LIGHT_ON_OPCODE,
)
from cross_event_verifier.transient_actions import ActionEvent
from cross_event_verifier.transports import (
    ActionTransportMode,
    action_publisher_from_environment,
    normalize_transport_mode,
    transport_label,
)


class _Services:
    def get_service(self, uuid: str) -> object:
        return object()


class _FakeBleakClient:
    instances: list["_FakeBleakClient"] = []

    def __init__(self, address: str, timeout: float) -> None:
        self.address = address
        self.timeout = timeout
        self.services = _Services()
        self.is_connected = False
        self.writes: list[tuple[str, bytes, bool]] = []
        self.notify_uuid: str | None = None
        self._status_callback = None
        self.instances.append(self)

    async def connect(self) -> None:
        self.is_connected = True

    async def start_notify(self, uuid: str, callback) -> None:
        self.notify_uuid = uuid
        self._status_callback = callback

    async def write_gatt_char(self, uuid: str, payload: bytes, response: bool) -> None:
        self.writes.append((uuid, payload, response))
        if self._status_callback is not None:
            try:
                event_id = json.loads(payload)["event_id"]
                status = {"status": "executed", "event_id": event_id}
            except (UnicodeDecodeError, json.JSONDecodeError):
                command = "ON" if payload == LIGHT_ON_OPCODE else "OFF"
                status = {"status": "executed", "command": command}
            self._status_callback(uuid, json.dumps(status).encode())

    async def disconnect(self) -> None:
        self.is_connected = False


class BleBridgeTests(unittest.TestCase):
    def tearDown(self) -> None:
        _FakeBleakClient.instances.clear()

    def test_publishes_existing_action_payload_and_reuses_connection(self) -> None:
        publisher = BleActionPublisher(
            BleConfig(
                "AA:BB:CC:DD:EE:FF",
                command_mode=BleCommandMode.EVENT_JSON,
            ),
            client_factory=_FakeBleakClient,
        )
        try:
            event = ActionEvent("P1", 7, ActionType.WAVE, 0.9, 12.5, "living-room")
            publisher.publish(event)
            publisher.publish(event)
            self.assertEqual(len(_FakeBleakClient.instances), 1)
            client = _FakeBleakClient.instances[0]
            self.assertEqual(client.writes[0][0], DEFAULT_COMMAND_UUID)
            self.assertTrue(client.writes[0][2])
            self.assertEqual(json.loads(client.writes[0][1]), json.loads(
                '{"identity_id":"P1","track_id":7,"protocol":"cevp.action.v1",'
                '"action":"WAVE",'
                '"event_type":"wave","event_id":"living-room:track-7:wave:12500",'
                '"area":"living-room","completed":true,"confidence":0.9,'
                '"timestamp":12.5,"camera_id":"living-room"}'
            ))
            self.assertEqual(
                json.loads(publisher.last_status),
                {"status": "executed", "event_id": "living-room:track-7:wave:12500"},
            )
        finally:
            publisher.close()

    def test_missing_service_is_rejected(self) -> None:
        class NoService(_FakeBleakClient):
            class _NoServices:
                def get_service(self, uuid: str):
                    return None

            def __init__(self, address: str, timeout: float) -> None:
                super().__init__(address, timeout)
                self.services = self._NoServices()

        publisher = BleActionPublisher(BleConfig("endpoint"), client_factory=NoService)
        try:
            with self.assertRaises(BleBridgeError):
                publisher.publish(ActionEvent("P1", 1, ActionType.HAND_UP, 0.8, 1.0, "cam"))
        finally:
            publisher.close()

    def test_application_rejection_is_reported(self) -> None:
        class Rejected(_FakeBleakClient):
            async def write_gatt_char(self, uuid: str, payload: bytes, response: bool) -> None:
                self.writes.append((uuid, payload, response))
                if self._status_callback is not None:
                    event_id = json.loads(payload)["event_id"]
                    self._status_callback(
                        uuid,
                        json.dumps(
                            {"status": "rejected", "event_id": event_id, "reason": "not_allowed"}
                        ).encode(),
                    )

        publisher = BleActionPublisher(
            BleConfig("endpoint", command_mode=BleCommandMode.EVENT_JSON),
            client_factory=Rejected,
        )
        try:
            with self.assertRaisesRegex(BleBridgeError, "not_allowed"):
                publisher.publish(ActionEvent("P1", 1, ActionType.WAVE, 0.8, 1.0, "cam"))
        finally:
            publisher.close()

    def test_light_mode_sends_only_mapped_opcode(self) -> None:
        publisher = BleActionPublisher(
            BleConfig("endpoint", command_mode=BleCommandMode.LIGHT),
            client_factory=_FakeBleakClient,
        )
        try:
            publisher.publish(ActionEvent("P1", 1, ActionType.HAND_UP, 0.8, 1.0, "cam"))
            publisher.publish(ActionEvent("P1", 1, ActionType.SQUAT, 0.8, 2.0, "cam"))
            publisher.publish(ActionEvent("P1", 1, ActionType.WAVE, 0.8, 3.0, "cam"))
            client = _FakeBleakClient.instances[0]
            self.assertEqual(
                [payload for _, payload, _ in client.writes],
                [LIGHT_ON_OPCODE, LIGHT_OFF_OPCODE],
            )
            self.assertEqual(json.loads(publisher.last_status)["command"], "OFF")
        finally:
            publisher.close()

    def test_direct_ble_config_defaults_to_light_mode(self) -> None:
        self.assertIs(BleConfig("endpoint").command_mode, BleCommandMode.LIGHT)

    def test_environment_factory_is_optional_and_mode_is_explicit(self) -> None:
        previous = os.environ.pop("CROSS_EVENT_BLE_ADDRESS", None)
        try:
            self.assertIsNone(action_publisher_from_environment(ActionTransportMode.BLE))
            self.assertIsNone(action_publisher_from_environment("ble"))
            self.assertEqual(ActionTransportMode.MQTT.value, "mqtt")
            self.assertEqual(DEFAULT_SERVICE_UUID.count("-"), 4)
        finally:
            if previous is not None:
                os.environ["CROSS_EVENT_BLE_ADDRESS"] = previous

    def test_transport_labels_expose_only_mqtt_and_ble_light(self) -> None:
        self.assertIs(normalize_transport_mode("MQTT"), ActionTransportMode.MQTT)
        self.assertIs(normalize_transport_mode("BLE Light"), ActionTransportMode.BLE_LIGHT)
        self.assertIs(normalize_transport_mode("ble-light"), ActionTransportMode.BLE_LIGHT)
        self.assertEqual(transport_label(ActionTransportMode.BLE), "BLE Light")
        with self.assertRaises(ValueError):
            normalize_transport_mode("websocket")
        with self.assertRaises(ValueError):
            normalize_transport_mode("http")

    def test_application_factory_forces_ble_light_payload(self) -> None:
        previous = {
            key: os.environ.get(key)
            for key in ("CROSS_EVENT_BLE_ADDRESS", "CROSS_EVENT_BLE_COMMAND_MODE")
        }
        os.environ["CROSS_EVENT_BLE_ADDRESS"] = "endpoint"
        os.environ["CROSS_EVENT_BLE_COMMAND_MODE"] = "event_json"
        try:
            with patch("cross_event_verifier.transports.BleActionPublisher") as publisher:
                action_publisher_from_environment("BLE Light")
                config = publisher.call_args.args[0]
                self.assertIs(config.command_mode, BleCommandMode.LIGHT)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
