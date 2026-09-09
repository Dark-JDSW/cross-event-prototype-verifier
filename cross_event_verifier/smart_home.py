"""Optional MQTT output for identity-authorized short action events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Protocol

from .transient_actions import ActionEvent


# Shared by MQTT and BLE so the ESP32 endpoint and Home Assistant consume the
# same versioned event contract.  Keep the existing field names for consumers
# that already subscribe to the MQTT topic.
ACTION_EVENT_PROTOCOL = "cevp.action.v1"


class ActionPublisher(Protocol):
    """Minimal sink contract shared by MQTT and BLE transports."""

    def publish(self, event: ActionEvent) -> None:
        """Deliver one completed action event."""

    def close(self) -> None:
        """Release network or radio resources owned by the sink."""


@dataclass(frozen=True)
class MqttConfig:
    host: str
    port: int = 1883
    topic_prefix: str = "cross-event-verifier"
    username: str | None = None
    password: str | None = None
    qos: int = 1

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("MQTT host cannot be blank")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("MQTT port must be in [1, 65535]")
        if int(self.qos) not in (0, 1, 2):
            raise ValueError("MQTT qos must be 0, 1, or 2")
        if not self.topic_prefix.strip(" /"):
            raise ValueError("MQTT topic prefix cannot be blank")

    @classmethod
    def from_environment(cls) -> "MqttConfig | None":
        host = os.getenv("CROSS_EVENT_MQTT_HOST", "").strip()
        if not host:
            return None
        return cls(
            host=host,
            port=int(os.getenv("CROSS_EVENT_MQTT_PORT", "1883")),
            topic_prefix=os.getenv("CROSS_EVENT_MQTT_TOPIC_PREFIX", "cross-event-verifier"),
            username=os.getenv("CROSS_EVENT_MQTT_USERNAME") or None,
            password=os.getenv("CROSS_EVENT_MQTT_PASSWORD") or None,
            qos=int(os.getenv("CROSS_EVENT_MQTT_QOS", "1")),
        )


def event_topic(config: MqttConfig, event: ActionEvent) -> str:
    return f"{config.topic_prefix.strip('/')}/{event.identity_id}/action"


def event_id(event: ActionEvent) -> str:
    """Return the deterministic identifier used for delivery de-duplication."""

    return (
        f"{event.camera_id}:track-{event.track_id}:"
        f"{event.action.value.lower()}:{round(event.timestamp * 1000)}"
    )


def event_payload(event: ActionEvent) -> str:
    payload = asdict(event)
    payload["protocol"] = ACTION_EVENT_PROTOCOL
    payload["action"] = event.action.value
    payload["event_type"] = event.action.value.lower()
    payload["event_id"] = event_id(event)
    payload["area"] = event.camera_id
    payload["completed"] = True
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class MqttActionPublisher:
    """Publish action events without coupling vision code to Home Assistant entities."""

    def __init__(self, config: MqttConfig) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as error:
            raise RuntimeError(
                "MQTT is configured but paho-mqtt is not installed; install the smart-home dependency"
            ) from error
        self.config = config
        self._mqtt: Any = mqtt
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            protocol=mqtt.MQTTv5,
        )
        if config.username is not None:
            self._client.username_pw_set(config.username, config.password)
        self._client.max_queued_messages_set(32)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.connect_async(config.host, int(config.port), keepalive=30)
        self._client.loop_start()

    @classmethod
    def from_environment(cls) -> "MqttActionPublisher | None":
        config = MqttConfig.from_environment()
        return cls(config) if config is not None else None

    def publish(self, event: ActionEvent) -> None:
        info = self._client.publish(
            event_topic(self.config, event),
            event_payload(event),
            qos=int(self.config.qos),
            retain=False,
        )
        if info.rc != self._mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish rejected with code {info.rc}")

    def close(self) -> None:
        self._client.disconnect()
        self._client.loop_stop()


__all__ = [
    "ACTION_EVENT_PROTOCOL",
    "ActionPublisher",
    "MqttActionPublisher",
    "MqttConfig",
    "event_payload",
    "event_id",
    "event_topic",
]
