"""Select one action-event transport for the desktop pipeline."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

from .ble_bridge import BleActionPublisher, BleCommandMode, BleConfig
from .smart_home import ActionPublisher, MqttActionPublisher


class ActionTransportMode(str, Enum):
    """The two action-event transports exposed by the GUI.

    ``BLE`` is the stable environment/API value; its user-facing name is
    ``BLE Light`` because the endpoint is restricted to the ESP32-S3 light
    control GATT service.
    """

    MQTT = "mqtt"
    BLE_LIGHT = "ble"
    # Keep the old Python name as an alias for callers that already imported
    # ``ActionTransportMode.BLE``; it is not a third transport.
    BLE = "ble"


def normalize_transport_mode(mode: ActionTransportMode | str) -> ActionTransportMode:
    """Normalize the two supported transport labels and reject everything else."""

    if isinstance(mode, ActionTransportMode):
        return mode
    normalized = str(mode).strip().lower().replace("_", " ").replace("-", " ")
    if normalized == "ble light":
        normalized = ActionTransportMode.BLE.value
    return ActionTransportMode(normalized)


def transport_label(mode: ActionTransportMode | str) -> str:
    """Return the concise label shown in either desktop GUI."""

    selected = normalize_transport_mode(mode)
    return "MQTT" if selected is ActionTransportMode.MQTT else "BLE Light"


def action_publisher_from_environment(
    mode: ActionTransportMode | str,
) -> ActionPublisher | None:
    """Build the selected publisher, or ``None`` when it is unconfigured."""

    selected = normalize_transport_mode(mode)
    if selected is ActionTransportMode.MQTT:
        return MqttActionPublisher.from_environment()
    if selected is ActionTransportMode.BLE_LIGHT:
        # The desktop transport is deliberately light-only.  Keep the lower
        # level BleActionPublisher's JSON mode for library compatibility, but
        # never expose it as a selectable output mode in the application.
        config = BleConfig.from_environment()
        if config is None:
            return None
        return BleActionPublisher(
            replace(config, command_mode=BleCommandMode.LIGHT)
        )
    raise ValueError(f"unsupported action transport: {mode!r}")


__all__ = [
    "ActionTransportMode",
    "action_publisher_from_environment",
    "normalize_transport_mode",
    "transport_label",
]
