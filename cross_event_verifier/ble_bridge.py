"""BLE GATT output for identity-authorized action events.

The vision pipeline remains synchronous, while Bleak exposes an asyncio API.
``BleActionPublisher`` owns one small event-loop thread and turns each
identity-authorized ``ActionEvent`` into a one-byte light opcode on the
configured command characteristic. The ESP32 firmware in
``firmware/esp32s3_ble`` implements the matching GATT service. A legacy JSON
encoder remains available only for explicit custom ``BleConfig`` callers; the
application factory, environment configuration, CLI, GUI, and included
firmware use BLE Light exclusively.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import os
import threading
from typing import Any, Callable

from .actions import ActionType
from .smart_home import event_id, event_payload
from .transient_actions import ActionEvent


DEFAULT_SERVICE_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d10"
DEFAULT_COMMAND_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d11"
DEFAULT_STATUS_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d12"
LIGHT_OFF_OPCODE = b"\x00"
LIGHT_ON_OPCODE = b"\x01"


class BleCommandMode(str, Enum):
    """Low-level payload contract; only ``LIGHT`` is an app output mode.

    ``EVENT_JSON`` is retained solely for callers that explicitly construct a
    legacy/custom endpoint.  It is not read from configuration, exposed by
    either GUI, or accepted as a CLI transport choice.
    """

    EVENT_JSON = "event_json"
    LIGHT = "light"


@dataclass(frozen=True)
class BleConfig:
    """Connection and GATT identifiers for the ESP32-S3 endpoint."""

    address: str
    service_uuid: str = DEFAULT_SERVICE_UUID
    command_uuid: str = DEFAULT_COMMAND_UUID
    status_uuid: str | None = DEFAULT_STATUS_UUID
    connect_timeout: float = 10.0
    write_timeout: float = 5.0
    max_payload_bytes: int = 512
    ack_timeout: float = 3.0
    require_application_ack: bool = True
    # Keep the explicit field for library compatibility; environment-created
    # configurations and the application transport factory always use LIGHT.
    command_mode: BleCommandMode | str = BleCommandMode.LIGHT
    light_on_action: ActionType | str = ActionType.HAND_UP
    light_off_action: ActionType | str = ActionType.WAVE

    def __post_init__(self) -> None:
        if not self.address.strip():
            raise ValueError("BLE address cannot be blank")
        for name in ("service_uuid", "command_uuid"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be blank")
        if self.status_uuid is not None and not self.status_uuid.strip():
            raise ValueError("status_uuid cannot be blank when provided")
        if float(self.connect_timeout) <= 0 or float(self.write_timeout) <= 0:
            raise ValueError("BLE timeouts must be positive")
        if float(self.ack_timeout) <= 0:
            raise ValueError("ack_timeout must be positive")
        if self.require_application_ack and self.status_uuid is None:
            raise ValueError(
                "status_uuid is required when application acknowledgements are enabled"
            )
        if int(self.max_payload_bytes) < 1:
            raise ValueError("max_payload_bytes must be positive")
        raw_command_mode = (
            self.command_mode.value
            if isinstance(self.command_mode, BleCommandMode)
            else str(self.command_mode)
        )
        try:
            command_mode = BleCommandMode(raw_command_mode.strip().lower())
        except ValueError as error:
            raise ValueError(
                "command_mode must be 'event_json' or 'light'"
            ) from error
        object.__setattr__(self, "command_mode", command_mode)
        on_action = ActionType.parse(self.light_on_action)
        off_action = ActionType.parse(self.light_off_action)
        if command_mode is BleCommandMode.LIGHT:
            if on_action in {ActionType.UNKNOWN, ActionType.OTHER}:
                raise ValueError("light_on_action must be a known action")
            if off_action in {ActionType.UNKNOWN, ActionType.OTHER}:
                raise ValueError("light_off_action must be a known action")
            if on_action is off_action:
                raise ValueError("light_on_action and light_off_action must differ")
        object.__setattr__(self, "light_on_action", on_action)
        object.__setattr__(self, "light_off_action", off_action)

    @classmethod
    def from_environment(cls) -> "BleConfig | None":
        """Read an optional BLE endpoint from ``CROSS_EVENT_BLE_*`` variables."""

        address = os.getenv("CROSS_EVENT_BLE_ADDRESS", "").strip()
        if not address:
            return None
        return cls(
            address=address,
            service_uuid=os.getenv(
                "CROSS_EVENT_BLE_SERVICE_UUID", DEFAULT_SERVICE_UUID
            ),
            command_uuid=os.getenv(
                "CROSS_EVENT_BLE_COMMAND_UUID", DEFAULT_COMMAND_UUID
            ),
            status_uuid=os.getenv("CROSS_EVENT_BLE_STATUS_UUID", DEFAULT_STATUS_UUID)
            or None,
            connect_timeout=float(os.getenv("CROSS_EVENT_BLE_CONNECT_TIMEOUT", "10")),
            write_timeout=float(os.getenv("CROSS_EVENT_BLE_WRITE_TIMEOUT", "5")),
            ack_timeout=float(os.getenv("CROSS_EVENT_BLE_ACK_TIMEOUT", "3")),
            require_application_ack=os.getenv(
                "CROSS_EVENT_BLE_REQUIRE_ACK", "1"
            ).strip().lower()
            not in {"0", "false", "no", "off"},
            max_payload_bytes=int(
                os.getenv("CROSS_EVENT_BLE_MAX_PAYLOAD_BYTES", "512")
            ),
            # The application endpoint is always the light-only contract.  A
            # caller that needs the legacy JSON encoder must construct an
            # explicit ``BleConfig(command_mode=BleCommandMode.EVENT_JSON)``;
            # no environment variable or GUI/CLI option exposes that mode.
            command_mode=BleCommandMode.LIGHT,
            light_on_action=os.getenv(
                "CROSS_EVENT_BLE_LIGHT_ON_ACTION", ActionType.HAND_UP.value
            ),
            light_off_action=os.getenv(
                "CROSS_EVENT_BLE_LIGHT_OFF_ACTION", ActionType.WAVE.value
            ),
        )


class BleBridgeError(RuntimeError):
    """Raised when the configured BLE endpoint cannot accept an event."""


BleakClientFactory = Callable[..., Any]


class BleActionPublisher:
    """Send action events to an ESP32 GATT server through Bleak.

    The BLE client is created lazily on the first event.  A dedicated asyncio
    loop keeps the connection reusable without making the frame worker itself
    host an event loop.
    """

    def __init__(
        self,
        config: BleConfig,
        *,
        client_factory: BleakClientFactory | None = None,
    ) -> None:
        if client_factory is None:
            try:
                from bleak import BleakClient
            except ImportError as error:
                raise RuntimeError(
                    "BLE is configured but bleak is not installed; "
                    "install the ble dependency"
                ) from error
            client_factory = BleakClient
        self.config = config
        self._client_factory = client_factory
        self._client: Any | None = None
        self._last_status: bytes | None = None
        self._pending_event_id: str | None = None
        self._pending_command: str | None = None
        self._pending_ack: asyncio.Event | None = None
        self._pending_ack_payload: dict[str, Any] | None = None
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="cross-event-ble",
            daemon=True,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=2.0):
            self._closed = True
            raise BleBridgeError("BLE event loop did not start")

    @classmethod
    def from_environment(cls) -> "BleActionPublisher | None":
        config = BleConfig.from_environment()
        return cls(config) if config is not None else None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()

    def _submit(self, coroutine: Any, timeout: float) -> Any:
        if self._closed:
            raise BleBridgeError("BLE publisher is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout)
        except TimeoutError as error:
            future.cancel()
            raise BleBridgeError("BLE operation timed out") from error
        except Exception as error:
            raise BleBridgeError(str(error)) from error

    async def _ensure_connected(self) -> Any:
        client = self._client
        if client is not None and bool(getattr(client, "is_connected", False)):
            return client
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                pass
        client = self._client_factory(
            self.config.address,
            timeout=float(self.config.connect_timeout),
        )
        try:
            await asyncio.wait_for(
                client.connect(),
                timeout=float(self.config.connect_timeout),
            )
            if not bool(getattr(client, "is_connected", True)):
                raise BleBridgeError("BLE client did not report a connected state")
            services = getattr(client, "services", None)
            get_service = getattr(services, "get_service", None)
            if get_service is not None and get_service(self.config.service_uuid) is None:
                raise BleBridgeError(
                    f"BLE service not found: {self.config.service_uuid}"
                )
            if self.config.status_uuid is not None and hasattr(client, "start_notify"):
                await client.start_notify(self.config.status_uuid, self._on_status)
        except Exception:
            try:
                await client.disconnect()
            except Exception:
                pass
            raise
        self._client = client
        return client

    def _on_status(self, _characteristic: Any, data: bytearray | bytes) -> None:
        raw = bytes(data)
        self._last_status = raw
        if self._pending_ack is None:
            return
        try:
            status = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(status, dict):
            return
        status_event_id = status.get("event_id")
        # Older firmware acknowledged without an event_id.  Accept that form
        # only while a write is pending; new firmware should always echo it.
        if self._pending_event_id is not None and status_event_id not in (
            None,
            self._pending_event_id,
        ):
            return
        status_command = status.get("command")
        if self._pending_command is not None and status_command not in (
            None,
            self._pending_command,
        ):
            return
        self._pending_ack_payload = status
        self._pending_ack.set()

    @property
    def last_status(self) -> bytes | None:
        """Most recent status notification received from the endpoint."""

        return self._last_status

    async def _write(
        self,
        payload: bytes,
        *,
        event_id: str | None,
        command: str | None = None,
    ) -> None:
        client = await self._ensure_connected()
        self._pending_event_id = event_id
        self._pending_command = command
        self._pending_ack = asyncio.Event()
        self._pending_ack_payload = None
        try:
            await asyncio.wait_for(
                client.write_gatt_char(
                    self.config.command_uuid,
                    payload,
                    response=True,
                ),
                timeout=float(self.config.write_timeout),
            )
            if self.config.require_application_ack:
                try:
                    await asyncio.wait_for(
                        self._pending_ack.wait(),
                        timeout=float(self.config.ack_timeout),
                    )
                except TimeoutError as error:
                    raise BleBridgeError(
                        f"BLE application acknowledgement timed out for {event_id}"
                    ) from error
                status = self._pending_ack_payload or {}
                result = str(status.get("status", "")).strip().lower()
                if result not in {"accepted", "executed", "duplicate"}:
                    reason = status.get("reason") or result or "unknown status"
                    raise BleBridgeError(
                        f"BLE endpoint rejected event {event_id}: {reason}"
                    )
        except Exception:
            self._client = None
            try:
                await client.disconnect()
            except Exception:
                pass
            raise
        finally:
            self._pending_event_id = None
            self._pending_command = None
            self._pending_ack = None
            self._pending_ack_payload = None

    def publish(self, event: ActionEvent) -> None:
        if self._closed:
            raise BleBridgeError("BLE publisher is closed")
        command_name: str | None = None
        if self.config.command_mode is BleCommandMode.LIGHT:
            if event.action is self.config.light_on_action:
                payload = LIGHT_ON_OPCODE
                command_name = "ON"
            elif event.action is self.config.light_off_action:
                payload = LIGHT_OFF_OPCODE
                command_name = "OFF"
            else:
                # The identity-authorized event remains visible to the caller,
                # but unrelated actions must never reach the light endpoint.
                return
            correlation_id: str | None = None
        else:
            payload = event_payload(event).encode("utf-8")
            correlation_id = event_id(event)
        if len(payload) > int(self.config.max_payload_bytes):
            raise BleBridgeError(
                f"BLE payload is {len(payload)} bytes; limit is "
                f"{self.config.max_payload_bytes}"
            )
        self._submit(
            self._write(
                payload,
                event_id=correlation_id,
                command=command_name,
            ),
            float(
                self.config.connect_timeout
                + self.config.write_timeout
                + (self.config.ack_timeout if self.config.require_application_ack else 0)
            ),
        )

    async def _disconnect(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._submit(self._disconnect(), float(self.config.write_timeout))
        finally:
            self._closed = True
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2.0)


__all__ = [
    "BleActionPublisher",
    "BleBridgeError",
    "BleCommandMode",
    "BleConfig",
    "DEFAULT_COMMAND_UUID",
    "DEFAULT_SERVICE_UUID",
    "DEFAULT_STATUS_UUID",
    "LIGHT_OFF_OPCODE",
    "LIGHT_ON_OPCODE",
]
