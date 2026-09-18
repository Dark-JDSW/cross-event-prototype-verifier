"""Versioned interaction events, policy and execution boundaries (opt-in)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol

from .actions import ActionType
from .contexts import InteractionContext, _freeze

INTERACTION_EVENT_PROTOCOL = "cevp.interaction.v1"


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, kw_only=True)
class InteractionEvent:
    context: InteractionContext = field(default_factory=InteractionContext)

    def to_dict(self) -> dict[str, object]:
        """Explicit V1 envelope, separate from the unchanged action payload."""
        from dataclasses import fields
        from .smart_home import event_payload
        import json

        context = {f.name: _plain(getattr(self.context, f.name))
                   for f in fields(self.context) if f.name != "action"}
        return {"protocol": INTERACTION_EVENT_PROTOCOL, "context": context,
                "action": json.loads(event_payload(self.context.action))
                if self.context.action is not None else None}


@dataclass(frozen=True, kw_only=True)
class InteractionCommand:
    device_id: str | None = None
    operation: str | None = None
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _freeze(self.parameters))


class InteractionPolicy(Protocol):
    def evaluate(self, event: InteractionEvent) -> tuple[InteractionCommand, ...]:
        ...


class CommandExecutor(Protocol):
    def execute(self, command: InteractionCommand) -> None:
        """Device effect only; runtime clients stay inside implementations."""
        ...


@dataclass(frozen=True, kw_only=True)
class ContextRule:
    """Minimal opt-in conjunction rule; incomplete/error context emits nothing.

    This illustrative policy owns no clients and performs no execution. New
    policies can implement InteractionPolicy without editing this module.
    """

    identity_id: str | None = None
    action: ActionType | None = None
    zone_id: str | None = None
    device_id: str | None = None
    operation: str | None = None

    def evaluate(self, event: InteractionEvent) -> tuple[InteractionCommand, ...]:
        c = event.context
        if (c.errors or c.action is None or not self.identity_id or self.action is None
                or not self.zone_id or not self.device_id or not self.operation
                or c.identity_id != self.identity_id or c.action.identity_id != c.identity_id
                or c.action.action != self.action or c.zone_id != self.zone_id
                or c.nearest_device_id != self.device_id):
            return ()
        return (InteractionCommand(device_id=self.device_id, operation=self.operation),)
