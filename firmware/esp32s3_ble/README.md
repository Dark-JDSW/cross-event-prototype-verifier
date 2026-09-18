# CEPV ESP32-S3 BLE endpoint

This is a standalone ESP-IDF project for an ESP32-S3 board. It uses the
ESP-IDF NimBLE host as a GATT server and exposes one custom service. The
desktop application selects this endpoint as **BLE Light**; the application
protocol is deliberately limited to a single light opcode, not a general
action-event transport.

For the complete DevKit1 wiring, security, build, discovery, troubleshooting and
acceptance checklist, see
[`ESP32S3_DEVICE_REQUIREMENTS.md`](ESP32S3_DEVICE_REQUIREMENTS.md).

| Attribute | UUID | Properties | Purpose |
| --- | --- | --- | --- |
| CEPV service | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d10` | — | Service container |
| Command | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d11` | Write / Write Without Response | One-byte light opcode |
| Status | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d12` | Read / Notify | `ready`, `accepted`, `executed`, `duplicate`, or rejection status |

The command characteristic accepts exactly one byte:

```text
0x01 = light ON
0x00 = light OFF
```

No identity, Track, camera, timestamp, action text, or JSON fields are
transmitted over BLE Light. The firmware must validate the one-byte value,
invoke `cevp_light_set(bool enabled)`, and send a status object such as
`{"status":"executed","command":"ON"}`. The sample `main.c` provides a
weak logging implementation; replace it with the DevKit1 GPIO or appliance
driver without changing the GATT contract.

## Device-side contract

The ESP32-S3 endpoint must run as a connectable NimBLE peripheral/GATT server,
advertise the service UUID above, and accept one desktop GATT client. The
sample keeps the service UUID in the primary advertisement and places the
complete `CEPV-ESP32S3` name in the scan response so the legacy 31-byte
advertisement limit is not exceeded. The
Command characteristic has this contract:

| Accepted command | Required device action |
| --- | --- |
| exactly one byte `0x01` or `0x00` | `0x01` turns the light on; `0x00` turns it off; any other length/value is rejected. |

When Status notifications are enabled, the endpoint should return a JSON
result for the same command, using `command: "ON"` or `command: "OFF"`. The
result should be one of `accepted`, `executed`, `duplicate`, or `rejected`,
with an optional `reason` for rejection. This endpoint intentionally accepts
direct unpaired connections: do not require pairing, bonding, Passkey, MITM,
or a user confirmation step. Because an unpaired endpoint cannot authenticate
the connecting host, BLE Light is intended only for low-risk light control
whose identity gate is enforced by CEPV.

## Build and flash

Install an ESP-IDF release that supports the ESP32-S3, then from this
directory run:

```powershell
idf.py set-target esp32s3
idf.py build
idf.py -p COM7 flash monitor
```

Use the actual serial port in place of `COM7`. The board advertises as
`CEPV-ESP32S3`; the host bridge uses the UUIDs above and the address configured
in `CROSS_EVENT_BLE_ADDRESS`.

The Python bridge waits for an application-level status notification by default.
Set `CROSS_EVENT_BLE_ACK_TIMEOUT` to change the wait (default 3 seconds), or
`CROSS_EVENT_BLE_REQUIRE_ACK=0` only for a deliberately fire-and-forget
deployment. In `menuconfig` and application code, keep pairing, bonding,
Passkey, MITM and automatic security-manager prompts disabled so a phone or
desktop client can connect directly without confirmation. Validate the opcode
on the device and replace the log-only handler with the device project's light
execution callback.
