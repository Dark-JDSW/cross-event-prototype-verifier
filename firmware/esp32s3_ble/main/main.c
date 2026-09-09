/*
 * CEPV ESP32-S3 BLE endpoint.
 *
 * The command characteristic receives exactly one light opcode from the
 * desktop BLE Light bridge: 0x01 turns the light on and 0x00 turns it off.
 * The firmware validates the opcode, invokes a weak hardware callback, and
 * notifies a compact JSON acknowledgement on the status characteristic.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_log.h"
#include "nvs_flash.h"

#include "host/ble_gap.h"
#include "host/ble_gatt.h"
#include "host/ble_hs.h"
#include "host/ble_hs_mbuf.h"
#include "host/ble_store.h"
#include "host/ble_uuid.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "os/os_mbuf.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#define TAG "CEPV_BLE"
#define DEVICE_NAME "CEPV-ESP32S3"
#define MAX_COMMAND_BYTES 1

/* BLE_UUID128_INIT expects the UUID bytes in little-endian order. */
static const ble_uuid128_t service_uuid = BLE_UUID128_INIT(
    0x10, 0x6d, 0x0a, 0x4b, 0x4a, 0x8f, 0x6b, 0x9f,
    0x3e, 0x4d, 0x4d, 0x2f, 0xd0, 0xc2, 0x6a, 0x8b);
static const ble_uuid128_t command_uuid = BLE_UUID128_INIT(
    0x11, 0x6d, 0x0a, 0x4b, 0x4a, 0x8f, 0x6b, 0x9f,
    0x3e, 0x4d, 0x4d, 0x2f, 0xd0, 0xc2, 0x6a, 0x8b);
static const ble_uuid128_t status_uuid = BLE_UUID128_INIT(
    0x12, 0x6d, 0x0a, 0x4b, 0x4a, 0x8f, 0x6b, 0x9f,
    0x3e, 0x4d, 0x4d, 0x2f, 0xd0, 0xc2, 0x6a, 0x8b);

static uint16_t command_value_handle;
static uint16_t status_value_handle;
static uint16_t connection_handle = BLE_HS_CONN_HANDLE_NONE;
static bool status_notifications_enabled;
static char last_status[96] = "{\"status\":\"ready\"}";

/* Replace this weak hook with the board/application-specific light driver. */
__attribute__((weak)) void cepv_light_set(bool enabled)
{
    ESP_LOGI(TAG, "light %s (override cepv_light_set for hardware output)",
             enabled ? "ON" : "OFF");
}

/* Implemented by the ESP-IDF NimBLE port; it persists bonding metadata. */
void ble_store_config_init(void);

static int command_access(
    uint16_t conn_handle,
    uint16_t attr_handle,
    struct ble_gatt_access_ctxt *ctxt,
    void *arg);
static int status_access(
    uint16_t conn_handle,
    uint16_t attr_handle,
    struct ble_gatt_access_ctxt *ctxt,
    void *arg);
static int gap_event_handler(struct ble_gap_event *event, void *arg);
static void start_advertising(void);

static const struct ble_gatt_svc_def gatt_services[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]){
            {
                .uuid = &command_uuid.u,
                .access_cb = command_access,
                .flags = BLE_GATT_CHR_F_WRITE | BLE_GATT_CHR_F_WRITE_NO_RSP,
                .val_handle = &command_value_handle,
            },
            {
                .uuid = &status_uuid.u,
                .access_cb = status_access,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &status_value_handle,
            },
            {0},
        },
    },
    {0},
};

static void notify_status(const char *message)
{
    size_t length = strlen(message);
    if (length >= sizeof(last_status)) {
        length = sizeof(last_status) - 1;
    }
    memcpy(last_status, message, length);
    last_status[length] = '\0';

    if (!status_notifications_enabled || connection_handle == BLE_HS_CONN_HANDLE_NONE) {
        return;
    }

    struct os_mbuf *om = ble_hs_mbuf_from_flat(last_status, (uint16_t)length);
    if (om == NULL) {
        ESP_LOGE(TAG, "unable to allocate status notification buffer");
        return;
    }
    int rc = ble_gatts_notify_custom(connection_handle, status_value_handle, om);
    if (rc != 0) {
        ESP_LOGW(TAG, "status notification failed: %d", rc);
    }
}

static int command_access(
    uint16_t conn_handle,
    uint16_t attr_handle,
    struct ble_gatt_access_ctxt *ctxt,
    void *arg)
{
    (void)arg;
    if (ctxt->op != BLE_GATT_ACCESS_OP_WRITE_CHR || attr_handle != command_value_handle) {
        return BLE_ATT_ERR_UNLIKELY;
    }

    uint16_t packet_length = OS_MBUF_PKTLEN(ctxt->om);
    if (packet_length != MAX_COMMAND_BYTES) {
        return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    }

    uint8_t command[MAX_COMMAND_BYTES];
    uint16_t copied = 0;
    int rc = ble_hs_mbuf_to_flat(
        ctxt->om,
        command,
        MAX_COMMAND_BYTES,
        &copied);
    if (rc != 0 || copied != MAX_COMMAND_BYTES) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    if (command[0] != 0x00 && command[0] != 0x01) {
        ESP_LOGW(TAG, "invalid light opcode (conn=%u, value=0x%02x)",
                 conn_handle, command[0]);
        notify_status("{\"status\":\"rejected\",\"reason\":\"invalid_opcode\"}");
        return BLE_ATT_ERR_UNLIKELY;
    }

    bool enabled = command[0] == 0x01;
    ESP_LOGI(TAG, "light command received (conn=%u, value=0x%02x)",
             conn_handle, command[0]);
    cepv_light_set(enabled);
    notify_status(enabled
                      ? "{\"status\":\"executed\",\"command\":\"ON\"}"
                      : "{\"status\":\"executed\",\"command\":\"OFF\"}");
    return 0;
}

static int status_access(
    uint16_t conn_handle,
    uint16_t attr_handle,
    struct ble_gatt_access_ctxt *ctxt,
    void *arg)
{
    (void)conn_handle;
    (void)arg;
    if (ctxt->op != BLE_GATT_ACCESS_OP_READ_CHR || attr_handle != status_value_handle) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    int rc = os_mbuf_append(ctxt->om, last_status, strlen(last_status));
    return rc == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
}

static void gatt_register_cb(struct ble_gatt_register_ctxt *ctxt, void *arg)
{
    (void)arg;
    char uuid_string[BLE_UUID_STR_LEN];
    switch (ctxt->op) {
    case BLE_GATT_REGISTER_OP_SVC:
        ESP_LOGD(TAG, "registered service %s handle=%u",
                 ble_uuid_to_str(ctxt->svc.svc_def->uuid, uuid_string),
                 ctxt->svc.handle);
        break;
    case BLE_GATT_REGISTER_OP_CHR:
        ESP_LOGD(TAG, "registered characteristic %s value_handle=%u",
                 ble_uuid_to_str(ctxt->chr.chr_def->uuid, uuid_string),
                 ctxt->chr.val_handle);
        break;
    default:
        break;
    }
}

static void gatt_subscribe_cb(struct ble_gap_event *event)
{
    if (event->subscribe.attr_handle == status_value_handle) {
        status_notifications_enabled = event->subscribe.cur_notify;
        ESP_LOGI(TAG, "status notifications %s",
                 status_notifications_enabled ? "enabled" : "disabled");
    }
}

static void on_stack_reset(int reason)
{
    ESP_LOGW(TAG, "NimBLE stack reset: %d", reason);
}

static void on_stack_sync(void)
{
    start_advertising();
}

static void nimble_host_task(void *param)
{
    (void)param;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

static void start_advertising(void)
{
    uint8_t own_addr_type;
    int rc = ble_hs_id_infer_auto(0, &own_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "cannot infer BLE address type: %d", rc);
        return;
    }

    struct ble_hs_adv_fields fields = {0};
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)DEVICE_NAME;
    fields.name_len = strlen(DEVICE_NAME);
    fields.name_is_complete = 1;
    fields.uuids128 = (ble_uuid128_t *)&service_uuid;
    fields.num_uuids128 = 1;
    fields.uuids128_is_complete = 1;

    rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "cannot set advertisement fields: %d", rc);
        return;
    }

    /* Keep the 128-bit service UUID in the primary advertisement.  The
     * complete device name would make that packet exceed the legacy 31-byte
     * limit, so publish the name in the scan response instead. */
    struct ble_hs_adv_fields response_fields = {0};
    response_fields.name = (uint8_t *)DEVICE_NAME;
    response_fields.name_len = strlen(DEVICE_NAME);
    response_fields.name_is_complete = 1;
    rc = ble_gap_adv_rsp_set_fields(&response_fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "cannot set scan response fields: %d", rc);
        return;
    }

    struct ble_gap_adv_params params = {0};
    params.conn_mode = BLE_GAP_CONN_MODE_UND;
    params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    params.itvl_min = BLE_GAP_ADV_ITVL_MS(200);
    params.itvl_max = BLE_GAP_ADV_ITVL_MS(220);
    rc = ble_gap_adv_start(
        own_addr_type,
        NULL,
        BLE_HS_FOREVER,
        &params,
        gap_event_handler,
        NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "cannot start advertising: %d", rc);
        return;
    }
    ESP_LOGI(TAG, "advertising as %s", DEVICE_NAME);
}

static int gap_event_handler(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            connection_handle = event->connect.conn_handle;
            ESP_LOGI(TAG, "connected (handle=%u)", connection_handle);
        } else {
            start_advertising();
        }
        return 0;
    case BLE_GAP_EVENT_DISCONNECT:
        connection_handle = BLE_HS_CONN_HANDLE_NONE;
        status_notifications_enabled = false;
        ESP_LOGI(TAG, "disconnected (reason=%d)", event->disconnect.reason);
        start_advertising();
        return 0;
    case BLE_GAP_EVENT_ADV_COMPLETE:
        start_advertising();
        return 0;
    case BLE_GAP_EVENT_SUBSCRIBE:
        gatt_subscribe_cb(event);
        return 0;
    default:
        return 0;
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(nimble_port_init());
    ble_svc_gap_init();
    int rc = ble_svc_gap_device_name_set(DEVICE_NAME);
    if (rc != 0) {
        ESP_LOGE(TAG, "unable to set device name: %d", rc);
        nimble_port_deinit();
        return;
    }
    ble_svc_gatt_init();

    rc = ble_gatts_count_cfg(gatt_services);
    if (rc != 0) {
        ESP_LOGE(TAG, "unable to count GATT services: %d", rc);
        nimble_port_deinit();
        return;
    }
    rc = ble_gatts_add_svcs(gatt_services);
    if (rc != 0) {
        ESP_LOGE(TAG, "unable to add GATT services: %d", rc);
        nimble_port_deinit();
        return;
    }

    ble_hs_cfg.reset_cb = on_stack_reset;
    ble_hs_cfg.sync_cb = on_stack_sync;
    ble_hs_cfg.gatts_register_cb = gatt_register_cb;
    ble_hs_cfg.store_status_cb = ble_store_util_status_rr;
    ble_store_config_init();

    nimble_port_freertos_init(nimble_host_task);
}
