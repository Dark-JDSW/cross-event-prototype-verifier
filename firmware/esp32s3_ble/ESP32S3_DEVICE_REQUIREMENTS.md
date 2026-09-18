# ESP32-S3 DevKit1 设备端要求

本文档定义 ESP32-S3 DevKit1 与 Cross-Event Prototype Verifier（CEPV）
之间的 BLE Light 软件和协议契约。文档只描述 ESP-IDF/NimBLE、广播发现、
GATT、单字节命令、手机/桌面端连接和验收要求；不要求填写 ESP32-S3 的引脚、
接线、供电、负载、电气参数或其他硬件资料。

设备端必须满足两个连接目标：

1. Android 或 iOS 手机能够扫描发现该设备和 CEPV Service UUID；
2. 手机和 CEPV 桌面端都能在不配对、不绑定、不输入 Passkey、无需用户确认
   的情况下直接建立 BLE GATT 连接并通信。

本文把这种连接约定称为“Just Works/无配对直连”：设备不得主动发起或要求
配对流程，不保存 bonding 信息，也不弹出数字比较、Passkey 或确认提示。严格
来说，Bluetooth 规范中的 Just Works 常被归入配对关联方法；本项目采用的是
更窄的工程约束，即连接阶段完全不触发配对和绑定。

## 1. 范围与非目标

设备端必须实现一个可连接的 BLE GATT 外设（peripheral/server），而不是只
监听某个固定无线信道的广播接收器。BLE 的射频信道由控制器在连接建立后按
链路参数跳频选择，应用层不能指定“持续监听某个 BLE 信道”。广播只用于让
手机或桌面端发现设备和服务；可靠命令必须通过 GATT 连接写入 Command 特征。

本协议覆盖：

- ESP32-S3 DevKit1 的 ESP-IDF NimBLE GATT 服务端；
- 可被手机主动扫描发现的 connectable BLE 广播；
- 手机和桌面端的无配对、无绑定直接 GATT 连接；
- `0x01` 开灯和 `0x00` 关灯两个命令；
- Status 特征的读取和通知回执；
- 桌面端 Bleak 客户端的发现、写入、回执等待和断线重连；
- 设备端将命令交给现有灯光执行接口的抽象回调。

本协议不覆盖：

- ESP32-S3 引脚、接线、供电、灯具驱动和电气安全资料；
- 米家账号登录、米家设备配网或 Xiaomi 私有协议；
- MQTT broker 和 Home Assistant 自动化；
- 通过 BLE 广播包承载动作指令；
- 通过 BLE Command 特征发送身份 ID、Track ID、摄像头 ID、坐标、时间戳或
  JSON 动作事件；
- 让 ESP32 反向参与 CEPV 身份判定。

MQTT 是 BLE Light 的替代输出通道。选择 BLE Light 时，只有配置为开灯/关灯
动作的 `ActionEvent` 会产生 BLE 写入；例如默认 `HAND_UP` 写入 `0x01`，
`WAVE` 写入 `0x00`，`SQUAT`、`SIT_DOWN` 和 `STAND_UP` 默认不会写入灯光端点。

## 2. ESP-IDF 与 NimBLE 软件要求

### 2.1 工程和目标

设备端必须使用支持 ESP32-S3 的 Espressif ESP-IDF，并使用其 NimBLE host。
工程目标必须设置为 `esp32s3`，不能把其他芯片 target 的固件烧录到 DevKit1。
仓库中的独立工程目录为 `firmware/esp32s3_ble`，入口为
`firmware/esp32s3_ble/main/main.c`。

工程至少需要以下软件组件：

- `bt`：NimBLE GAP/GATT host 和 controller 接口；
- `nvs_flash`：ESP-IDF 蓝牙栈初始化所需的 NVS 支持；
- ESP-IDF 默认的 FreeRTOS、日志和系统组件；
- 如设备项目另有灯光执行模块，由设备项目自行加入相应组件，本协议不规定
  该模块的硬件实现。

必须具备 `idf.py`、CMake、Ninja 和 ESP32-S3 对应工具链。ESP-IDF 版本、
工程版本和生成的 `sdkconfig` 应作为设备交付资料记录下来。

### 2.2 NimBLE 角色

| 功能 | 要求 | 说明 |
| --- | --- | --- |
| Bluetooth Controller | 启用 | 提供 ESP32-S3 BLE 射频和链路 |
| NimBLE host | 启用 | 使用 ESP-IDF NimBLE |
| Peripheral role | 启用 | 设备主动广播并接受手机/桌面端连接 |
| Central role | 不需要 | 固件不扫描其他 BLE 设备 |
| GATT server | 启用 | 提供 CEPV Service、Command、Status |
| GATT client | 不需要 | 手机或桌面端才是 GATT client |
| GAP/GATT service | 启用 | 支持设备名和标准服务发现 |
| Bond/NVS 持久化 | 不用于本协议 | 不保存配对或绑定密钥 |

设备可以使用 NVS 保存其他应用配置，但不得把 BLE bonding 当作本协议的前置
条件，也不得在每次启动时等待或恢复绑定关系后才开始广播。

### 2.3 无配对直连策略

设备端必须采用以下连接策略：

- 不主动调用配对发起接口；
- 不要求手机或桌面端调用系统配对 API；
- 不启用 Passkey、数字比较、MITM 或用户确认流程；
- 不要求 bonding，不保存或校验 bond 记录；
- 不设置“只有已绑定主机才能连接”的白名单规则；
- Command access callback 不得因为连接未加密/未认证而拒绝本协议命令；
- 连接成功后立即允许通过 GATT 发现服务和写入合法单字节命令。

手机系统可能第一次使用应用时显示“允许应用使用蓝牙”的系统权限提示，
这是操作系统权限，不是 BLE 配对确认。设备端不得显示或等待 Passkey、数字
比较或“是否配对”的确认对话框。如果手机弹出系统蓝牙配对窗口，通常说明
刷入了旧固件、调用了错误的安全管理 API 或启用了 bonding，应先修正固件。

这种无配对设计不提供经过认证的主机身份，也不能抵御附近 BLE 客户端发送
合法的 `0x00`/`0x01`。因此 BLE Light 只适合低风险、已由 CEPV 身份门控的
灯光控制；高风险设备不得把“无配对”当作访问控制。

## 3. 广播和手机发现要求

### 3.1 广播用途

固件必须持续进行可连接的普通 BLE 广播（connectable undirected advertising）。
广播至少用于：

- 告知手机和桌面端设备可以被发现；
- 携带 CEPV Service UUID，供扫描器按服务过滤；
- 通过 scan response 提供完整设备名。

广播包不得携带开关灯命令，也不得把身份、Track、动作 JSON 或坐标放进
Manufacturer Data、Service Data 或设备名。广播无确认、可能丢包且会被附近
设备看到，不能替代 GATT 写入和 Status 回执。

默认设备名为 `CEPV-ESP32S3`。当前工程将 128 位 Service UUID 放在主广播，
将完整设备名放在 scan response，以避免传统 BLE 广播 31 字节限制。手机扫描
器应启用 active scan；扫描结果可能先没有名称，但仍应能按 Service UUID 找到
设备。

### 3.2 手机端必须支持的发现流程

Android 和 iOS 应用或 GATT 调试工具必须能够：

1. 请求本机蓝牙扫描权限（这是手机应用权限，不是设备配对）；
2. 按 Service UUID 过滤扫描结果，名称可作为辅助显示字段；
3. 发现 `CEPV-ESP32S3` 后记录当前系统返回的设备标识；
4. 直接发起 GATT 连接，不调用系统配对、创建 bond 或 Passkey 流程；
5. 发现 Service、Command 和 Status 特征；
6. 订阅 Status notify（如果应用需要回执）；
7. 写入 `0x01`/`0x00` 并读取或接收结果。

手机端不能把“扫描到设备”误认为“已经连接”，也不能只按设备名连接而不
验证 Service UUID。Windows、Android 和 iOS 对设备地址的显示方式可能不同，
应用应使用本平台扫描 API 返回的标识，不应跨平台硬编码一个 MAC 地址。

### 3.3 Android 连接约束

Android 客户端应使用 `BluetoothLeScanner` 配合 Service UUID 扫描过滤器，
发现目标后停止扫描，再使用 `connectGatt` 建立 GATT 连接。Android 12 及以上
应申请并获得 `BLUETOOTH_SCAN`、`BLUETOOTH_CONNECT` 等运行时权限；较旧版本
按目标 SDK 和系统版本处理蓝牙/位置权限。权限允许后，应用直接连接即可：

- 不调用 `BluetoothDevice.createBond()`；
- 不跳转系统“配对新设备”页面；
- 不等待或显示 Passkey；
- 在 `onConnectionStateChange` 成功后调用 `discoverServices()`；
- 在服务发现成功后按 UUID 获取 Command/Status 特征。

可用 nRF Connect、LightBlue 等工具验证时，也应选择“Connect”而不是“Pair”，
并确认工具没有自动建立系统 bond。

### 3.4 iOS 连接约束

iOS 客户端应使用 CoreBluetooth 的 `CBCentralManager`：

1. `scanForPeripherals(withServices:)` 传入 CEPV Service UUID；
2. 在发现回调中保存 `CBPeripheral`，停止扫描；
3. 调用 `connect(_:options:)`，不调用任何配对/绑定 API；
4. 在连接成功后 `discoverServices`；
5. 再按 UUID `discoverCharacteristics`；
6. 对 Status 调用 `setNotifyValue(true, for:)`，对 Command 写入原始字节。

iOS 可能显示应用首次使用蓝牙的系统授权提示，但本协议不要求用户在设备
层面确认配对。若系统要求输入 Passkey 或显示配对请求，应停止测试并检查
固件是否错误启用了安全管理流程。

## 4. 连接数和连接生命周期

当前示例固件维护一个 `connection_handle` 和一个 Status 通知订阅状态，设备端
契约按“一个 CEPV 控制客户端同时连接一个端点”设计：

- 同一时刻建议只让一个手机或桌面端写入 Command；
- 手机测试完成后断开，再让桌面端连接；
- `BLE_GAP_EVENT_CONNECT` 成功后保存连接句柄；
- `BLE_GAP_EVENT_DISCONNECT` 时清除句柄和通知订阅并重新开始广播；
- 广播停止或重新启动不得要求重新配对；
- 未实现多客户端仲裁前，不要让多个进程同时写入同一端点；
- 如果未来支持多客户端，必须为每个连接独立维护句柄、订阅状态和命令顺序。

BLE 控制器自动处理连接参数、信道跳频、链路层重传和确认；应用代码不能
自行选择、锁定或监听某个固定 BLE 信道。

## 5. 固定 GATT 服务契约

### 5.1 UUID

所有 UUID 必须与 CEPV 桌面端默认配置一致：

| 属性 | UUID | 类型 | 用途 |
| --- | --- | --- | --- |
| CEPV Light Service | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d10` | Primary service | 灯光控制服务容器 |
| Command | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d11` | Characteristic | 接收单字节开关灯命令 |
| Status | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d12` | Characteristic | 读取最近结果并通知状态 |

桌面端和手机端使用标准 UUID 字符串；NimBLE 的 `BLE_UUID128_INIT` 按小端
字节序初始化，不能把 UUID 字符串从左到右直接当作 C 数组字节。

### 5.2 Service

- 类型必须是 primary service；
- UUID 必须是上表 Service UUID；
- 必须在 `ble_gatts_count_cfg()` 和 `ble_gatts_add_svcs()` 成功后开始广播；
- 手机和桌面端服务发现失败时，必须把端点视为不可用，不能向未知特征写入。

### 5.3 Command 特征

属性：

- `Write`：支持带响应写入；CEPV Bleak 客户端默认使用 `response=True`；
- `Write Without Response`：可保留，便于手机工具低延迟测试；
- 不需要 Read、Notify 或 Indicate；
- access callback 只接受 `BLE_GATT_ACCESS_OP_WRITE_CHR`。

长度和取值必须严格校验：

| 写入字节序列 | 含义 | 处理 |
| --- | --- | --- |
| `01` | 开灯 | 调用灯光执行接口，随后回送 ON 状态 |
| `00` | 关灯 | 调用灯光执行接口，随后回送 OFF 状态 |
| 空数据 | 无效 | 拒绝，不执行 |
| 多于 1 字节 | 无效 | 拒绝，不截断、不拼接 |
| 单字节且不是 `00`/`01` | 无效 | 拒绝，不近似解释 |

`01` 是一个原始字节，不是 ASCII 文本 `"01"`，也不是两个字符 `0x30 0x31`。
Command 特征绝不能接受 UTF-8 文本、JSON、身份字段、Track 字段或动作名称。

### 5.4 Status 特征

属性：

- `Read`：返回最近一次状态文本；
- `Notify`：手机或桌面端订阅后，在命令处理完成时通知；
- 主机不向 Status 写入任何内容；
- Status JSON 仅用于结果回执，不是新的控制命令。

最小有效状态示例：

```json
{"status":"executed","command":"ON"}
```

允许的状态值：

| `status` | 使用时机 | CEPV 是否视为成功 |
| --- | --- | --- |
| `ready` | 服务启动后的初始状态 | 不是某条命令的成功回执 |
| `accepted` | 已通过长度、取值和执行条件检查 | 是 |
| `executed` | 已完成灯光执行接口调用 | 是 |
| `duplicate` | 相同状态已生效，按幂等策略不重复执行 | 是 |
| `rejected` | 长度、取值或应用条件不满足 | 否 |

有效回执的 `command` 必须是 `ON` 或 `OFF`，与 `0x01`/`0x00` 对应。拒绝时
可增加稳定的 `reason`，例如 `invalid_opcode`、`invalid_length` 或
`not_authorized`。回执不得包含身份、Track、摄像头、坐标或敏感信息。

当前仓库样例固件对有效命令直接通知 `executed`，对无效 opcode 通知
`rejected`；CEPV 桌面端接受 `accepted`、`executed` 和 `duplicate`。

## 6. 固件运行行为

### 6.1 启动顺序

设备启动时必须按以下顺序运行：

1. 初始化 ESP-IDF 运行环境和必要的 NVS；
2. 初始化 NimBLE host/controller；
3. 初始化 GAP/GATT 标准服务；
4. 注册 CEPV Service、Command 和 Status；
5. 配置广播字段和 scan response；
6. 完成 NimBLE sync 后开始可连接广播；
7. 进入 host task，持续处理手机/桌面端连接和 GATT 事件。

启动广播前如果服务注册失败，设备必须记录明确错误并停止宣称“可用”，
不能广播一个没有 CEPV Service 的假端点。

### 6.2 Command callback 顺序

Command access callback 必须执行：

1. 确认操作类型为写 Command 特征；
2. 取得 mbuf 的完整长度；
3. 长度必须等于 1；
4. 复制并验证字节只能是 `0x00` 或 `0x01`；
5. 不执行任何配对、绑定、Passkey 或加密等待；
6. 调用设备项目提供的灯光执行接口；
7. 更新最近状态并通知 Status；
8. 返回正确的 ATT 状态码。

失败步骤必须在灯光执行接口之前结束。不得截断较长 payload，也不得在 GATT
回调中阻塞等待网络、MQTT、文件系统或长时间动画；耗时任务应复制命令后交
给应用任务，并保持回执顺序可解释。

### 6.3 无配对连接的 GAP 行为

- 连接成功后直接允许服务发现；
- 收到安全管理请求时不得主动发起或要求配对；
- 不保存 bond key，不检查“是否已绑定”；
- 不因 `encrypted=false` 或 `authenticated=false` 拒绝合法的 `00`/`01`；
- 手机/桌面端断开后立即恢复广播；
- 重启后不等待旧主机恢复安全关系即可广播和接受新连接。

如果设备项目自身存在通用安全管理模板，必须把它从本 BLE Light 端点移除或
关闭。不能一边在文档中声明无配对直连，一边在固件中调用自动配对 API。

### 6.4 灯光执行接口

仓库样例提供以下抽象回调，默认实现只记录日志：

```c
void cepv_light_set(bool enabled);
```

设备项目可以用自己的执行模块覆盖该回调，但本协议不要求在 CEPV 文档中
填写具体引脚、接线或电气资料。回调必须满足：

- `true` 表示执行 ON，`false` 表示执行 OFF；
- 返回或记录可诊断的执行结果；
- 不把身份或动作文本写入 BLE Command；
- 执行失败时回送 `rejected` 或约定的失败状态，而不是伪造 `executed`；
- 断线、重启和重复命令的行为由设备项目明确规定并测试。

## 7. CEPV 桌面端对接

### 7.1 安装和配置

在 CEPV 虚拟环境中安装：

```powershell
python -m pip install -e ".[ble]"
```

启动 GUI 的同一个 PowerShell 进程中设置：

```powershell
$env:CROSS_EVENT_TRANSPORT = "ble-light"
$env:CROSS_EVENT_BLE_ADDRESS = "当前平台扫描到的设备地址"
```

默认 UUID 已与本协议一致。只有更换为兼容自定义固件时才覆盖：

```powershell
$env:CROSS_EVENT_BLE_SERVICE_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d10"
$env:CROSS_EVENT_BLE_COMMAND_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d11"
$env:CROSS_EVENT_BLE_STATUS_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d12"
```

可调参数：

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `CROSS_EVENT_BLE_CONNECT_TIMEOUT` | `10` 秒 | GATT 连接超时 |
| `CROSS_EVENT_BLE_WRITE_TIMEOUT` | `5` 秒 | Command 写入超时 |
| `CROSS_EVENT_BLE_ACK_TIMEOUT` | `3` 秒 | Status 通知等待超时 |
| `CROSS_EVENT_BLE_REQUIRE_ACK` | `1` | 是否要求应用层回执 |
| `CROSS_EVENT_BLE_LIGHT_ON_ACTION` | `HAND_UP` | 映射到 `0x01` 的动作 |
| `CROSS_EVENT_BLE_LIGHT_OFF_ACTION` | `WAVE` | 映射到 `0x00` 的动作 |

### 7.2 桌面端连接顺序

1. GUI 校验地址和配置，未配置地址显示“未配置”；
2. 只有已确认身份完成动作门控的 `ActionEvent` 才进入 BLE publisher；
3. 首次发布时创建 Bleak client 并连接设备；
4. 发现 Service、Command 和 Status UUID；
5. 对 Status 启用 notify；
6. 使用带响应写入向 Command 写入一个原始字节；
7. 等待 `accepted`、`executed` 或 `duplicate`；
8. 失败、超时或服务缺失显示为传输错误，但不停止视觉跟踪；
9. 连接失败时释放客户端，下一条已授权动作再重新连接。

桌面端不调用配对、绑定、Passkey 或系统蓝牙设置页面。`BLE Light` 的命令
载荷始终只有：

```text
HAND_UP（默认开灯） -> 01
WAVE（默认关灯）    -> 00
```

完整 `ActionEvent` 的 `identity_id`、`track_id`、`camera_id`、`timestamp`、
`confidence` 和动作文本只存在于 CEPV 内存事件或 MQTT JSON 中，不经 BLE
Command 发送。

### 7.3 手动动作门控

GUI 允许操作者从当前 Track 中选择目标，再选择 `SQUAT`、`SIT_DOWN`、
`STAND_UP`、`HAND_UP` 或 `WAVE`。请求在工作线程的帧边界排队；只有下一帧
仍由同一 Track 映射到同一 `KNOWN` 身份时，才生成事件并进入发布链路。

Track 消失、身份切换、ID switch 或请求超过 TTL 时，命令会被丢弃，不会发给
ESP32。设备端不能把收到 `0x01` 理解为“某个指定身份已经通过认证”；身份
门控已经在桌面端完成。

## 8. 构建、烧录和初次启动

在已加载 ESP-IDF 环境的 PowerShell 中：

```powershell
cd C:\Users\wzige\PycharmProjects\cross-event-prototype-verifier\firmware\esp32s3_ble
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p COM7 flash monitor
```

`COM7` 仅为命令格式示例，应替换为设备管理器实际端口。`menuconfig` 中重点
核对 Bluetooth/NimBLE、Peripheral、GATT server 和广播配置；确认没有启用
会主动发起配对、bonding、MITM、Passkey 或确认交互的代码/选项。不要把
“Security Manager 默认存在”误认为“本协议需要安全配对”。

串口日志至少应显示：

- NimBLE host 初始化成功；
- CEPV Service/Characteristic 注册成功；
- `advertising as CEPV-ESP32S3`；
- 手机或桌面端连接成功；
- Status notify 订阅成功（如果客户端订阅）；
- 有效 `0x00`/`0x01` 命令和 ON/OFF 处理结果。

按 EN/RESET 后，设备应在合理时间内重新广播并允许新的手机连接；不应因为
没有旧 bond、没有配对确认或没有 Passkey 而停止工作。无配对协议不需要为
“清除 bonding”执行擦除操作。

## 9. 手机和桌面端联调步骤

### 9.1 手机发现和无配对连接

1. 在手机打开蓝牙，并授予测试应用扫描/连接权限；
2. 用 nRF Connect、LightBlue 或自研 Android/iOS 应用按 Service UUID 扫描；
3. 找到 `CEPV-ESP32S3` 或同 Service UUID 的设备；
4. 直接选择 Connect/GATT Connect，不选择 Pair、Bond 或系统配对；
5. 发现三个 UUID，订阅 Status；
6. 写入原始单字节 `01`，确认收到 ON 回执；
7. 写入原始单字节 `00`，确认收到 OFF 回执；
8. 断开手机，再确认设备仍可被扫描和连接；
9. 观察整个流程没有 Passkey、数字比较或配对确认弹窗。

### 9.2 桌面端验证

1. 运行 `BleakScanner.discover()`，记录当前平台返回的设备标识；
2. 设置 `CROSS_EVENT_TRANSPORT=ble-light` 和地址；
3. 启动 CEPV GUI，确认下拉框只有 MQTT、BLE Light；
4. 选择一个已确认身份和映射动作；
5. 确认 BLE 写入只有一个字节，而不是 JSON 或动作文本；
6. 确认 Status 成功回执显示在 GUI 状态栏；
7. 断开设备后再次触发已授权动作，确认 bridge 能重新连接；
8. 选择未确认身份或制造 ID switch，确认不会产生 BLE 写入。

### 9.3 负向命令测试

以下写入都必须被拒绝，且不得调用灯光执行接口：

- 空 payload；
- `02`、`ff` 等未知单字节；
- `00 01` 等多字节 payload；
- ASCII 字符串 `"0"`、`"1"`、`"ON"`、`"OFF"`；
- UTF-8 JSON 或包含身份字段的文本。

拒绝后设备仍应保持连接或按协议返回 ATT 错误，不能因一条非法命令永久
停止广播。

## 10. 验收清单

### 10.1 固件和广播

- [ ] `idf.py set-target esp32s3` 成功；
- [ ] `idf.py build` 无编译/链接错误；
- [ ] 设备启动后注册固定 Service/Characteristic；
- [ ] 主广播包含 Service UUID；
- [ ] scan response 能显示 `CEPV-ESP32S3`；
- [ ] 广播可连接，不依赖固定 BLE 信道；
- [ ] 手机和桌面端都能扫描发现端点。

### 10.2 无配对连接

- [ ] Android 能直接 Connect，不调用 `createBond()`；
- [ ] iOS 能直接 Connect，不经过系统配对；
- [ ] Windows/Bleak 能直接连接，不要求系统配对；
- [ ] 全流程没有 Passkey、数字比较或用户确认；
- [ ] 固件不主动发起配对；
- [ ] 固件不保存或校验 bonding；
- [ ] 重启、断开和换手机后仍能重新扫描/连接；
- [ ] 若出现配对弹窗，已定位并修正旧固件或安全管理配置。

### 10.3 GATT 和命令

- [ ] 三个 UUID 与本文档完全一致；
- [ ] Command 支持 Write，且可选 Write Without Response；
- [ ] Status 支持 Read/Notify；
- [ ] 只有单字节 `0x01`/`0x00` 会执行；
- [ ] 非法长度和值不会执行；
- [ ] Status 的 ON/OFF 与命令一致；
- [ ] 手机和桌面端均能订阅并接收回执；
- [ ] BLE 抓包确认 Command 没有 JSON、身份或动作文本。

### 10.4 CEPV 上位机

- [ ] `bleak` 安装并能扫描设备；
- [ ] `CROSS_EVENT_TRANSPORT=ble-light` 配置正确；
- [ ] GUI 只显示 MQTT/BLE Light；
- [ ] 未确认身份、冲突或 ID switch 时不发 BLE；
- [ ] 配置的开灯动作发送 `0x01`；
- [ ] 配置的关灯动作发送 `0x00`；
- [ ] 未映射动作不会产生 BLE 写入；
- [ ] 回执成功、拒绝和超时均能显示；
- [ ] 连接失败后下一条授权动作可以重新连接。

## 11. 常见故障处理

### 11.1 手机扫描不到设备

依次检查：

1. 串口是否出现 `advertising as CEPV-ESP32S3`；
2. 手机蓝牙和应用扫描权限是否已打开；
3. 扫描器是否使用 active scan；
4. 是否按 Service UUID 过滤而不是只按名称过滤；
5. 是否已有其他客户端连接；
6. 是否刷入了目标为 `esp32s3` 的最新固件；
7. 设备是否因 host reset 或 GATT 注册错误而没有真正开始广播。

### 11.2 手机出现配对提示

本协议不需要配对提示。出现提示时：

- 取消提示，不输入 Passkey；
- 检查是否刷入旧固件；
- 检查应用是否调用了系统配对、`createBond()` 或同类 API；
- 检查固件是否启用了 bonding、MITM 或自动配对模板；
- 重新烧录无配对版本并清理手机侧旧的系统配对记录（仅清理手机设置，
  不需要为本协议在设备端保存或恢复 bond）。

### 11.3 能发现但无法建立 GATT 连接

- 确认端点是 connectable advertising，而不是不可连接广播；
- 关闭占用该端点的其他手机/桌面客户端；
- 确认手机应用在获得蓝牙权限后才调用连接；
- 检查连接回调是否在超时前触发服务发现；
- 对比固件和客户端的 Service UUID；
- 不要通过系统配对页面解决 GATT 服务发现问题。

### 11.4 能连接但服务或特征不存在

- 对比三个 UUID 的每一位；
- 确认刷入的不是旧工程或另一套 GATT 服务；
- 修改 UUID 后重新 build/flash；
- 确认 GATT 服务注册成功后才开始广播；
- 不要尝试向“第一个特征”写入来绕过 UUID 校验。

### 11.5 写入成功但没有回执

- 确认客户端已对 Status 启用 Notify；
- 确认固件在命令完成后调用通知函数；
- 确认 Status UUID 正确且连接没有中断；
- 检查串口是否出现通知失败日志；
- 仅在明确接受无回执风险时设置 `CROSS_EVENT_BLE_REQUIRE_ACK=0`；
- 不要用无限增大超时掩盖固件任务阻塞或连接问题。

## 12. 版本和变更约束

以下内容属于协议兼容面，未经同时修改固件、手机端、桌面端和测试不得更改：

- Service、Command、Status UUID；
- Command 的单字节长度；
- `0x00`/`0x01` 的 ON/OFF 语义；
- Status 的成功值和 `command` 字段含义；
- 设备名、主广播 Service UUID 和 scan response 约定；
- 无配对、无绑定、无 Passkey 的连接行为；
- 桌面端 BLE Light 的地址、连接、回执和重连行为。

如果未来需要传输更多动作或设备能力，应新建版本化服务/特征或明确的能力
协商协议，不得把 JSON 偷塞进当前 Command 特征，也不得让同一 UUID 在不同
固件中有不同字节语义。当前版本必须保持“一个字节、一个灯光状态”的边界。

## 13. 设备端交付资料和最终清单

ESP32-S3 设备端交付给 CEPV 部署人员时，至少提供：

1. ESP-IDF 版本和工程版本/commit；
2. 固件广播名称和手机端扫描方法；
3. 三个 GATT UUID（若未使用默认值）；
4. 已验证的 Command 长度和值规则；
5. Status 回执格式、状态值和超时建议；
6. 明确声明“不配对、不绑定、不使用 Passkey”；
7. Android、iOS 和 Windows/Bleak 的直连测试记录；
8. 断线、重启、重复命令和非法命令测试记录；
9. 设备项目对灯光执行接口失败时的状态处理说明。

满足上述清单后，ESP32-S3 才能作为 CEPV 的 BLE Light 端点接入。GATT 通信
验收通过只表示手机/桌面端能够无配对地发现、连接并传输单字节命令，不表示
身份系统、米家自动化或其他上层授权策略已经完成验收。
