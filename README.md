# Cross-Event Prototype Verifier
> 面向跨摄像头、跨采集事件的开放集身份验证原型，支持用户注册、身份识别、视频回放与身份状态核验。

## 项目简介

Cross-Event Prototype Verifier（CEPV）把同一个人的多次观测组织成可审计的原型，而不是把一次视频中的临时 `Track ID` 当作永久身份。系统以外观特征建立稳定的视觉身份，再用步态特征在不同时间、机位、视角和采集事件之间进行验证；无法达到开放集阈值时会保留为未知、候选或冲突状态，不强行匹配。

项目提供 Python API、命令行入口和桌面 GUI。生产视觉链路由 YOLO11L、ByteTrack、OSNet-AIN、RTMPose-S 与 GaitGraph2 组成，状态和审计信息写入本地 SQLite。短时动作识别复用 RTMPose 的 COCO-17 关键点，在 Track 级内存状态机中识别下蹲、坐下、站起、举手和挥手，并可将已确认身份的动作事件发布到 MQTT 或 BLE Light（ESP32-S3 BLE GATT 灯光端点）。

“注册用户”在本项目中指创建或确认 `P1`、`P2` 等伪匿名视觉身份，不等同于实名、账号密码或法律意义上的身份认证。姓名、权限和设备控制策略应由上层业务系统维护。

## 核心功能

| 能力 | 说明 |
| --- | --- |
| 用户注册 | 支持 OSNet-first 自动注册，也支持 GUI 选中 Track 后人工登记身份 ID；身份库保存在 SQLite。 |
| 身份识别 | 多目标检测与跟踪后，先做外观身份确认，再融合跨事件步态原型；采用一对一指派、校准阈值和 open-set 拒识。 |
| 跨事件验证 | 每次输入都有独立的 capture session/event 记录，步态样本按质量、视角和独立事件累计，避免临时 Track ID 串库。 |
| 身份状态核验 | 区分 `KNOWN`、`AMBIGUOUS`、`UNKNOWN`、`NEED_MORE_DATA`、`CONFLICT` 等决策，以及身份生命周期和步态就绪度。 |
| 摄像头与视频 | 支持摄像头实时流和 MP4、AVI、MOV、MKV、WMV 等视频文件；GUI 可设置视频重复学习次数并显示逐帧结果。 |
| 短时动作 | 复用 RTMPose 关键点和 Track 级 FSM，当前轻量检测器覆盖 `SQUAT`、`SIT_DOWN`、`STAND_UP`、`HAND_UP`、`WAVE`；动作不会反向参与身份判定。 |
| 智能家居事件 | `ActionEvent` 统一输出 JSON；控制通道仅保留 MQTT（Home Assistant 等订阅）与 BLE Light（ESP32-S3 自定义 GATT 灯光控制）。BLE Light 只发送 `0x01`/`0x00` 开关灯指令。 |
| 运行维护 | 支持模型预加载诊断、运行时参数热更新、地面控制点标定导入/导出、数据库审计和清除身份前自动备份。 |

## 工作方式

系统将“这个人是谁”和“这个人现在做什么”严格分层：

```text
摄像头/视频
    │
    ▼
YOLO11L 检测 ──► ByteTrack 生成临时 Track
    │                         │
    │                         ├─► OSNet-AIN 外观特征 ──► 视觉身份 P1/P2
    │                         │                              │
    │                         └─► RTMPose COCO-17 ───────────┤
    │                                                        ├─► GaitGraph2 步态事件/原型
    │                                                        └─► Track 级短时动作 FSM
    ▼
开放集校准、一对一指派、状态机与 SQLite 审计
    │
    ├─► 身份状态/GUI 结果
    └─► 已确认身份的 ActionEvent ──► MQTT 或 BLE Light ──► 家居自动化
```

一次观测的主要流程如下：

1. 检测器找到人体，ByteTrack 在当前来源内维护临时轨迹。轨迹号只在本次来源中有效，不能作为跨视频身份。
2. OSNet-AIN 对稳定外观窗口生成 embedding。生产 GUI 使用 OSNet-first 流程创建或重新绑定视觉身份；已有身份不会因为动作变化而改变。
3. RTMPose-S 产生 COCO-17 关键点。明确的 `WALK` 且质量足够时才进入 GaitGraph2；下蹲、坐下等非行走窗口通过 `ActionRouter` 隔离，不能写入步态库。
4. GaitGraph2 将足够长的真实帧序列变成步态 embedding。跨独立事件、机位和视角的原型经过校准后进行 open-set 匹配，并生成正式决策和状态迁移。
5. 同一份姿态流同时进入内存中的 `TransientActionManager`。动作状态按 Track、分段和 TTL 管理；只有当前决策为已知身份时，最近已完成的动作才会释放为 `ActionEvent`。
6. `ActionEvent` 可由 GUI 选择 MQTT 或 BLE Light 发布。视觉程序只发布事件，不直接保存米家设备 ID、灯光实体或 ESP32 GPIO 规则，具体自动化由 Home Assistant 或固件侧完成。操作员也可以在“手动动作触发”面板选中当前 Track/已确认身份并指定下蹲、坐下、站起、举手或挥手；该请求仍需下一帧通过 `KNOWN` 身份门控。

跨事件的关键是保存“独立证据”和模型协议，而不是简单拼接帧数。当前默认门槛包括：单个步态样本至少 25 个有效姿态帧，正式检索通常至少需要 45 个真实帧且覆盖率达到 0.75；步态暂时可用通常需要 2 个独立事件，步态就绪通常需要 3 个独立事件和至少 2 个摄像头/视角。实际阈值可在运行时参数页调整，但应以目标摄像头验证集重新校准。

## 使用场景

- **家庭身份化交互**：识别为 `P1` 的人在客厅坐下，可由 Home Assistant 切换观影场景；`P2` 可使用不同的灯光、空调或窗帘偏好。
- **无接触手势控制**：已确认身份举手或挥手，发布动作事件给 MQTT 自动化；需要更低延迟或无 Wi-Fi 的局部控制时，可发送到 ESP32-S3 BLE 端点。
- **跨机位/跨时段验证**：用不同摄像头或不同日期的视频验证同一伪匿名身份，观察视觉身份、步态学习和冲突隔离状态的变化。
- **多人物视频分析**：同一画面中的多条 Track 通过一对一指派分别核验，未知人物不会被最近邻结果强行吸收到已知身份。
- **算法与数据集研究**：使用固定的模型版本、特征协议、采集事件和数据分割，评估跨视角、服装变化、遮挡及 open-set 拒识表现。

## 快速开始

### 安装

推荐在 Windows PowerShell 中使用独立虚拟环境。项目要求 Python 3.10 或更高版本；仓库中的 `requirements.txt` 面向包含 NVIDIA CUDA 运行时的生产部署，并固定了对应的 PyTorch、Ultralytics 和 ONNX Runtime 版本。

```powershell
cd C:\Users\wzige\PycharmProjects\cross-event-prototype-verifier
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果只做无模型的 API/演示或在已有环境中按需安装，可使用项目 extras：

```powershell
python -m pip install -e ".[gui-qt,smart-home,ble]"
```

生产运行还需要模型文件。按 `models/README.md` 的文件名准备 `yolo11l.pt`、`rtmpose-s.onnx`、`osnet_ain_x1_0_dg.pth`、`gaitgraph2_grew_state.pt` 和 `bytetrack-cross-event.yaml`，然后执行：

```powershell
python -m cross_event_verifier download-models
python -m cross_event_verifier doctor
```

`download-models` 只负责按清单同步公开模型；二进制权重不提交到仓库。`doctor` 会检查 PyTorch/CUDA、ONNX Runtime provider、文件存在性和 SHA-256；必须先处理它报告的 `issues`，再进行生产识别。仅使用演示后端时不需要这些权重。

### 启动

先激活环境，再选择所需入口：

```powershell
.\.venv\Scripts\Activate.ps1

# 最小化烟雾测试，不加载生产视觉模型
python -m cross_event_verifier demo

# 生产 GUI，默认 Qt 界面
python -m cross_event_verifier gui --vision-backend production

# 指定数据库和后端；auto 会在可用时选择生产适配器
python -m cross_event_verifier gui `
  --database data/verifier-production-v1.sqlite3 `
  --vision-backend auto
```

GUI 入口默认为 PyQt6；安装失败时请补装 `python -m pip install -e ".[gui-qt]"`。代码仍保留 `launch_tk_gui` 兼容入口，但标准启动、功能展示和传输选择以 Qt GUI 为准。首次打开窗口会后台预加载模型，状态栏显示“正在后台准备生产视觉后端”，完成后才可开始采集。

## 使用方法

### 第一次使用

1. 安装依赖并运行 `doctor`，确认生产模型和推理 provider 均为 ready；没有 GPU 时先用 `demo` 或经过验证的 CPU/ONNX 配置，不要直接假设实时帧率满足要求。
2. 启动 GUI，在“输入源”选择“摄像头”并填写设备号，或选择“视频文件”并通过“浏览…”选取文件；为每个真实机位设置稳定的“来源 ID”。
3. 让人物以完整身体进入画面，保证肩、髋、膝、踝等关键点尽量不被遮挡。先完成视觉身份注册，再用不同时间或机位采集独立步态事件。
4. 在“人物注册”面板保持自动注册，或按“注册用户”步骤人工登记。观察右侧 Track 表中的身份、结果、动作和分数，确认未知/冲突目标没有被误吸收。
5. 不需要智能家居时保持事件传输为未配置即可；需要联动时，先按环境变量配置 MQTT 或 BLE Light，再停止采集后在“事件传输”下拉框选择协议。
6. 用至少两个独立事件验证同一身份，再检查“步态学习中/暂时可用/就绪”等状态。重复同一个视频只能增加同一采集会话内的观测，不应替代真实独立事件。

**连接 ESP32-S3 的完整流程（BLE GATT）**

本项目将 BLE 广播用于“让上位机发现 `CEPV-ESP32S3`”，再通过发现后的 GATT 连接写入 Command 特征。桌面应用的 `BLE Light` 通道只写入 `0x01`（开灯）或 `0x00`（关灯）；身份、Track、摄像头和时间等字段只留在应用内或 MQTT JSON 中，不经 BLE 发送。纯广播包不适合承载可靠动作指令或回执，因此广播负责发现，GATT 连接负责灯光 opcode 传输。

1. **准备并烧录固件。** 本项目按 Espressif 官方 ESP-IDF 工程构建，目标板为 `ESP32-S3-DevKitC-1`（常简称 DevKit1）。安装支持 ESP32-S3 的 ESP-IDF，使用 USB 数据线连接开发板，在 PowerShell 中执行：

   ```powershell
   cd firmware/esp32s3_ble
   idf.py set-target esp32s3
   idf.py build
   idf.py -p COM7 flash monitor
   ```

   将 `COM7` 换成设备管理器中实际的串口；DevKit1 可使用 USB-UART 或 USB-JTAG 暴露的实际 COM 端口。串口日志出现 `advertising as CEPV-ESP32S3` 后，说明 NimBLE GATT 服务已经开始广播。上位机的 `BLE Light` 通道只发送开关灯 opcode，固件负责实现灯光回调。固件默认服务名为 `CEPV-ESP32S3`，并使用以下固定 UUID：

   | GATT 属性 | UUID | 用途 |
   | --- | --- | --- |
   | Service | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d10` | CEPV 自定义服务 |
   | Command | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d11` | `BLE Light` 写入单字节 `0x01`/`0x00` |
   | Status | `8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d12` | 读取/订阅 `ready`、`accepted`、`executed`、`duplicate` 或拒绝状态 |

2. **确认电脑能发现开发板。** 上位机必须有可用的 Bluetooth LE 适配器。安装项目依赖后，可以用以下脚本列出附近 BLE 设备及地址：

   ```powershell
   python -c "import bleak; print('bleak import ok')"
   ```

   上面的单行命令只用于确认 `bleak` 可导入；实际扫描请运行：

   ```python
   import asyncio
   from bleak import BleakScanner

   async def main():
       for device in await BleakScanner.discover(timeout=8.0):
           print(device.name or "<unnamed>", device.address)

   asyncio.run(main())
   ```

   找到名称为 `CEPV-ESP32S3` 的设备后，记录操作系统返回的 `address`。Windows 通常显示蓝牙 MAC 地址；macOS/Linux 可能显示系统生成的标识，必须使用当前平台扫描到的值。若系统要求配对，请先在 Windows“蓝牙和设备”中完成配对，再运行上位机。

3. **配置上位机环境。** 在启动 GUI 的同一个 PowerShell 窗口设置 BLE 模式和设备地址；服务、命令、状态 UUID 已有默认值，只有地址是必需项：

   ```powershell
   $env:CROSS_EVENT_TRANSPORT = "ble-light"
   $env:CROSS_EVENT_BLE_ADDRESS = "AA:BB:CC:DD:EE:FF"
   $env:CROSS_EVENT_BLE_LIGHT_ON_ACTION = "HAND_UP"
   $env:CROSS_EVENT_BLE_LIGHT_OFF_ACTION = "WAVE"
   $env:CROSS_EVENT_BLE_SERVICE_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d10"
   $env:CROSS_EVENT_BLE_COMMAND_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d11"
   $env:CROSS_EVENT_BLE_STATUS_UUID = "8b6ac2d0-2f4d-4d3e-9f6b-8f4a4b0a6d12"
   ```

   也可以在启动 GUI 时直接传入 `--action-transport ble-light --ble-address AA:BB:CC:DD:EE:FF`；CLI 参数只覆盖本次进程的环境配置。为兼容旧脚本，`--action-transport ble` 仍接受但会归一化为 `BLE Light`。

   将地址替换为扫描结果；不要把示例地址当作真实设备地址。可选参数为 `CROSS_EVENT_BLE_LIGHT_ON_ACTION`（默认 `HAND_UP`）、`CROSS_EVENT_BLE_LIGHT_OFF_ACTION`（默认 `WAVE`）、`CROSS_EVENT_BLE_CONNECT_TIMEOUT`（默认 10 秒）、`CROSS_EVENT_BLE_WRITE_TIMEOUT`（默认 5 秒）、`CROSS_EVENT_BLE_ACK_TIMEOUT`（默认 3 秒）和 `CROSS_EVENT_BLE_REQUIRE_ACK`（默认 `1`）。确认 `requirements.txt` 或 `python -m pip install -e ".[ble]"` 已安装 `bleak`。

4. **在 GUI 中选择 BLE Light。** 启动 `python -m cross_event_verifier gui --vision-backend production`，等待模型预加载完成，在“事件传输”下拉框选择 `BLE Light`。采集运行时不能切换协议；需要切换时先点击“停止”，修改环境变量后再选择新的传输方式。GUI 显示“BLE Light：已启用”只表示客户端已创建并通过配置校验，不代表设备已经收到事件。

5. **触发首次连接并验证回执。** Bleak 客户端采用按需连接：不会在打开 GUI 时持续占用连接，而是在已确认身份完成动作后，首次调用 `publish()` 时连接设备、发现服务、订阅 Status 特征并向 Command 特征写入数据。`BLE Light` 只写入 `0x01`（开灯）或 `0x00`（关灯），未映射动作不会产生 BLE 写入。状态特征应回送带 `command` 的 `accepted`、`executed` 或 `duplicate` 结果；拒绝、格式错误和应用层回执超时会显示为动作事件传输失败。

6. **确认接口边界。** `BLE Light` 只发送一个灯光 opcode，不携带身份、Track、摄像头、动作文本或时间字段。ESP32-S3 固件必须校验写入长度和值、调用灯光处理函数，并以 Status 特征回送处理结果；具体 GPIO、米家设备或继电器映射仍由固件/上层自动化实现。生产部署还应在 `menuconfig` 启用验证过的 BLE 配对与加密。

7. **断线重连。** 每次写入失败时 bridge 会断开旧客户端；下一条已授权动作会重新连接。若连续失败，检查开发板是否仍在广播、地址是否变化、电脑蓝牙适配器权限、串口日志和服务 UUID；不要通过降低身份阈值来“测试”连接。BLE Light 应只写入一个字节（`0x01` 或 `0x00`）；长度和值错误、服务不存在或未订阅 Status 都会被报告为传输错误。

**世界坐标标定与使用教程（固定机位地面平面）**

地面控制点（Ground Control Point，GCP）用于建立“图像像素 → 地面世界坐标”的 Homography。当前实现假设目标落在同一地面平面：图像坐标 `U` 向右、`V` 向下，单位为像素；世界坐标使用用户定义的 `X/Y` 平面，通常以米为单位，原点和轴方向必须在现场明确记录，未记录的 `Z` 默认视为 0。不要把墙面、桌面或人体关键点当作地面控制点。

1. **先定义坐标系。** 例如把客厅左后方地面角点设为原点 `(X=0,Y=0)`，沿房间宽度向右为 `+X`，沿房间深度向前为 `+Y`，单位填写 `m`，世界坐标系字段填写 `ground-meters`。用卷尺量出地面角点、门槛或地砖交点的实际距离，并在纸面或配置管理中保留这份定义。

   **X/Y 的实际测量方法：** X、Y 是地面上的物理距离，不是从图片像素换算出来的数值。准备卷尺、激光测距仪或带直角辅助线的测量工具，在地面贴出原点和两条互相垂直的轴线。对每个标记点，先沿 `+X` 方向从原点量到该点的垂足得到 `X`，再沿 `+Y` 方向量到另一条垂足得到 `Y`；如果点不在墙边，用直角尺、激光十字线或拉两条垂直基准线确定垂足，不要直接量点到原点的斜线作为 X 或 Y。以原点在房间左后角、坐标单位为米为例：地面标记点距左墙 1.20 m、距后墙 0.80 m，就记录为 `(X=1.20,Y=0.80)`；位于 `+X` 轴上的点记录 `Y=0`，位于 `+Y` 轴上的点记录 `X=0`。

   测量时应遵循以下顺序：

   - 先用卷尺或激光测距仪确定原点到两条轴线端点的长度，并在地面贴胶带保持轴线不移动；
   - 在每个候选控制点放置平整、可在画面中清楚看到的标记，测量标记中心到两条轴线的垂直距离；
   - 将读数统一换算成同一单位（推荐米，保留至少厘米级精度），立即记录点 ID、X、Y 和测量误差；
   - 用另一把尺或从相邻墙面反向复测至少一个点。两次读数差异明显时，先修正现场标记和坐标记录，再打开 GUI；
   - 不要把人体脚底、检测框中心、墙面高度或斜坡上的点当作地面 X/Y。Homography 只描述同一平面上的位置。

   例如现场量测记录可以先写成下表，再逐点录入 GUI；`P1`～`P4` 是地砖交点或胶带标记中心，括号中的误差是测量复核估计值：

   | 点 ID | 从原点沿 X 测量 | 从原点沿 Y 测量 | 录入世界坐标 | 复测差异 |
   | --- | ---: | ---: | --- | ---: |
   | P1 | 0.00 m | 0.00 m | `(0.00, 0.00)` | 0.00 m |
   | P2 | 6.00 m | 0.00 m | `(6.00, 0.00)` | 0.01 m |
   | P3 | 6.00 m | 4.00 m | `(6.00, 4.00)` | 0.02 m |
   | P4 | 0.00 m | 4.00 m | `(0.00, 4.00)` | 0.01 m |

   表中的“从原点沿 X/Y 测量”是实际量尺读数；录入时只填写最后一列坐标中的数值到“世界 X”“世界 Y”字段。若使用英尺、厘米等其他单位，必须让所有控制点、`单位`字段和后续区域判断保持一致，不能混用 `m` 与 `cm`。

2. **准备同一机位的静态画面。** 打开 GUI 的“03 / 地面控制点”页，点击“加载画面…”选择固定摄像头的一张清晰画面。画面分辨率、裁剪和镜头必须与之后运行识别的来源一致；换摄像头、改分辨率或改变相机位置后必须重新标定。把“摄像头 ID”填成输入源中的同一值，例如 `camera-1`。
3. **在画面上标记地面点。** 依次点击地面上至少 4 个不共线、尽量覆盖整个可用区域的点。点击后 GUI 会自动填入“像素 U”和“像素 V”；在右侧填写唯一“点 ID”以及现场量测的“世界 X”“世界 Y”，然后点击“添加 / 更新”。下面是一组 6 m × 4 m 地面矩形的字段格式，数值仅用于说明如何填写，实际部署应替换为现场量测值：

   | 点 ID | 像素 U | 像素 V | 世界 X (m) | 世界 Y (m) |
   | --- | ---: | ---: | ---: | ---: |
   | P1 | 420 | 760 | 0.0 | 0.0 |
   | P2 | 1500 | 760 | 6.0 | 0.0 |
   | P3 | 1700 | 350 | 6.0 | 4.0 |
   | P4 | 300 | 350 | 0.0 | 4.0 |

   点应分布在地面四周，不要把 4 个点挤在画面一角；可以继续添加 5～8 个点，让 RANSAC 有机会识别误标点。点 ID、图像坐标和世界坐标都不能重复，也不能全部共线。
4. **计算并校核。** 点击“计算并校核 Homography”。表格中的“内点/外点”表示 RANSAC 判断，状态栏会显示内点数量、RMSE 和最大重投影误差，右侧会显示 3×3 的 `H` 矩阵。单位为 `m` 时，RMSE 应结合现场测量误差设定验收标准；出现外点或误差很大时，优先检查点位、坐标顺序、单位和相机画面是否对应，不要只删除异常点。
5. **导出并按机位管理。** 校核通过后点击“导出标定…”，保存为例如 `calibration/camera-1-ground.json`。文件格式为 `gcp-v1`，包含 `camera_id`、`coordinate_frame`、`unit`、`image_size`、控制点、`image_to_ground` 矩阵及误差报告。每个摄像头单独保存一份，导入时确认画面与标定文件属于同一机位；点击“导入标定…”后 GUI 会重新加载点表并再次校核。
6. **把像素位置换算成世界位置。** 对一个人的地面位置，建议取检测框底边中心而不是框中心：`u=(x1+x2)/2`、`v=y2`。在当前版本中，GUI 会完成控制点录入、Homography 校核和 JSON 导出，但不会自动把实时 Track 或 `ActionEvent` 增加 `world_x/world_y` 字段；需要在外部自动化、数据分析或后续集成代码中读取导出的矩阵。下面的调用会把像素 `(u,v)` 转成 `(X,Y)`：

   ```python
   import json
   from pathlib import Path

   import cv2
   import numpy as np

   calibration = json.loads(
       Path("calibration/camera-1-ground.json").read_text(encoding="utf-8")
   )
   H = np.asarray(
       calibration["homography"]["image_to_ground"], dtype=np.float64
   ).reshape(3, 3)

   x1, y1, x2, y2 = 480.0, 260.0, 760.0, 920.0
   pixel = np.array([[[((x1 + x2) / 2.0), y2]]], dtype=np.float64)
   world_x, world_y = cv2.perspectiveTransform(pixel, H)[0, 0]
   print(f"ground position: X={world_x:.3f}, Y={world_y:.3f} m")
   ```

   得到世界位置后即可按区域做业务判断，例如定义 `0≤X≤6`、`0≤Y≤4` 为客厅，再结合 `identity_id` 和 `action` 触发场景。透视变换只对标定覆盖的地面范围可靠；画面外、斜坡、台阶和非地面平面不能直接套用同一个 `H`。

### 注册用户

1. 在“输入源”启动摄像头或视频，等待目标形成稳定 Track。生产 GUI 默认勾选“自动注册新人物（默认开启）”，OSNet 会先确认视觉身份并生成伪匿名编号，再让后续独立事件学习步态原型。
2. 需要明确指定 ID 时，在“人物注册”面板的“人工身份 ID”输入已经规划的编号（例如 `P1`），在 Track 表选中目标行，点击“人工登记选中目标（兜底）”。登记请求会排队到采集线程安全执行。
3. “跨事件候选键”只用于外部流程提供稳定的事件关联键；不要填写临时 Track 号，也不要把动作标签当作候选键。来源切换时，已有正式身份会由 OSNet 重新绑定。
4. 登记成功后，先检查“当前正式身份”列表和结果列，再采集下一次独立事件。清除身份时使用“清除现有 ID（重新建库）”；程序会在清除前创建数据库备份，确认备份位置后再继续。
5. 注册完成只表示视觉身份已经建立或确认，并不代表步态已经 `READY`。步态状态需要满足独立事件、有效帧和多视角等条件后才会提升。

### 身份识别

1. 启动来源后，系统对每个 Track 依次执行检测、跟踪、外观特征提取和姿态质量检查。
2. 对已有视觉身份，OSNet 先给出稳定候选；当 GaitGraph2 已有足够的跨事件原型时，再按校准阈值融合步态证据。多人同时出现时使用一对一指派，避免一个身份被多个 Track 占用。
3. 右侧“当前目标”表的列依次显示 `Track`、`动作`、`身份/候选`、`结果`、`自动流程` 和 `分数`。`UNKNOWN`、`AMBIGUOUS`、`NEED_MORE_DATA` 等结果会保留为安全的非确认状态。
4. 关键点不足、轨迹过短、动作不是明确 `WALK` 或出现身份冲突时，系统会隔离该窗口，不把失败伪装成某个身份的负证据，也不把非行走动作写入 GaitGraph2。
5. 短时动作只在身份决策为已知时发布；“P1 正在挥手”是“已识别的 P1 做了动作”的描述，不会反过来证明该 Track 是 P1。

### 视频识别

1. 在“输入源”选择“视频文件”，点击“浏览…”选择 `.mp4`、`.avi`、`.mov`、`.mkv` 或 `.wmv` 文件。
2. 设置“来源 ID”以标识机位或数据源；需要循环同一文件时设置“视频重复学习”次数为正整数。重复次数只在当前 `capture_session` 中回放，不会自动生成独立事件。
3. 点击“开始”，GUI 会逐帧读取视频、绘制检测框和 Track 结果，并更新身份状态、动作、自动流程和处理耗时；点击“停止”可结束当前来源。
4. 要构建正式跨事件样本，应使用不同日期、不同摄像头或重新导出的真实采集事件，并保持事件元数据、来源 ID 和模型协议可追溯。
5. 文件播放结束后检查状态栏和数据库；若只需验证视频读取链路，可先用 `--vision-backend demo`，不要把演示后端的结果当作生产身份结论。

## 界面说明

- **输入源区**：摄像头/视频文件单选、设备号、文件浏览、来源 ID、视频重复学习次数、开始/停止按钮，以及生产后端预加载进度。
- **事件传输选择**：输入源区下方的“事件传输”下拉框仅提供 `MQTT` 与 `BLE Light`。状态显示“已启用”“未配置”或“不可用”；采集运行中不能切换，需先停止来源。
- **实时识别页**：左侧显示当前画面和框选结果，右侧显示 Track 表、人物注册、外观吸收请求、当前正式身份和清除身份按钮。
- **手动动作触发**：在 Track 表或“目标（Track / 身份）”下拉框选择当前对象，从动作下拉框选择 `SQUAT`、`SIT_DOWN`、`STAND_UP`、`HAND_UP` 或 `WAVE`，点击“按所选动作触发”。管线会在下一帧确认该 Track 仍绑定 `KNOWN` 身份后生成事件；未知目标只显示排队/门控状态，不会发出控制消息。
- **人物注册面板**：自动注册开关、人工身份 ID、跨事件候选键和人工登记按钮。自动流程状态会提示视觉身份确认及步态学习阶段。
- **实时参数页**：查看并整组应用本次运行的阈值、帧数和质量参数；参数更新在下一帧边界生效，不会改写历史数据库。
- **地面控制点页**：加载固定机位画面后点击地面点，录入像素 `U/V` 与现场量测的世界 `X/Y`，使用至少 4 个不共线点计算并校核 Homography，再导入/导出 `gcp-v1` JSON；该页只负责标定与校核，不自动修改实时识别流，完整操作见“第一次使用”中的世界坐标教程。
- **状态栏**：显示模型加载、来源状态、帧号、目标数量、处理耗时、登记结果、动作事件传输状态和异常信息。

## 身份状态说明

系统有三个相互关联但不等价的状态层。GUI 的“结果”列通常展示决策类别，SQLite 还保存生命周期和步态就绪度。

| 层级 | 可能取值 | 含义 |
| --- | --- | --- |
| 公开决策 `DecisionKind` | `formal_match` | 通过当前校准和开放集检查的正式匹配。 |
|  | `visual_identity_created` | 根据稳定外观创建了新的视觉身份。 |
|  | `unknown` | 没有可靠的已知身份匹配。 |
|  | `ambiguous` | 多个候选分数接近或指派不唯一，暂不确认。 |
|  | `need_more_data` | 有效帧、覆盖率或质量不足，需要继续采集。 |
|  | `deferred` / `candidate_created` / `candidate_updated` | 证据被延后或隔离候选已创建/更新，尚未成为正式身份。 |
|  | `conflict` | 外观、步态或事件协议出现矛盾，进入隔离路径。 |
| 身份生命周期 `VerificationState` | `unknown` | 尚未建立可用候选。 |
|  | `isolated_candidate` | 只存在于隔离候选库，不能用于高风险自动化。 |
|  | `provisional_identity` | 有初步身份但仍需独立证据。 |
|  | `confirmed_identity` | 已确认的正式伪匿名身份。 |
|  | `suspended` / `merged` / `revoked` | 暂停、合并到其他身份或撤销，默认不参与正常匹配。 |
| 步态就绪度 `GaitReadinessState` | `not_started` | 尚未采集有效步态事件。 |
|  | `learning` | 正在积累独立步态事件。 |
|  | `provisional` | 已有可用但不足以完全放开的步态原型。 |
|  | `ready` | 达到默认事件、视角和开放集检查门槛。 |
|  | `conflict` | 步态原型之间存在冲突，需要复核或重新建库。 |

“视觉身份已确认”与“步态就绪”可以同时处于不同阶段；例如 `confirmed_identity + learning` 是正常状态，不表示识别失败。动作识别没有独立的身份状态，动作质量不足时只会保持 `UNKNOWN` 或不产生事件。

## 使用建议

- 固定摄像头高度、焦距和来源 ID，尽量拍到完整人体；侧面或斜侧面应重新验证膝、髋、腕关键点阈值。
- 用不同日期、服装、行走方向和摄像头采集独立事件；不要用连续回放同一个文件制造“多事件”假象。
- 注册和首次学习阶段优先保证光照、无遮挡和稳定跟踪，再逐步引入真实部署中的遮挡与视角变化。
- 对灯光、门锁、燃气等高风险设备增加 Home Assistant 条件、冷却时间和人工确认；不要仅凭一次 `HAND_UP` 直接执行不可逆操作。
- MQTT 使用非 retain 的动作消息和可追踪 `event_id`；自动化侧应做幂等和身份/区域二次校验。BLE Light 端点上线前应在 ESP-IDF 中启用配对/加密，并在固件严格验证单字节 opcode。
- 将数据库、模型清单、校准文件和运行参数一起纳入部署记录；不在日志、环境变量或 MQTT payload 中暴露不必要的个人信息和密码。

## 当前限制

- 这是开放集视觉原型，不是人脸、证件或实名系统；`P1` 等 ID 是应用层伪匿名标识，不能单独作为法律或高风险授权依据。
- 生产模型权重未随仓库提交，CUDA、驱动、ONNX Runtime provider 和显存决定实际吞吐；`doctor` 通过只代表依赖就绪，不代表目标摄像头精度达标。
- 步态匹配对视角、服装、遮挡、跟踪断裂和关键点质量敏感；25/45 帧、覆盖率和多事件门槛是当前默认工程参数，不是跨场景保证。
- 当前轻量动作检测器只保守覆盖下蹲、坐下、站起、举手和挥手；`RUNNING`、搬运、推拉等标签可被路由隔离，但没有同等成熟的几何 FSM。可选 PoseC3D/HMDB51 仍需额外 MMAction2 环境和目标域校准。
- 项目没有原生米家设备协议适配。当前路径是 MQTT 交给 Home Assistant（再由其接入米家）或自定义 BLE GATT 交给 ESP32-S3；米家账号、设备配网和具体自动化不在视觉仓库内。
- ESP32-S3 固件目前是 NimBLE GATT 传输演示：`BLE Light` 只接收 `0x01`/`0x00` 并需要实现灯光回调；仓库示例仍不默认执行 GPIO/家电动作，也没有替用户完成安全配对，必须先做硬件、射频、断线重连和权限验证。
- 数据库为本地 SQLite，尚无多主机同步、云端权限、集中密钥管理或跨节点冲突合并服务；Track ID、BLE 地址和相机来源也不能跨部署直接复用。
- 同一视频的重复学习仍属于同一个采集会话；实际摄像头、Home Assistant、BLE 硬件和不同光照服装下的验收需要使用者自行完成。
- 依赖和模型各自受上游许可证、权重条款及数据集使用范围约束，不能仅依据本项目许可证推断所有组件均可商用。

## 项目状态

当前代码已经具备从输入源到 GUI 决策的端到端原型：检测/跟踪、OSNet-first 注册、GaitGraph2 跨事件状态、动作隔离、SQLite 审计、运行时参数和地面控制点页面均已接入；核心模块在 `tests/` 中有相应的状态、路由、动作和传输边界测试。

MQTT 发布器、Bleak GATT 客户端桥接、ESP-IDF ESP32-S3 NimBLE 服务端和 GUI 协议选择已提供可运行的集成边界，但需要真实 broker、Windows 蓝牙适配器、ESP32-S3 板卡和 Home Assistant/设备侧配置才能完成现场验收。模型二进制、目标摄像头验证集、米家自动化和 BLE 安全策略仍属于部署工作，不应视为仓库开箱即用的生产承诺。

## Roadmap

- **近期：部署闭环**：在目标 Windows 机器上固定模型哈希和 CUDA/ONNX provider，完成 MQTT broker、Home Assistant 自动化、ESP32-S3 BLE 连接、断线重连、事件幂等和安全配对测试。
- **近期：质量校准**：为每个摄像头建立真实验证集，调节 gait/action 阈值、遮挡处理、冷却时间和 open-set 拒识；记录误接受、误拒绝、未知和冲突比例。
- **中期：数据与运维**：完成 `cross_event_training_v1` 的合法数据处理与事件分割，补充跨服装/跨视角评测、模型版本迁移、集中配置和可观测性。
- **中期：动作扩展**：在现有几何 FSM 上加入双手举起、弯腰、躺下和动作组合；对喝水、拿取、开门等复杂行为，再评估 PoseC3D、ST-GCN++ 或定域训练分类器，而不是默认增加大型模型。
- **后期：设备生态**：按需要增加 Home Assistant MQTT discovery、米家侧稳定集成或其他厂商适配；扩展 BLE 命令认证、设备能力协商和多 ESP32 端点管理。
- **后期：规模化**：引入多摄像头服务编排、远程审计、权限隔离和可恢复的原型同步，同时保持动作层不污染身份原型的边界。

## 文档

- `CONTEXT.md`：领域术语、身份/步态状态、质量门槛和安全不变量。
- `docs/architecture/README.md`、`docs/architecture/module-map.md`：模块边界、数据流和扩展入口。
- `docs/research/`：跨视角、OSNet、GaitGraph2、动作识别、模型兼容性和 GUI 研究资料。
- `research results/cross-event-verifier_短时动作识别与智能家居扩展方案.md`：短时动作与智能家居的设计论证及分阶段实施方案。
- `models/README.md`：生产模型清单、下载方式、哈希和可选 PoseC3D 运行条件。
- `datasets/cross_event_training_v1/README.md`：训练数据集构建、事件分割、合法来源和验证工具。
- `firmware/esp32s3_ble/README.md`：ESP32-S3 NimBLE GATT 服务、UUID、编译、烧录和安全注意事项。
- `THIRD_PARTY_NOTICES.md`：Ultralytics、OpenGait、MMAction2、Bleak、ESP-IDF 等第三方组件和模型说明。
- `CONTRIBUTING.md` 与 `LICENSE`：贡献约定、Apache-2.0 项目许可和发布要求。

## FAQ

**必须有 GPU 才能运行吗？**

不需要 GPU 才能运行 `demo`、核心 API 或部分诊断；生产 YOLO/RTMPose/OSNet/GaitGraph2 链路对模型、驱动和推理 provider 有较高要求。先运行 `doctor`，再根据实际硬件选择生产或演示后端。

**为什么画面里有人却一直是 UNKNOWN 或 NEED_MORE_DATA？**

系统采用 open-set 安全策略。完整人体不足、Track 太短、外观候选接近、步态事件不够或模型协议不一致都会保留非确认状态。改善光照、机位、有效帧和独立事件数量，并检查 `doctor` 与运行时参数，而不是降低阈值后强行接受。

**如何确认用户已经注册？**

在“人物注册”面板查看“当前正式身份”，并在 Track 表观察 `formal_match`/视觉身份创建结果。正式视觉身份和 `GaitReadinessState=ready` 是两件事；注册后仍需独立步态事件。

**重复同一个视频为什么不等于多个跨事件样本？**

重复回放沿用当前来源的 `capture_session`。它可用于检查管线和增加同一事件的稳定帧，但不能证明跨日期、跨机位或跨事件泛化；正式评估必须提供新的事件元数据和真实采集。

**动作识别会改变身份吗？**

不会。动作在 Track 级内存中短时维护，只能在身份已经被确认后发布；动作失败、遮挡或 ID switch 会清理动作状态，不会写入身份原型，也不会生成身份负证据。

**如何选择 MQTT 或 BLE Light？**

先设置环境变量，再启动 GUI；采集停止时可在“事件传输”下拉框切换。MQTT 示例配置为 `CROSS_EVENT_TRANSPORT=mqtt`、`CROSS_EVENT_MQTT_HOST=192.168.1.20`、端口 `1883`；BLE Light 配置为 `CROSS_EVENT_TRANSPORT=ble-light`、`CROSS_EVENT_BLE_ADDRESS` 为操作系统扫描到的 ESP32-S3 地址。GUI 只显示 `MQTT` 和 `BLE Light` 两项；未配置对应地址或 broker 时显示“未配置”，不会伪造已发送状态。

**如何手动指定一个对象执行动作？**

在“手动动作触发”面板选择当前 Track（已确认身份会显示为 `Track N · P1`），再选择 `SQUAT`、`SIT_DOWN`、`STAND_UP`、`HAND_UP` 或 `WAVE`，点击“按所选动作触发”。请求由采集线程在下一帧处理，只有仍为 `KNOWN` 的身份才会生成并发布 `ActionEvent`；请求不会写入步态原型，超过短时 TTL 或发生 Track 消失/身份切换会被丢弃。

当输出选择为 `BLE Light` 时，只有配置为“开灯动作”或“关灯动作”的标签会产生 BLE 写入（默认 `HAND_UP`→`0x01`、`WAVE`→`0x00`）；其他动作仍可在 MQTT 中发布或在界面中审计，但不会向灯光端点发送指令。

**BLE Light 能直接控制米家灯或空调吗？**

不能直接控制。BLE Light 固件提供的是本项目自定义 GATT 服务，桌面端只写入一个灯光 opcode（`0x01` 开灯或 `0x00` 关灯）并接收状态回执，不发送动作 JSON。要控制米家生态，应让 MQTT 事件进入 Home Assistant，或在经过认证的 ESP32 设备侧实现适配；设备映射和场景规则不属于视觉识别核心。

**如何编译 ESP32-S3 固件？**

安装支持 ESP32-S3 的 ESP-IDF 后，在 `firmware/esp32s3_ble` 执行 `idf.py set-target esp32s3`、`idf.py build`，再用实际串口执行 `idf.py -p COM7 flash monitor`。首次接入生产执行器前，必须在 `menuconfig` 启用配对/加密并替换仅记录日志的处理函数。

**如何处理模型下载或 GUI 启动失败？**

先激活 `.venv`，重新运行 `python -m cross_event_verifier doctor` 并查看 `issues`。缺少 Qt 安装 `.[gui-qt]`，缺少 MQTT 安装 `.[smart-home]`，缺少 BLE 安装 `.[ble]`；模型文件缺失或哈希不符时按 `models/README.md` 重新同步，不要删除数据库来掩盖模型协议问题。

## License

本项目代码以仓库根目录 `LICENSE` 中的 Apache License 2.0 发布。该许可不自动覆盖第三方依赖、预训练权重、数据集或厂商 SDK；发布和商用前必须同时阅读 `THIRD_PARTY_NOTICES.md` 及各上游条款。特别是 Ultralytics 的代码/模型许可、OpenGait/GaitGraph2 的研究与权重限制、MMAction2/PoseC3D 权重和数据集条款，需要按实际版本和用途确认；米家、Home Assistant、ESP-IDF 和具体硬件也各有独立商标与许可边界。

## Acknowledgements

- 感谢 Ultralytics YOLO 与 ByteTrack 生态提供检测、跟踪基础。
- 感谢 OSNet-AIN、OpenGait/GaitGraph2 及其研究工作提供外观和步态表示思路。
- 感谢 OpenMMLab MMPose/RTMPose 与可选 MMAction2/PoseC3D 提供姿态和动作模型接口。
- 感谢 PyTorch、ONNX Runtime、OpenCV、NumPy、SQLite 和 PyQt6/Tk 提供运行时与桌面基础设施。
- 感谢 Eclipse Paho MQTT、Bleak 和 Espressif ESP-IDF/NimBLE 项目，使动作事件能够连接到消息总线和 ESP32-S3 GATT 端点。
- 感谢 Home Assistant 及其 MQTT 自动化生态提供与灯、插座、空调、窗帘和场景编排的解耦接入路径。
