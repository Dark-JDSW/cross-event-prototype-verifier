# cross-event-prototype-verifier

一个独立运行的跨事件人物验证系统：以步态作为身份确认锚点，外观作为由强步态授权后才能吸收的证据，并提供摄像头/视频 GUI、自动建号、开放集拒识、隔离图库和可回滚持久化。

```text
摄像头 / 视频
      ↓
低照度检测与 CLAHE（按需）
      ↓
YOLO11L 人物检测 → ByteTrack 多目标轨迹
      ├──────────────→ OSNet-AIN 512 维外观特征
      └→ RTMPose 17 点姿态 → GaitGraph2 384 维时序步态特征
                                  ↓
                      质量门控 + 分支校准 + 一对一指派
                                  ↓
      UNKNOWN / 等待更多数据 / 强步态确认 / 外观授权响应直接放行
                                  ↓
                  quarantine / formal 多原型 SQLite 图库
```
## 更新日志
- V1.00
  - 基础框架，效果差，好在已跑通技术栈
- V1.05
  - 修复了GUI的小BUG
  - 取消项目分工设计
  - 优化视频流逻辑，流畅度大幅提升！
  - 重构算法，一定程度解决了强步态Pg得分过低，导致自动注册功能失效的问题(但还是不好用！)
  - 试着用轮廓识别方案替换GaitGraphic2，但是效果很差，又换回来了（

## 先运行起来

```powershell
.\.venv\Scripts\Activate.ps1
python -m cross_event_verifier doctor
python -m cross_event_verifier gui
```

如果模型文件不在本机，先看“模型资产与云端迁移”一节，从私有云端同步完成后再运行 `doctor`。

## 证据规则

- 步态是正式身份锚点。未知 Track 只有在完整下肢可见、确实存在行走周期且多次时序嵌入稳定时，才允许自动创建 `P1`、`P2`……。
- 自动建号只写入步态原型，不把当时衣着直接当作永久身份依据。
- 强步态确认会签发一次性 `appearance_request_id`，自动绑定到提出请求的 Track。
- 请求仍有效且当前外观通过强质量门时，外观响应直接放行对应身份，并吸收 OSNet 外观原型。
- 没有步态授权令牌的强外观不能创建或确认正式身份，只能作为隔离证据。
- 高质量步态若明确低于“新人物步态上限”，即使衣着外观很像旧人物，也会拒绝全部旧 ID 并进入稳定步态采集，避免新人物永久卡在“疑似旧 ID”。
- `DEFERRED` 只表示“尚未判定”，不会携带或显示正式身份；自动路径在图库只有一个步态身份时，必须有强外观佐证才能确认，否则继续采集或拒绝注册。
- 自动建号还要通过开放集新颖性门：如果稳定步态仍落在已有 formal gait 原型的相似区间，不会复制成新的 `P` 身份；这类证据会停留在隔离区，清空当前窗口后重新采集/复核，不会永久终止该候选人。
- 只有两个分支都达到质量门、概率门和 Top-2 间隔门时才算“证据冲突”。单帧冲突会丢弃当前步态窗口并重采集，不会直接终止候选人；明确挂起仍需人工复核。
- 步态质量 `Q_gait` 与身份相似度 `S_gait` 分开处理：质量分为 `INVALID`、`PARTIAL`、`STRONG`。`PARTIAL` 只进入等待/隔离区，不能作为“不是该身份”的负证据；严重 ID switch、腿部完全不可见、框截断或序列不足才会硬拒绝。
- 多个 formal gait 原型的 Top-1/Top-2 间隔不足时返回 `AMBIGUOUS`，不强行选择旧 ID，也不立即创建新 ID；自动建号必须由至少两个独立采集事件相互确认。
- 正式图库只有一个步态身份或步态 Top-2 过近时，不自动更新该身份的 gait 原型，避免把陌生人的错误向量反馈污染 `P1`。
- 触碰画面边界、人体框异常、下肢关键点不足、遮挡严重或轨迹跳变时，不允许产生强步态/强外观证据。
- 多人同框使用全局一对一身份指派，一个正式身份不会在同一帧分配给多个 Track。

默认自动流程：

```text
T 临时 Track
  → ByteTrack 稳定跟踪
  → 至少 25 个有效全身姿态帧并检测到行走周期
  → 连续 8 个强步态样本通过稳定性门
  → 记录第一个独立步态事件并进入等待
  → 第二个独立采集事件再次通过稳定性门
  → 自动创建 P 身份（仅 gait）
  → 自动签发一次性外观请求
  → 强外观响应直接放行并吸收 appearance
```

## Windows下环境安装
#这里可以根据自身GPU型号适当改变CUDA版本喵

建议使用 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m cross_event_verifier download-models
```

安装依赖并下载模型后，再执行环境与模型自检：

```powershell
python -m cross_event_verifier doctor
```

成功时输出中应包含：

```json
{
  "cuda_available": true,
  "cuda_device": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "production_ready": true,
  "issues": []
}
```

`doctor` 会检查运行依赖、CUDA、ONNX Runtime provider、模型文件大小及 SHA-256。返回码非零表示生产 GUI 不应启动。

## 启动 GUI

```powershell
python -m cross_event_verifier gui
```

也可使用安装后的命令：

```powershell
cross-event-verifier gui
```

首次打开摄像头/视频时需要把四个模型加载到内存，RTX 5060 实测约需 8～15 秒；这段时间窗口仍会显示加载状态。完成预热后，单帧耗时取决于分辨率和同框人数。

GUI 操作要点：

1. 选择摄像头设备号（通常为 `0`），或选择 `mp4`、`avi`、`mov`、`mkv` 视频。
2. 点击开始后，让人物从头到脚完整出现在画面中；脚踝被裁掉时系统会继续跟踪，但不会把代理信息误当成强步态。
3. 人物需要实际行走，而不是只站立。界面会依次显示步态预热、稳定采集、外观待响应和已吸收等阶段。
4. “自动注册新人物”默认开启。关闭后只禁止创建新的 `P` 身份，已有身份识别和已签发请求的响应仍继续。
5. 人工登记和手工令牌响应保留为异常场景兜底；人工登记也只允许以强步态建号。登记时会优先使用当前 Track 最近窗口内最强的有效步态快照，避免最后一帧短暂遮挡导致误报；如果整个窗口都低于强步态门，仍需让目标完整、连续行走后重试。
6. 切换到“实时参数”页，可以在不中断视频、不重载模型的情况下调整开放集、质量、自动注册、校准和生产视觉参数。

### 实时参数页

参数页会显示当前实际生效值、允许范围和用途。点击“应用到运行中”后，整组参数先完成类型及跨字段校验，再通过采集线程在下一帧边界一次性生效；因此不会出现一帧读取到半组新值的情况。无效组合会整组拒绝，原参数保持不变。参数真正改变时会清空尚未完成的步态采样窗口，避免把调参前后的证据混在一次自动建号中；已确认身份和待响应外观令牌不受影响。

最直接影响“新人物被认作旧 ID”的参数是：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| 新人物步态上限 | `0.35` | 高质量步态对图库最佳 ID 的校准概率低于此值时，明确视为新人物候选；调高会更容易建新号，也会增加重复建号风险 |
| 强步态确认概率 | `0.90` | 步态达到此值且通过质量、Top2 间隔后，才确认已有 ID |
| 强步态 Top2 间隔 | `0.08` | 防止两个已有 ID 分数过近时强行确认 |
| 强步态质量 | `0.70` | 新人物拒识、旧人物确认和自动建号共同使用的质量门 |
| 部分步态质量下限 | `0.35` | `Q_gait` 低于此值为 `INVALID`；达到此值但未到强门时为 `PARTIAL` |
| 稳定步态样本数 | `8` | 新人物自动建号前需要连续收集的稳定嵌入数量 |
| 独立步态事件数 | `2` | 新人物自动建号前需要来自不同采集会话/挑战的稳定事件数 |
| 单图库步态近重复上限 | `0.985` | 只有步态余弦相似度低于此值、且强质量外观明确拒绝唯一旧 ID 时，才允许从单身份图库引入新号 |
| 单图库外观拒识阈值 | `0.30` | 单身份图库中，强质量外观低于此概率才可作为新人物的负证据；没有可靠外观时保持保守等待 |

生产视觉页还提供三项吞吐量参数：

- “检测推理间隔”默认值为 `2`。YOLO/ByteTrack 每两帧刷新一次，中间帧复用
  最近轨迹框；缓存为空时仍会逐帧检测，以便新人物及时进入管线。
- “外观提取间隔”默认值为 `6`。OSNet 首次取得外观后每六个 Track 帧刷新，
  没有需要刷新的目标时不会发起空推理调用。周期性刷新会优先安排在不执行
  YOLO 和 GaitGraph2 的帧，减少多个重模型同帧运行造成的卡顿。
- “步态推理间隔”默认值为 `3`。RTMPose 姿态仍逐帧采集，但 GaitGraph2 只在
  轨迹首次成熟或距离上次推理达到该间隔时刷新；同一帧到期的所有 Track 会合并
  为一个 GPU 批次。周期性刷新同样优先避开 YOLO 帧；首次成熟特征不会延迟。

生产 GaitGraph2 输入使用 RTMPose 的原始全帧坐标，并采用与 OpenGait GREW
检查点一致的 COCO 图归一化；检测框只用于质量和行走指标，不再把姿态缩放到框内。
因此更换模型输入格式或继续使用旧库时，必须先清空旧的 gait 原型。

错峰只是短暂顺延周期性刷新，并设置了最大延迟。如果把检测间隔调为 `1`，系统
仍会在达到延迟上限后执行外观和步态刷新，不会因为每帧检测而饿死特征分支。

检测和外观间隔调大可以提升吞吐量，但快速运动时复用框的滞后会增加，外观更新
也会变慢。步态间隔调大则会延长自动注册收集足够稳定步态样本所需的时间。

建议先观察默认值。若陌生人仍长时间显示“疑似 P…”，可小幅提高“新人物步态上限”；若同一人容易重复建号，则应降低该值，并先用目标摄像头数据检查步态概率分布后再考虑小幅降低“强步态确认概率”。校准中点和斜率会改变概率含义，不应只凭单段视频随意修改。

“填入默认值（未应用）”只会填写表单，仍需点击“应用到运行中”。当前版本的动态参数仅对本次进程生效，重启后恢复代码默认值；审计中的 `threshold_version` 会更新为 `runtime-vN`，便于区分调参前后的观测。诊断后端不支持的生产视觉参数会在页面中禁用。

默认数据库是：

```text
data/verifier-production-v1.sqlite3
```
需要指定其他数据库时：
```powershell
python -m cross_event_verifier gui --database data/my-gallery.sqlite3
```

## 视觉后端

生产后端是默认值；缺依赖或权重时会明确报错，不会悄悄用低精度检测器继续自动建库。

```powershell
#一般正常使用这一条即可(即默认)：
python -m cross_event_verifier gui

# 默认：允许自动注册
python -m cross_event_verifier gui --vision-backend production

# 自动回退：生产链路不可用时进入 HOG 诊断模式，并强制关闭自动注册
python -m cross_event_verifier gui --vision-backend auto

# 明确使用 HOG/前景轮廓，仅用于 GUI、摄像头和数据流排障
python -m cross_event_verifier gui --vision-backend demo
#其实大部分时候你可以直接在GUI里改
```

诊断后端的检测框和轮廓特征不具备身份注册精度，所以代码层面的 `supports_automatic_registration=False` 无法由 GUI 勾选绕过。

## 本地模型

生产链路要求以下文件位于 `models/`：

| 文件 | 用途 | 输出/说明 |
|---|---|---|
| `yolo11l.pt` | 人物检测 | YOLO11L，仅保留 person 类 |
| `bytetrack-cross-event.yaml` | 多目标跟踪 | 为跨事件场景设置的 ByteTrack 门限 |
| `rtmpose-s.onnx` | 17 点人体姿态 | ONNX Runtime CUDAExecutionProvider 批量 top-down 推理 |
| `osnet_ain_x1_0_dg.pth` | 外观 ReID | 512 维 L2 归一化向量 |
| `gaitgraph2_grew_state.pt` | 时序步态 | 60 帧重采样、384 维 L2 归一化向量 |

模型大小和哈希记录在 `models/manifest.json`。运行时仅加载本项目内的文件，不访问 `videotracker`。GaitGraph2 权重已转换为只包含张量的安全 state dict，运行时使用 `weights_only=True` 加载。

说明：ONNX Runtime 的 CUDA EP 会把主卷积/推理算子放到 GPU；RTMPose 图中的少量 shape/布局辅助算子可能由 ORT 的内置 CPU EP 处理。强行禁用这类图级 CPU 节点会使该 ONNX 模型无法创建，因此项目不把它误报为“100% 每个算子 GPU”，但也不会允许 RTMPose 在 CUDA provider 不存在时整体退回 CPU。

## 模型资产与云端迁移

四个二进制模型合计约 154 MB，默认保存在项目的 `models/` 目录，并被 `.gitignore` 忽略；`models/manifest.json` 保存直接文件的字节数/SHA-256、转换 checkpoint 的 tensor 指纹和下载源。这样代码仓库不会被大文件和模型版本变更拖慢，但部署机器仍需要在启动前取得模型。

`requirements` 负责 Python 包依赖，模型属于外部大文件和许可资产，需要单独下载。所以克隆项目后的标准部署顺序就是：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m cross_event_verifier download-models
python -m cross_event_verifier doctor
```

`download-models` 默认从 manifest 中的固定 HTTPS 地址下载，并在写入 `models/` 前校验源文件 SHA-256；OSNet-AIN 和 GaitGraph2 会在源文件校验通过后转换为本项目运行时需要的 tensor-only state dict，再校验最终文件。已存在且校验通过的文件不会重复下载。

也可以只下载某一个文件或强制重新下载：

```powershell
python -m cross_event_verifier download-models --model yolo11l.pt
python -m cross_event_verifier download-models --model rtmpose-s.onnx --force
```

建议把模型放到私有对象存储、私有 GitHub Release 或私有 Git LFS。不要在没有确认许可和访问控制的情况下，把 YOLO/OpenGait 权重上传到公开仓库。当前工作区检测到 GitHub 远程地址，但没有可用的 Git LFS 上传凭据，因此本项目没有自动执行远程推送。

如果部署环境不能访问这些公共源，本项目也提供两个不依赖云厂商 SDK 的 PowerShell 工具，用于把同一份 manifest 资产切换到你自己的私有对象存储：

```powershell
# 1. 生成待上传的模型包，并在打包前校验 manifest 哈希
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_model_bundle.ps1 `
    -Output .\dist\model-bundle-v1.zip

# 2. 上传 model-bundle-v1.zip 解压后的 models/下四个二进制文件，
#    以及 manifest.json；然后用 HTTPS 前缀同步到本机并再次校验
$env:CROSS_EVENT_MODEL_BASE_URL = "https://your-private-bucket.example/models"
powershell -ExecutionPolicy Bypass -File .\scripts\sync_model_assets.ps1
python -m cross_event_verifier doctor
```

`CROSS_EVENT_MODEL_BASE_URL` 可以指向私有对象存储的临时签名 URL 前缀、私有 CDN，或 GitHub Release 的资产目录。同步脚本只会接受 manifest 中列出的文件，下载到临时文件并在 SHA-256 校验成功后替换目标；已正确存在的文件会跳过，哈希不匹配时默认停止，不会静默覆盖。

默认公共源与用途如下；下载源和上游许可随模型版本记录在 `models/manifest.json`：

| 模型 | 默认来源 |
|---|---|
| YOLO11L | Ultralytics 官方 assets release |
| RTMPose-s ONNX | Hugging Face `ziq/rtm` 仓库中的 ONNX 导出 |
| OSNet-AIN DG | deep-person-reid 使用的官方 Google Drive checkpoint |
| GaitGraph2 GREW | OpenGait 官方 Hugging Face checkpoint，下载后提取 model state |

如果选择 GitHub LFS，必须确认仓库可用 LFS 配额、模型许可允许上传，并在确认远程目标后执行：

```powershell
git lfs install
git lfs track "models/*.pt" "models/*.pth" "models/*.onnx"
git add .gitattributes models/manifest.json models/README.md
git add -f models/yolo11l.pt models/rtmpose-s.onnx `
    models/osnet_ain_x1_0_dg.pth models/gaitgraph2_grew_state.pt
git commit -m "Store production model assets with Git LFS"
git lfs push --all origin main
git push origin main
```

上面是有外部写入影响的人工确认步骤，脚本不会替你执行。更适合生产部署的做法是使用私有对象存储，并给运行机器发放只读、短期有效的下载 URL；密钥不要写进 README、代码或 SQLite。

## Python API

核心验证器仍可脱离 GUI 和重模型独立调用：

```python
import numpy as np

from cross_event_verifier import (
    CrossEventVerifier,
    FeatureBundle,
    Observation,
    TrackQuality,
)

verifier = CrossEventVerifier()
verifier.register_identity(
    "P1",
    FeatureBundle(
        appearance=np.array([1.0, 0.0, 0.0]),
        gait=np.array([0.0, 1.0, 0.0]),
    ),
)

decision = verifier.verify(Observation(
    camera_id="cam-a",
    capture_session_id="session-a",
    track_id="track-7",
    features=FeatureBundle(
        appearance=np.array([0.98, 0.10, 0.0]),
        gait=np.array([0.0, 0.98, 0.10]),
    ),
    quality=TrackQuality(
        detection_confidence=0.92,
        box_height=180,
        frame_count=30,
        gait_cycles=2,
        walking_ratio=0.90,
        keypoint_visibility=0.88,
        occlusion=0.05,
    ),
))

print(decision.kind, decision.identity_id, decision.score)
```

真实视频管线应使用稳定的 `candidate_id` 关联跨事件候选；否则系统会为每个事件创建独立候选，避免未经授权的自动合并。

## 目录结构

源码现在按职责组织在同一个 Python 包中；模块之间通过公开值对象、视觉适配器、验证器和帧管线的稳定 seam 协作。完整的职责、依赖方向和稳定接口见：

- [模块地图](docs/architecture/module-map.md)
- [组内协作约定](docs/architecture/contributor-guide.md)
- [共同研究入口](CONTRIBUTING.md)

```text
cross_event_verifier/
├── engine.py、config.py、calibration.py、fusion.py、assignment.py
│                         # 身份验证、开放集决策与分数校准
├── gait_graph.py、osnet_ain.py、memory.py、stability.py
│                         # 步态/外观表征与正式原型记忆
├── production_vision.py、vision.py、vision_factory.py、adapters.py
│                         # YOLO、ByteTrack、RTMPose 与视觉适配器
├── pipeline.py、automation.py、runtime_parameters.py、media.py
│                         # 在线帧编排、自动建号与线程/参数管理
├── gui.py、cli.py、storage.py、vector_index.py、model_assets.py
│                         # GUI、CLI、SQLite、向量索引与模型资产
├── types.py             # 公开值对象和模块间接口契约
├── __init__.py          # 公开 Python 门面
└── __main__.py          # python -m cross_event_verifier 入口
docs/architecture/
├── module-map.md        # 模块分组、依赖方向、稳定 seam 和测试入口
└── contributor-guide.md # 分工、实验记录和合并检查
models/
├── manifest.json
├── bytetrack-cross-event.yaml
└── README.md
scripts/
├── prepare_model_bundle.ps1 # 校验并生成待上传 zip
└── sync_model_assets.ps1    # 从 HTTPS/私有对象存储同步并校验模型
```

## 验证

```powershell
python -m unittest discover -s tests -v
python -m compileall cross_event_verifier tests
python -m cross_event_verifier demo
python -m cross_event_verifier doctor
```

测试覆盖质量门控、校准与动态融合、开放集拒识、全局指派、新人物不粘连旧 ID、运行时参数事务、GUI 参数页、人工登记的短暂质量回落、自动建号、一次性外观吸收、防污染图库、SQLite 回滚、混合特征维度迁移，以及生产适配器的批量步态编码、推理间隔和时序特征输出。

## 常见问题

- `doctor` 报模型哈希不符：不要继续使用该文件；重新从可信资产源同步，或恢复 `models/manifest.json` 对应版本。
- 只有上半身或脚踝出画：系统会保留检测框和 Track，但步态质量为零，不会自动建号。
- 首次启动较慢：YOLO、RTMPose、OSNet-AIN 和 GaitGraph2 会在采集线程首次使用时加载；RTX 5060 冷启动通常需要数秒，之后才进入稳定帧率。
- 仍看到 CPU 占用：摄像头读取、OpenCV 预处理、ByteTrack/SQLite/Tk 以及 RTMPose 少量 shape 辅助节点会使用 CPU；用 `doctor` 查看 `cuda_available`、`onnx_providers`，并确认生产后端不是 `demo` 或 `auto`。
- 画面能显示但自动注册没有进展：检查人物是否完整入镜、是否实际行走、是否持续跟踪至少 25 个有效姿态帧，并观察 GUI 中的步态质量提示；若一直显示“疑似旧 ID”，到“实时参数”页查看“新人物步态上限”，不要先大幅降低质量门。
- 自动注册提示“开放集不新颖”：这表示当前步态与已有 formal gait 原型过近，系统已选择拒绝复制身份；应先检查 GaitGraph2 的输入/权重和目标摄像头采集质量，再调整校准阈值，不要强行降低质量门。
- 自动注册提示“步态 Top-2 歧义”：表示最佳和次佳 formal gait 原型没有足够间隔，系统会等待新的独立事件；这不是质量失败，也不会把当前样本当作身份负证据。
- 需要只排查摄像头或 GUI：使用 `--vision-backend demo`；该模式强制禁止自动注册，不应拿来建立正式图库。

## 精度与许可边界

- 默认阈值只能作为可运行起点。部署前应使用目标摄像头数据按 FAR/FPIR 拟合分支校准器，并评估换衣、遮挡、逆光、拥挤和跨镜头条件。
- P2 的模型域差异暂不通过替换模型解决。`cross_event_verifier.evaluation` 提供 `compare_encoder_embeddings()`，可将同一批标注序列的 `HRNet`/`RTMPose` embedding 分别送入，记录 genuine/impostor similarity、max-impostor、d-prime 和 FNIR@FPIR；当前生产模型不会因此自动改变。
- 自动注册要求完整人体和有效行走序列；只看到上半身时没有可靠步态，系统会选择等待，而不是自动建错号。
- Ultralytics 代码/模型涉及 AGPL-3.0 或企业许可；OpenGait 官方资源声明限学术用途。商业部署前必须完成对应授权或替换模型。完整第三方归属与许可提示见 `THIRD_PARTY_NOTICES.md`。

##当前存在问题：
- 难以通过步态识别人物，不过这似乎和我的数据集有关系
- 自动注册功能≈摆设
- GUI丑死了!
- 遮挡场景仍需继续评估吞吐量与轨迹精度的平衡
- ***电脑涨价太猛，我买不起新显卡了***

## 写在最后
- 本项目已在华硕天选Air2025 H350版本上运行成功，该机型性能低于其他同配置机型，故加载时间稍长
