# Cross-Event RGB Gait Identity Dataset v1

这是面向 `cross-event-prototype-verifier` 的数据集构建包，不是把现有 `train/` proxy 数据重新命名。它把外部真实 RGB 视频、身份/服装/session 元数据、项目要求的轨迹和 RTMPose 输出放进同一套可验证协议。

当前状态：`SOURCE_READY_PENDING_PROCESSING`

只有在原始视频、合法使用授权、逐视频时间/哈希映射、人工确认的 walking 事件和重新运行项目检测→跟踪→RTMPose 链路全部完成后，才可以把状态改为 `PRODUCTION_TRAINING_READY`。本目录目前不包含外部原始视频、人员图像或伪造的姿态数组。

## 来源决策

主来源选择 **MEVID**。官方资料报告它包含 158 个身份、598 套服装、8,092 条轨迹、9 个日期（73 天窗口）和 33 个摄像机视角，并公开原始视频 URL、RGB 行人轨迹包和身份/服装/摄像机字段。它最接近本项目的跨事件、换衣、跨视角和遮挡需求。

MEVID 的官方轨迹标注不是当前项目需要的 RTMPose 原图坐标；因此必须从合法取得的原始视频重新运行 YOLO/ByteTrack/RTMPose，并保留真实框、分数、缺帧和时间戳。只使用 MEVID 的 cropped JPEG 或直接把其已有轨迹当成 RTMPose 结果，不会通过本项目质量门。

FVG-B 作为辅助压力测试源记录在 `source_registry.json`：它有 226 人、每人 12 段、3 个前向角度和换衣/携物条件，但多数身份没有足够独立 session/日期，不能单独满足本项目的 P1 目标。DeepChange 的访问协议明确限制为非商业研究且禁止再分发，因此不进入本项目的生产候选集。

## P1 严格拆分

需求文档的“60 个身份最低规模”和五个身份隔离桶存在数量张力：若 calibration-known、calibration-unknown、locked-test-known、locked-test-unknown 都分别使用独立身份，并保留 40 个 train/dev 身份，严格可执行的最低桶数是 80 个身份。因此本构建包默认要求：

| 桶 | 身份数 | 用途 |
|---|---:|---|
| `train-dev` | 40 | 训练、调试和目标域适配 |
| `calibration-known` | 10 | 已知身份校准 |
| `calibration-unknown` | 10 | 开放集阈值选择，不进入 formal gallery |
| `locked-test-known` | 10 | 锁定的已知身份跨事件测试 |
| `locked-test-unknown` | 10 | 锁定的 unknown 测试 |

每个入选身份还必须同时达到：至少 4 个真实独立 session、至少 2 个日期/时间段、至少 2 个服装条件、至少 3 个视角、至少 12 个经人工确认的 walking 事件。生成器不足时直接失败，不会用连续窗口、重复播放或 tracker ID 补数量。

## 目录

```text
datasets/cross_event_training_v1/
├── README.md
├── RESEARCH.md
├── requirements.json
├── source_registry.json
├── source_metadata.example.json
├── events.jsonl
├── identity_split.json
├── build_report.json
├── tracks/
│   └── README.md
├── poses/
│   └── README.md
├── schemas/
│   ├── action_branch_annotations.schema.json
│   ├── events.schema.json
│   ├── source_metadata.schema.json
│   └── tracks.schema.json
├── tools/
│   ├── build_mevid_manifest.py
│   └── validate_dataset.py
└── pose_protocol.md
```

仓库内的三个 JSON/JSONL 文件和两个目录是待处理状态清单，已经补齐以便读取器能够明确报告 `pending_external_data`；它们不包含虚构事件。构建成功后，生成器会在你指定的输出目录写入真实的 `events.jsonl`、`identity_split.json` 和 `build_report.json`。原始视频和外部数据包应留在仓库外；本项目不应提交 30GB/13GB 的 MEVID 外部包。

## 使用方式

1. 依照 MEVID/MEVA 官方协议取得数据访问授权和数据包。不要绕过访问协议或把外部视频复制进本仓库。
2. 按 `source_metadata.example.json` 生成一行一条 MEVID 轨迹的元数据。`capture_session_id`、`capture_period`、原始视频路径、视频 SHA-256、原始帧区间、walking 人工确认、视角、授权 ID 都是硬性字段。
3. 在仓库外运行：

```powershell
python datasets/cross_event_training_v1/tools/build_mevid_manifest.py `
  --mevid-root E:\datasets\mevid `
  --source-metadata E:\datasets\mevid-cross-event\source_metadata.jsonl `
  --output E:\datasets\mevid-cross-event\manifest
```

4. 用项目固定的 YOLO/ByteTrack/RTMPose 版本处理 `events.jsonl` 中的原始视频，按照 `pose_protocol.md` 写出 `tracks/` 和 `poses/`。不得用插值帧增加有效训练帧数。
5. 先检查元数据阶段，再检查生产阶段：

```powershell
python datasets/cross_event_training_v1/tools/validate_dataset.py `
  --dataset-root E:\datasets\mevid-cross-event\manifest `
  --stage metadata

python datasets/cross_event_training_v1/tools/validate_dataset.py `
  --dataset-root E:\datasets\mevid-cross-event\manifest `
  --stage production
```

生产阶段只有在事件、轨迹、姿态、视频引用和切分都通过后才允许进入训练或开放集校准。

### 机器预筛与复核队列

在已经下载真实视频、且生产视觉依赖和 CUDA 环境就绪后，可以先对原始视频目录运行机器预筛：

```powershell
python datasets/cross_event_training_v1/tools/download_and_batch_process_mevid.py screen `
  --video-dir E:\datasets\mevid-cross-event\raw\videos `
  --output-dir E:\datasets\mevid-cross-event\screen
```

该步骤使用项目固定的 YOLO/ByteTrack/RTMPose/GPGait++ 链路，输出：

- `auto_passed.jsonl`：只有一个稳定候选轨迹、步行和姿态质量都达到机器门槛的视频；
- `review_queue.jsonl`：多轨迹、遮挡、低置信度、跟踪断裂、非完整步行或推理异常的视频；
- `machine_screen.jsonl`、`screen_report.json`：全量机器结果和可审计原因。

## 动作分类与独立步态分支

如果目标是先判断人物正在做什么、再让不同动作进入互不干扰的步态分支，请启动独立工具：

```powershell
python datasets\cross_event_training_v1\tools\action_branch_gui.py
```

在工具中载入 `E:\datasets\mevid-cross-event\fast_screen\fast_screen.jsonl`，选择画面中对应的本视频轨迹 ID，设置动作和该动作的开始、结束帧后保存。对于 `walking`，必须同时标注身体视角（`side`、`oblique`、`front`、`back`）和实际行进方向（`left_to_right`、`right_to_left`、`approaching`、`leaving`）。二者共同组成分支，例如 `gait_walking_side_left_to_right`；绝不合并到通用 `gait_walking`。任一字段为 `transition` 或 `unknown` 的片段会保留为隔离记录，不应输入任何正式步态分支训练；`running`、`carrying` 等仍各自使用独立动作分支。

该工具刻意不包含身份、授权、官方轨迹索引或生产许可字段；这些字段与本地动作分类器和动作分支训练无关。

完成一批动作标注后，可把标注的长视频片段转换为项目实际 RTMPose 张量，再训练
轻量动作门控模型。处理会按源视频分组，视频只回放一次；每条标注仍保留自己的
本视频 `track_id` 和帧区间，因此一个视频包含多个人时不会把轨迹混在一起：

```powershell
python scripts\extract_action_branch_poses.py `
  --annotations E:\datasets\mevid-cross-event\action_branch_annotations.jsonl `
  --output-dir E:\datasets\mevid-cross-event\action_branch_poses

python scripts\train_action_classifier.py `
  --manifest E:\datasets\mevid-cross-event\action_branch_poses\manifest.jsonl `
  --output E:\datasets\mevid-cross-event\action_classifier_v1.npz
```

提取脚本必须使用项目固定的 detector→ByteTrack→RTMPose 链路，并从视频开头回放到
标注结束帧，以保持源轨迹 ID 的上下文。若标注来自 `fast_screen`，GUI 会同时保存
其采样步长和 detector image size，提取时按相同帧 cadence/尺寸重放 tracker，避免
长视频全帧重放后本地 track ID 漂移；直接从视频目录打开的标注默认逐帧处理。
`train_action_classifier.py` 按完整源视频
留出验证集，拒绝 `unknown`、未匹配轨迹和单样本动作类别。当前动作模型只是路由
门控，不是身份模型：只有质量足够的 `WALK` 才进入现有正式 GPGait++ 图库；
`RUNNING`、`CARRYING`、`SQUAT`、`SIT_STAND` 等动作各自拥有独立的动作分支标识；walking 的“视角 × 行进方向”分支也必须各自训练、校准和建立隔离图库，
但在没有为该动作训练并校准独立 gait encoder 前，不会写入 WALK formal gallery。

训练完成后，在验证 GUI 中显式加载该模型即可启用学习到的动作门：

```powershell
python -m cross_event_verifier gui `
  --action-classifier E:\datasets\mevid-cross-event\action_classifier_v1.npz
```

不传该参数时仍使用内置的保守运动学门，不会自动猜测或复用一个未校准的模型。

## 与当前项目的边界

- `identity_id` 是跨事件稳定伪名，不能由 tracker 临时 ID 生成。
- `event_id`、`capture_session_id`、`camera_id` 是独立分组键；滑动窗口继承事件键，不能作为新的独立事件。
- `poses/*.npz` 必须是项目当前 GPGait++ 使用的 `[T,17,3]` 语义：原图像素坐标、RTMPose 置信度和真实框链路，详细字段见 `pose_protocol.md`。
- 外部数据源的预提取 skeleton、silhouette 或 proxy COCO17 只能放在单独的 proxy 版本，不能与 `rtmpose` 版本混合训练。
- unknown 必须与所有训练、图库和 calibration-known 身份不相交；正式 `FPIR ≤ 0.1%` 需要约 3,000 个独立 unknown search，不能把连续窗口当独立样本。
