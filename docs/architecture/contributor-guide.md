# 组内协作约定

## 三条研究线

现在不再把“全部视觉”与“全部身份规则”分别交给一个人，而是按流水线拆成三条研究线：

| 研究线 | 负责范围 | 典型问题 |
|---|---|---|
| 感知与轨迹 | 人物检测、跟踪、姿态和质量 | YOLO/ByteTrack、RTMPose、遮挡、出框、轨迹稳定性 |
| 身份表征与证据 | ReID、步态向量、身份决策和校准 | OSNet、GaitGraph2、新模型、开放集拒识、误认旧 ID |
| 在线系统与交付 | 帧编排、GUI、存储、测试和发布 | 摄像头/视频、实时参数、线程、SQLite、可复现实验 |

## 三人协作分配

以下用“参与者 A/B/C”作为占位名称，后续可以直接替换为组内姓名。A 和 B 共同覆盖原来的视觉与证据研究，但各自负责不同的处理阶段；C 负责在线系统和工程交付。

`cross_event_verifier/types.py`、根目录 `__init__.py` 和 `__main__.py` 是三人的共享契约/入口，不归某一人独占；除此之外不在包根目录新增实现。新实现应提交到下面对应的参与者子包。

| 参与者 | 主责范围 | 具体负责内容 | 主要交付物 | 需要协作的接口 |
|---|---|---|---|---|
| 参与者 A | 身份表征与证据 | `participant_a/gait_graph.py`、`participant_a/osnet_ain.py`、`participant_a/engine.py`、`participant_a/config.py`、`participant_a/state.py`、`participant_a/challenge.py`、`participant_a/absorption.py`、`participant_a/memory.py`、`participant_a/stability.py`、`participant_a/assignment.py`、`participant_a/calibration.py`、`participant_a/fusion.py`、`participant_a/evaluation.py`、`participant_a/reliability.py`；`models/osnet_ain_x1_0_dg.pth`、`models/gaitgraph2_grew_state.pt` | 外观与步态向量、开放集拒识、步态锚定、自动建号条件、多人一对一指派、阈值校准、候选模型 A/B 报告 | 接收参与者 B 的 Track/姿态/质量；向参与者 C 提供稳定的 `Decision`、运行参数和审计语义 |
| 参与者 B | 感知与轨迹 | `participant_b/vision.py`、`participant_b/vision_factory.py`、`participant_b/adapters.py`、`participant_b/production_vision.py` 中的检测/跟踪/RTMPose/质量路径；`models/yolo11x.pt`、`models/bytetrack-cross-event.yaml`、`models/rtmpose-s.onnx` | 稳定的人物框和 Track ID、17 点姿态、出框/遮挡判断、`TrackQuality`、检测与跟踪延迟基线 | 向参与者 A 提供稳定的 Track/姿态/质量输入；向参与者 C 提供可启动的 `VisionAdapter` 和后端状态 |
| 参与者 C | 在线系统、数据与交付 | `participant_c/pipeline.py`、`participant_c/automation.py`、`participant_c/runtime_parameters.py`、`participant_c/media.py`、`participant_c/gui.py`、`participant_c/cli.py`、`participant_c/storage.py`、`participant_c/vector_index.py`、`participant_c/model_assets.py`、`tests/`、`scripts/`、研究文档和 `models/manifest.json` | 摄像头/视频链路、实时参数页、线程安全、SQLite/索引集成、模型同步、回归测试和可复现实验流程 | 接入参与者 B 的 `VisionAdapter`；调用参与者 A 的验证器接口；维护端到端测试和发布检查 |

`participant_b/production_vision.py` 当前同时包含感知编排和特征提取调用。B 只修改检测/跟踪/姿态/质量区域，A 只修改 `participant_a/` 中的 ReID/步态编码实现；如需修改该编排文件中的跨 seam 调用，必须先互相评审。

### 分工边界

- 参与者 B 决定“画面中有哪些可靠的人和姿态”，但不在感知模块中决定身份 ID。
- 参与者 A 决定“特征代表谁、证据是否足够、何时拒识或建号”，但不在验证器中实现检测器逻辑。
- 参与者 C 决定“如何把各模块安全地串成可操作的软件”，但不绕过 A/B 的 seam 修改身份状态或视觉特征。
- 三人共同维护公开值对象和回归测试；修改 `VisionTrack`、`FeatureBundle`、`TrackQuality`、`Decision`、SQLite schema 或模型输出维度时，必须先通知另外两人。

### 交接顺序

```text
参与者 B：检测框 / Track ID / RTMPose / TrackQuality
        ↓
参与者 A：OSNet / GaitGraph2 / FeatureBundle / Decision / calibration
        ↓
参与者 C：FrameResult / GUI / storage / release checks
```

跨线任务由主责参与者负责最终合并，协作者只提交接口所需的最小变更；检测器/姿态替换由 B 主导，ReID/步态模型和阈值替换由 A 主导，数据库迁移和发布流程由 C 主导，三类变更都需要另外两人复核。

## 开始工作前

1. 先读本目录的模块地图，确认改动所属模块组和 seam。
2. 运行一次基线：

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   .\.venv\Scripts\python.exe -m compileall -q cross_event_verifier tests
   ```

3. 如果是生产视觉改动，再运行：

   ```powershell
   .\.venv\Scripts\python.exe -m cross_event_verifier doctor
   ```

## 变更规则

- 一个提交尽量只跨一个模块组；必须跨组时，在提交说明中写出依赖原因。
- 修改公共值对象、阈值或数据库 schema 时，先补回归测试，再改实现。
- 模型替换必须同时提交模型版本、输入格式、输出维度、权重哈希和许可说明。
- 新增运行时参数必须有范围、跨字段约束、默认值和 GUI/非 GUI 测试。
- 不直接在 GUI 中调用内部存储或修改候选状态；通过 pipeline/engine seam。
- 不把真实视频、个人身份数据、私有下载 URL 或密钥提交到仓库。
- 不把大模型文件加入 Git；使用 `models/manifest.json` 和同步脚本。

## 实验记录最低要求

每个模型或阈值实验至少记录：

- 数据集/摄像头、分辨率、视角、遮挡和换衣条件；
- 模型版本、权重 SHA-256、输入帧数和输出维度；
- FAR/FPIR、FRR、Rank-1 或项目实际使用的指标；
- GPU、批量大小、端到端延迟；
- 是否改变了默认阈值、图库迁移或审计格式。

没有这些信息的“准确率提升”不能直接替换生产默认值。

## 合并前检查

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q cross_event_verifier tests
.\.venv\Scripts\python.exe -m cross_event_verifier demo
.\.venv\Scripts\python.exe -m cross_event_verifier doctor
git diff --check
```

如果只改文档，可以跳过模型 `doctor`，但仍应检查 Markdown 中的命令、路径和模块名称。
