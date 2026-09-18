# Public API V1 与多分支工程规则

适用范围：整个仓库及 main / interaction / spatial / edge / HyperGait 分支。
目标：stable outside, flexible inside。执行 **Prefer extension over modification**、
**Allow composition, prohibit reverse invasion**。

## 1. Identity Decision Boundary

- 正式识别数据流：Camera/Video → VisionAdapter → VisionTrack → Observation → CrossEventVerifier → Decision。
- Identity Core 拥有身份分数、开放集门、formal/quarantine、晋升、回滚与 gait readiness；禁止分支实现第二套 verifier、状态机或图库规则。
- `automation.py`、`appearance_first.py` 是现有身份采集/绑定编排，调用 Core 的验证、匹配和建号接口；不得将其复制为 Interaction 的身份判断器。
- 保留现有 appearance-first 视觉身份与 gait-ready 的区别，以及旧 gait-first API 行为。视觉身份已确认不等于 gait-ready。
- 已绑定视觉身份的决策构造由 Core 的内部 `_visual_identity_decision` 承担；该内部入口仅供身份编排使用，不是插件接口。
- 下游只消费已经返回的 Decision；不得构造、替换或提升正式身份结论，不得持有或写入 verifier.memory/store。
- `register_identity` 是现有可信初始化入口；`promote_candidate` 是 quarantine → formal 的正式受控边界，保留人工 force 的显式审计语义。

## 2. Stable Public API

- Public API V1 名称基线在 `tests/contracts/public_api_v1.json`：包级 `__all__`、CrossEventVerifier 全部既有公开成员、共享字段和 enum 值。
- 必须保留 `verify`、`verify_batch`、`register_identity`、`promote_candidate`、`formal_identities`（属性）、候选查询、匹配、建号、readiness、maintenance、rollback、close 等现有入口。
- 不得删除、直接重命名、改变旧位置参数/关键字调用方式、返回语义或身份规则。新增参数优先 keyword-only 且提供默认值。
- 允许新增可选 API。不要 snapshot 完整 signature；测试真实旧调用形状和必要参数语义。破坏性演进必须另建版本并提供迁移期兼容入口。
- 不将 package version `0.1.0`、Public API V1、模型 embedding 版本或外部事件协议版本混为一谈。

## 3. Shared Domain Contracts

- `Observation`、`Decision`、`FeatureBundle`、`TrackQuality`、`VisionTrack`、`ActionEvent`、既有身份状态 enum/dataclass 属于共享契约。
- 新字段必须有安全默认值，保留旧字段顺序和语义；允许 optional 字段、enum 新值、metadata 和辅助类型。
- 禁止将可选字段改成必填，禁止把 HyperGait、TensorRT、Home Assistant 客户端或实现类放入领域对象。
- 核心既有对象可能包含可变数组/字典，不能把 dataclass frozen 当作深层不可变保证。跨插件边界传递复制后的只读事实。
- 实验上下文优先放入独立 Context 的 namespaced attributes/metadata，不无限扩展 ActionEvent 或 types.py。

## 4. Vision Boundary

- 保留 `VisionAdapter.process(frame_bgr) -> tuple[VisionTrack, ...]` 与 `reset() -> None`；继续支持现有可选 `process_at`。
- 模型和硬件实现位于 adapter/backend 内，上层不得按 YOLO、ByteTrack、OSNet、GaitGraph2、HyperGait、TensorRT 实现复制身份逻辑。
- `VisionTrack` 和 `VisionAdapter` 必须能在 NumPy-only 基础安装导入；OpenCV 等在具体 runtime 使用时加载。
- 模型替换传递既有 model/embedding 元数据；不能因为 Python 接口相同就沿用旧模型标签。

## 5. Embedding Compatibility

- **API compatibility != embedding compatibility**。不同 gait encoder 默认不兼容，包括同维向量。
- 保留 `model_version / feature_schema / artifact_sha256 / preprocess_version / joint_format / sequence_length / tta_mode / coordinate_contract / embedding_dimensions`；`EmbeddingContract.dimensions` 是其既有值对象表示。
- 继续使用 Core、Memory、Storage 现有协议过滤、图库隔离、持久化和晋升预检；不得跳过过滤直接比较向量。
- 禁止静默混合 GaitGraph2 与 HyperGait。任何兼容映射必须有明确验证、版本化迁移和回归测试；未知协议不能作为跨模型通配符。
- 模型变化配套独立图库/受控迁移，保留旧库审计信息，不将旧原型重标为新模型。

## 6. HyperGait Rules

- 可自由重构 encoder、预处理、时序缓存、维度、batching 和内部 pipeline。
- 输出继续通过 backend/adapter → FeatureBundle → Observation → CrossEventVerifier。
- 不建立 HyperGaitVerifier，不复制 candidate promotion/open-set/gallery 逻辑。
- production runtime 和模型资产均为 optional；替换后更新完整 embedding 契约并运行不兼容模型测试。

## 7. Interaction Rules

- 基础输入为 Decision + ActionEvent；gait 是 identity biometric，action 是 interaction signal。
- WAVE、SQUAT、SIT_DOWN、STAND_UP、HAND_UP 等短动作默认只驻留短期内存；不得据此修改正式外观/步态原型、身份状态或 gait readiness。
- 保留现有 WALK 动作质量门，它是步态证据有效性筛选，不是设备交互规则。
- `ActionEvent` 保持原子事实。`cevp.action.v1` 字段含义、ActionPublisher.publish/close 和旧 ESP32/MQTT 消费方式保持兼容。
- 复杂事件使用独立 `cevp.interaction.v1`；禁止在旧 action topic 上静默替换载荷。

## 8. Contextual Interaction

- 数据流：Decision + ActionEvent + SpatialContext + DeviceContext → ContextComposer → InteractionContext → InteractionEvent → InteractionPolicy → InteractionCommand → CommandExecutor。
- 复用 `contexts.py` 的 ContextProvider；Spatial/Device/Camera/Occupancy/Environment 提供者实现相同协议，不要求各建一套共享模型。
- Producer produces facts; Composer combines facts; Policy makes decisions。Composer 不计算 Homography、世界坐标、身份、动作，也不调用设备 API。
- Provider 只接收 ContextRequest 的只读身份、动作和关联信息，不注入 Core/Memory/Store。插件是受信任代码，Protocol 不是恶意代码沙箱。
- 必须校验 camera_id、track_id、capture_session_id、timestamp；接入方在跨会话轨迹可能复用时必须提供 session。默认允许事实年龄 1 秒，拒绝未来/过期事实。
- 合并冲突拒绝整条 contribution，不静默覆盖；provider 异常记录在 errors。自动执行策略遇到 errors、未确认身份或不匹配动作必须不输出命令。
- 世界坐标/距离使用米，速度使用米/秒，并保留 coordinate_frame/calibration_id；不同坐标系不能静默组合。图像 camera_zone 不等于房间，运动方向不等于身体朝向。
- 通用位置、区域、速度、朝向、最近设备保留稳定字段；trajectory_state、facing_tv、room_occupancy、dwell_time、camera_transition 等用 namespaced attributes/metadata。
- 会话历史可放 Interaction 内存状态，必须有容量/TTL、轨迹重置和身份切换清理；不得进入 persistent identity memory。
- 新层默认不挂接 pipeline。应用在 Core 事务完成后显式调用 Composer/Policy/Executor；设备故障不得回滚身份决策。设备重试/去重属于执行层。

## 9. Spatial Rules

- 可选流：VisionTrack → SpatialProvider.locate → SpatialContext。无 Homography/GCP/拓扑时可返回 None。
- `VisionMetadataSpatialProvider` 只适配现有 production metadata；不要重写已有 ground_control/production_vision 几何算法。
- 无 spatial/provider/world coordinates 时 Core 必须能照常 verify；不得给 verify 增加必填 SpatialContext。
- Spatial 与 Interaction 仅共享值对象和 Provider 协议；不得互相 import 具体实现、私有状态或设备 SDK。

## 10. Optional Spatial Prior

- `IdentityPriorProvider` 当前仅预留协议，未接入评分；默认无 provider、无新先验效果。
- 未来接入必须显式可关闭、权重默认 0、有界且有审计；provider 只能返回弱证据，不能返回身份 Decision。
- 必须保持 gait 质量、embedding、开放集、margin、晋升门；“P1 刚在附近”不能单独认成 P1。
- 既有 `spatial_prior_weight=0.10` 和 Core 摄像头转场逻辑是旧 baseline，本次不得借架构改造改变。新增 provider 不得隐式叠加该权重。

## 11. Edge Rules

- 允许 ONNX/TensorRT、FP16/INT8、batching、跳帧、缓存、异步 worker、硬件和生命周期优化。
- 优先复用 VisionAdapter、现有 process_at、media worker 和 runtime 参数边界；实际需要时才新增 Scheduler/InferencePolicy/RuntimeBackend。
- 不复制身份状态机、图库、晋升、开放集。上层继续消费 VisionTrack/Observation/Decision。
- 无 scheduler/runtime 插件是合法部署；异步实现必须保留时间戳、会话、轨迹重置与结果顺序语义。

## 12. Optional Dependency Rules

- 基础依赖仍仅 NumPy。基础 import/verify 不要求 MQTT、BLE、Home Assistant、OpenCV、Torch、HyperGait、TensorRT 或 GPU。
- 新配置必须有默认值；新 feature 默认关闭/中性，不要求旧配置补字段。
- SDK 使用延迟导入和 optional extras；未启用时不连网、不加载模型、不创建 worker。
- 禁止在包 __init__ 导入时探测硬件、创建客户端或进行模型初始化。

## 13. Merge Safety

- HIGH-CONFLICT SHARED FILES：engine.py、types.py、pipeline.py、vision.py、config.py、__init__.py、storage.py、memory.py。
- 改动前先检查 new module / Protocol / Adapter / Provider / Publisher / Strategy / dependency injection 能否满足需求。
- 必须修改共享文件时记录原因，保持最小 additive diff 并添加 regression test。不要借功能分支批量格式化这些文件。
- 在新模块暴露新能力，不为每个分支扩张包级 re-export；保留既有导入路径。
- 不覆盖工作区已有修改，不顺手改 GUI、模型、识别阈值或状态机。跨分支读取上下文合法，反向写身份核心非法。
- 合并时检查旧调用形状、enum/字段、默认配置、模型契约、外部协议和存储迁移；禁止通过重生成/缩减 V1 清单掩盖破坏性变更。

## 14. Required Tests

- 必须执行 `python -m pytest`；Windows 项目环境可用 `.\.venv\Scripts\python.exe -m pytest`。
- `tests/contracts/` 必须保留 public_api、public_signatures、domain_contracts、import_contract、base_smoke、optional_features、context_isolation 测试。
- 验证旧 API/构造方式、正式/隔离晋升、模型不兼容、无新 runtime 的 import/verify、上下文组合、错误隔离、action 协议及原型/readiness 不变。
- 共享文件修改额外运行相关 core/appearance-first/vision/pipeline 回归测试；完整 suite 已覆盖时无需重复。
- 可行时执行 `python -m pip install -e .`，在只装基础依赖的新环境验证 import 和 verify。
- 运行 `git diff --check`。禁止删除/弱化断言或跳过失败来冒充通过；分别报告代码回归和环境失败。
