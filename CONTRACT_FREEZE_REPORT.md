# 公共契约冻结检查报告

本次保留识别算法、阈值、模型、图库规则与默认配置。新增能力不自动接入旧管线。
规范：[AGENTS.md](AGENTS.md)。接口与接入示例：[PUBLIC_API_V1.md](PUBLIC_API_V1.md)。

| # | 检查项 | 交付结果 |
|---|---|---|
| 1 | Public API V1 | 冻结既有包级导出、CrossEventVerifier 全部公开成员和共享领域字段/enum；名称基线在 tests/contracts/public_api_v1.json。保护 verify/verify_batch/register_identity/promote_candidate、formal_identities 属性、匹配/建号/readiness/maintenance/rollback/close 等旧调用。保留 VisionAdapter 与 ActionPublisher 边界。 |
| 2 | Extension points | 新增 ContextProvider、SpatialProvider、InteractionPolicy、CommandExecutor；IdentityPriorProvider 仅预留。复用 VisionAdapter、FeatureExtractor、ActionPublisher、现有 worker/runtime 参数，不增设无必要的 Edge factory。 |
| 3 | Context 机制 | 新增 ContextRequest、ContextFacts、ContextContribution、SpatialContext、DeviceContext、InteractionContext、ContextComposer。JSON-shaped 扩展数据递归复制冻结。 |
| 4 | Spatial/Interaction 共享 | 通过值对象和 Provider 共享只读事实；Composer 校验 camera/track/session/time，原子拒绝冲突，捕获 provider 错误。Policy 消费组合结果，Executor 执行设备命令。 |
| 5 | Spatial 可关闭 | 是；不调用 provider 或返回 None 即可，verify 不需要 SpatialContext。已有 Homography 仅通过 metadata 适配器复用。 |
| 6 | Interaction 可关闭 | 是；新增模块未自动挂接 pipeline。旧短时动作行为保持原样；Base Core 不需要 Interaction 或设备连接。 |
| 7 | 可选身份弱先验 | 提供 IdentityPriorProvider 协议，当前未接入评分。未来要求显式启用、默认权重 0、保留全部生物特征/开放集门。旧 spatial_prior_weight=0.10 保持 baseline。 |
| 8 | HyperGait | 可通过现有 adapter/backend 输出 FeatureBundle/Observation；本次不实现新模型。完整 embedding contract 隔离模型；同维 GaitGraph2/HyperGait 不得静默比较。 |
| 9 | Edge | 可复用 VisionAdapter/process_at/worker 改变内部 runtime，继续输出共享语义；本次不实现调度器或硬件 runtime，不复制身份逻辑。 |
| 10 | ActionEvent / action V1 | 字段和语义未变；新增 cevp.interaction.v1 独立 envelope，旧 MQTT/BLE publisher 不变。 |
| 11 | 强制依赖 | 无新增；基础仍仅 NumPy。OpenCV 改为具体 demo adapter 初始化时导入，修复基础包间接依赖 OpenCV。 |
| 12 | Shared files | 本次修改 engine.py：将既有两处视觉身份 Decision 构造集中到 Core，字段/判断条件不变；vision.py：延迟 OpenCV 导入。另改 appearance_first.py 两处委托调用。types/pipeline/config/__init__/storage/memory 未被本次修改；用户原有 memory.py 改动保留。 |
| 13 | 测试结果 | 最终 python -m pytest：227 passed，18.42 秒；含 49 项新增契约测试。pip install -e . 成功；全新基础环境仅装项目+NumPy，确认 cv2/torch/paho/bleak/homeassistant/tensorrt/onnxruntime/hypergait 未安装，import+verify 成功。git diff --check 与新增文件空白检查通过。 |
| 14 | 剩余 merge 风险 | engine/pipeline/production_vision/appearance_first/runtime_parameters 仍是复杂交汇点；所有既定 shared files 仍需最小修改与回归保护。新 provider 应在独立模块实现。未来 pipeline 自动接线、先验启用、模型图库迁移和实际设备执行仍需专门变更，不隐式启用。 |

## 测试环境记录

- 系统默认 Python 未安装 pytest；完整 suite 使用仓库 `.venv`，通过临时 PATH 选择该解释器执行 `python -m pytest`。Windows Store Python 的 GUI 测试还需要把 `TCL_LIBRARY` 指向 `...\tcl\tcl8.6`、`TK_LIBRARY` 指向 `...\tcl\tk8.6`；这只修正测试运行环境，不改变源码或测试。
- 基线首次运行 176 passed / 2 failed，失败为本机 Tcl/Tk 资源加载；独立 Tk 初始化成功，后续完整运行全部通过，未修改或跳过 GUI 测试。
- 最终 227 项全部通过；没有删除、弱化或重写既有测试。
- 兼容保证基于现有回归与新增契约测试；本次未重新评估真实视频识别精度，也未运行实际 HyperGait/TensorRT/设备 SDK。

## 交付文件

- `cross_event_verifier/contexts.py`：事实、Provider、Composer、关联/时效/冲突处理。
- `cross_event_verifier/spatial.py`：SpatialProvider、metadata 适配、ContextProvider 适配及先验预留。
- `cross_event_verifier/interaction.py`：事件序列化、policy、command、executor 协议与默认不执行的 ContextRule。
- `tests/contracts/`：七个要求的测试文件、Public API V1 清单；共 49 项测试。
- `AGENTS.md`：14 类可执行工程规则；`PUBLIC_API_V1.md`：接入代码与边界说明。

本次未改动用户预先存在的 `.idea/claudeCodeTabState.xml`、README、memory.py、
tests/test_core.py 和 firmware 文档修改；未执行 git commit 或覆盖这些内容。
