# 模块地图

## 总体依赖方向

```text
接口层（GUI / CLI）
        ↓
运行编排层（媒体、帧管线、自动注册、运行时参数）
        ↓                 ↓
决策层（验证、校准、指派）  视觉适配层（检测、姿态、ReID、步态）
        ↓          ↘       ↓
领域与证据层      持久化层  （都只依赖稳定值对象）
（值对象、规则、状态、记忆）
```

依赖方向是维护约束，不要求每个文件严格位于一层。尤其是 `engine.py` 作为验证器深模块，会协调领域规则和持久化适配器；调用者不应复制其中的判定流程。领域与证据层不应反向依赖 GUI、视觉模型或 SQLite。

## 分工表

| 模块组 | 当前文件 | 稳定 seam（接口接缝） | 主要职责 | 测试入口 |
|---|---|---|---|---|
| 参与者 A：身份与证据 | `participant_a/` 下的 `gait_graph.py`、`osnet_ain.py`、`engine.py`、`assignment.py`、`calibration.py`、`fusion.py`、`evaluation.py`、`reliability.py`、`config.py`、`state.py`、`challenge.py`、`absorption.py`、`memory.py`、`stability.py` | `CrossEventVerifier.verify()`、`verify_batch()`、`enroll_gait_identity()` | 外观 ReID、步态编码、开放集拒识、步态锚定、自动建号、校准、融合、多人一对一指派 | `tests/test_core.py`、`tests/test_numeric.py`、`tests/test_production_vision.py` |
| 参与者 B：感知与轨迹 | `participant_b/` 下的 `production_vision.py`、`vision.py`、`vision_factory.py`、`adapters.py` | `VisionAdapter.process/reset`、`VisionTrack`、`build_vision_adapter()` | YOLO/ByteTrack、RTMPose、出框/遮挡判断、`TrackQuality` 和视觉后端选择；生产编排中调用 A 的特征编码器 | `tests/test_adapters.py`、`tests/test_production_vision.py` |
| 参与者 C：在线与交付 | `participant_c/` 下的 `pipeline.py`、`automation.py`、`runtime_parameters.py`、`media.py`、`gui.py`、`cli.py`、`storage.py`、`vector_index.py`、`model_assets.py` | `VideoVerifierPipeline.process_frame()`、`FrameResult`、`FrameWorker`、`RuntimeParameterController`、`SqliteStore` | 帧级编排、自动建号触发、GUI、热参数事务、摄像头线程、SQLite、向量索引、模型同步和发布入口 | `tests/test_gui_pipeline.py`、`tests/test_core.py`、`tests/test_model_assets.py` |
| 研究与交付 | `tests/`、`models/`、`scripts/`、`README.md`、`THIRD_PARTY_NOTICES.md` | manifest、回归命令、实验记录 | 公开数据/模型资产、回归套件、模型同步、许可和研究文档 | `python -m unittest discover -s tests -v` |

## 各组的修改边界

### 领域与证据

这里定义“什么证据算有效”。任何质量门、候选状态或原型写入规则的修改，都必须说明误识、拒识和污染图库的影响，并补一个最小回归测试。不要在 GUI 或视觉适配器中复制质量计算。

### 验证与决策

这里定义“有效证据如何变成决定”。`CrossEventVerifier` 是深模块：GUI、自动注册和实验脚本调用它，不应重新实现 `verify()` 的分支判断。校准器的分数不是原始 cosine，相同模型更换后必须重新校准。

### 视觉与模型

这里定义“如何产生 FeatureBundle 和 TrackQuality”。模型替换优先实现一个新的 `VisionAdapter` 或模型内部 adapter，不要让 `engine.py` 直接导入 YOLO、ONNX Runtime 或具体 checkpoint。模型二进制只通过 `models/manifest.json` 管理。

### 运行编排

这里负责时序和线程，不负责发明新的身份语义。参数更新必须走 `RuntimeParameterController` 的整组校验和帧边界提交；自动建号的 gait 样本窗口不能和 GUI 临时状态混用。

### 持久化与索引

这里负责数据可恢复性。SQLite schema、审计事件、快照格式和向量维度迁移要保持向后兼容；任何删除/重写都要先增加回滚测试。不要把模型推理逻辑放进存储层。

### 用户接口

GUI 和 CLI 只负责输入、展示和调用深模块。GUI 不应直接修改 `memory`、`storage` 或候选状态；新增控件要优先接入已有 pipeline/runtime 参数 seam。

## 公开接口速查

```python
from cross_event_verifier import (
    CrossEventVerifier,
    FeatureBundle,
    Observation,
    TrackQuality,
)

verifier = CrossEventVerifier()
decision = verifier.verify(observation)
batch = verifier.verify_batch(observations)
enrolled = verifier.enroll_gait_identity(observation, identity_id="P001")
```

GUI/视觉组只需知道：

```python
tracks = vision.process(frame_bgr)
result = pipeline.process_frame(frame_bgr)
state = pipeline.update_runtime_parameters(values)
```

## 物理拆包与包根目录规则

三个参与者子包是当前实现的规范位置。包根目录只保留公开门面、共享值对象和 Python 包入口，不再放置参与者实现或旧模块兼容转发层。

1. 新功能提交到对应参与者子包，不直接编辑包根目录入口或共享值对象。
2. 跨参与者调用通过 `types.py`、`VisionAdapter`、`CrossEventVerifier`、`VideoVerifierPipeline` 等稳定 seam。
3. 修改公共值对象或模型输出维度时，必须补回归测试并通知另外两位参与者。
4. 外部实验脚本应使用参与者子包路径或公开包门面，不再依赖旧的平铺模块路径。
