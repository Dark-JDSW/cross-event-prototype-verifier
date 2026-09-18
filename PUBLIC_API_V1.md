# Public API V1 与可选上下文接入

规范见根目录 [AGENTS.md](AGENTS.md)，名称与领域字段清单见
[public_api_v1.json](tests/contracts/public_api_v1.json)。清单只做向后兼容子集检查，
允许新字段、新 enum 值、新可选 API；不冻结整份函数 signature。

原有 `CrossEventVerifier`、`VisionAdapter`、`FeatureExtractor`、`ActionPublisher`、
embedding/storage 边界继续使用。新增模块不自动加入 pipeline、不修改默认配置，
无需为了 Edge 或 HyperGait 再建一层 factory。

```python
from cross_event_verifier.contexts import ContextComposer, DeviceContext
from cross_event_verifier.spatial import VisionMetadataSpatialProvider
from cross_event_verifier.interaction import ContextRule, InteractionEvent
from cross_event_verifier.actions import ActionType

# observation 与 track 来自同一 camera/session，action 已由该 Decision 授权。
decision = verifier.verify(observation)
spatial = VisionMetadataSpatialProvider().locate(
    track, camera_id=observation.camera_id, timestamp=observation.timestamp,
    capture_session_id=observation.capture_session_id,
)
device = DeviceContext(
    camera_id=observation.camera_id, track_id=track.track_id,
    timestamp=observation.timestamp, capture_session_id=observation.capture_session_id,
    zone_id="bedroom", nearest_device_id="bedroom_light",
)
context = ContextComposer().compose(
    decision, action, track_id=track.track_id, camera_id=observation.camera_id,
    timestamp=observation.timestamp, capture_session_id=observation.capture_session_id,
    spatial=spatial, device=device,
)
policy = ContextRule(identity_id="P1", action=ActionType.WAVE, zone_id="bedroom",
                     device_id="bedroom_light", operation="toggle")
for command in policy.evaluate(InteractionEvent(context=context)):
    executor.execute(command)
```

示例中的设备/区域事实由部署方配置或 provider 提供，不是 Composer 推断。
调用者负责把同一帧/轨迹的 Decision、ActionEvent、Observation 关联起来：现有
Decision 没有轨迹字段，Composer 不能凭空恢复该关系。跨会话必须提供 session ID。
原始 VisionTrack 可含可变数组；新 Provider 仅接收 ContextRequest，不接收原型或图库。

`ContextContribution` 可表达任意可选 Provider 的输出。实验字段使用 namespaced
attributes，例如 `occupancy.room_count`、`environment.facing_tv`。它们仅接受有限
JSON 值并递归复制冻结。重复字段同值可合并、异值整条拒绝；失败输出 errors，
默认 ContextRule 不执行任何命令。可选 provider 失败不会触碰之前完成的核心事务。

`SpatialContext.world_position` 使用 `(world_x, world_y)`；速度是 `(vx, vy)`。
旧 metadata 的 `ground_speed` 是标量速度，`world_direction_degrees` 是运动方向，
适配器保留原名字放入 attributes，不将它们误当成向量速度或身体朝向。
轨迹和 camera transition 同样可以放入 attributes；已有 Homography/GCP 计算不迁移。

`IdentityPriorProvider` 是未来接口预留，当前没有执行接线。旧摄像头先验不变。
Edge 复用 VisionAdapter/worker，HyperGait 复用 FeatureBundle/Observation 与完整
embedding contract；本次不提供新模型、TensorRT runtime 或设备执行实现。

## 已有身份编排的边界

生产 appearance-first 的视觉身份建号、轨迹绑定、gait readiness 是现有行为，不能
为冻结接口而替换成单一 gait-first 算法。身份编排仍调用 Core 的公开 matching 和
enrollment 接口；两处视觉身份 Decision 的构造已收回 Core 内部，字段和值保持一致。
该整理不是新的插件身份入口。Interaction/Spatial 不得调用内部决策构造方法。

## 兼容性验证

契约测试冻结旧包导出、公开成员、领域字段和 enum 值，使用代表性旧调用绑定测试。
新领域可选字段必须有默认值。Action V1 使用旧字段语义断言，Interaction V1 是单独
envelope，不改变旧 topic 或 publisher。测试还检查完整数据库快照、原型向量与
gait readiness 在上下文处理成功/失败前后不变。
