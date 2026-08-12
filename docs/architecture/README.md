# 架构与协作

本目录是组内共同研究和分工的入口。

- [模块地图](module-map.md)：每个模块组负责什么、依赖什么、从哪里测试。
- [协作约定](contributor-guide.md)：如何分工、提交实验、修改 seam 和合并变更。

源码已经按参与者拆分到三个子包：

- `cross_event_verifier/participant_a/`：ReID、GaitGraph2、身份判定与证据规则；
- `cross_event_verifier/participant_b/`：YOLO、ByteTrack、RTMPose、视觉质量和适配器；
- `cross_event_verifier/participant_c/`：pipeline、GUI、运行时参数、存储、CLI 和交付工具。

`cross_event_verifier/types.py` 保留为 A/B/C 共用的公开契约。新代码和实验脚本应直接从对应参与者子包导入，例如
`cross_event_verifier.participant_a.engine` 或 `cross_event_verifier.participant_b.production_vision`；顶层
`from cross_event_verifier import CrossEventVerifier` 这一公开门面仍然保持不变。
