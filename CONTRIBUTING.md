# 共同研究与开发

本项目按模块组协作维护。开始分工前请阅读 [模块地图](docs/architecture/module-map.md) 和 [协作约定](docs/architecture/contributor-guide.md)。

核心原则：在稳定 seam 上协作，让复杂行为留在深模块内部；跨模块调用优先使用 `CrossEventVerifier`、`VisionAdapter`、`VideoVerifierPipeline` 和 `SqliteStore` 的接口，不在调用方复制实现。
