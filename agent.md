# Agent 环境入口

所有 agent 必须阅读并遵守 [AGENTS.md](AGENTS.md)，其中包含公共 API 冻结规则和执行环境要求。

Windows 环境优先使用 Git Bash 执行部署、构建、测试和代码修改；Git Bash 未安装或不可用时，先安装/修复 Git for Windows。Linux 环境使用 Linux 原生 shell。不要把 C:\Windows\System32\bash.exe 当作 Git Bash；本机 WSL2 不再是默认工作区。

依赖、虚拟环境、路径大小写、环境变量和命令示例见 [开发环境指引](docs/DEVELOPMENT_ENVIRONMENTS.md)。Linux 文件名区分大小写：本文件 agent.md 不能替代自动加载的 AGENTS.md。
