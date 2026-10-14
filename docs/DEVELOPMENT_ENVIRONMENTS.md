# 开发环境指引：Windows Git Bash 与 Linux

本仓库按操作系统选择原生命令环境：Windows 优先且尽可能使用 Git Bash，Linux 使用 Linux 原生 shell。WSL2 不再是默认工作区；此前位于 WSL Linux 文件系统中的项目副本已按用户要求删除，Ubuntu/WSL 发行版本身不在本指引中管理。

## Windows：Git Bash

所有项目部署、构建、测试和代码修改优先从 Git Bash 执行。推荐入口是 C:\Program Files\Git\bin\bash.exe。先确认：

```bash
bash --version
git --version
uname -s
pwd
./.venv/Scripts/python.exe -V
```

uname -s 应显示 MINGW64_NT... 或 MSYS_NT...。C:\Windows\System32\bash.exe 是 WSL 启动器，不能代替 Git Bash。

如果 Git Bash 未安装或配置异常，先从 Git for Windows 官方页面安装/修复：https://git-scm.com/download/win。也可在 Windows 引导终端用 winget install --id Git.Git -e 启动官方包安装。PowerShell 只用于这类必要的安装引导；安装完成后打开 Git Bash，重新验证 Bash、Git 与 PATH，再执行项目操作。不要在 Git Bash 与 PowerShell 间混用同一个未激活的环境。

在 Git Bash 中创建和使用 Windows 虚拟环境：

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest
```

Windows 的 Python launcher 也可使用 py -3.10 或项目要求的其他受支持版本。若已有虚拟环境，核验 ./.venv/Scripts/python.exe -V 后再安装依赖。

## Linux：原生 shell

Linux 上使用发行版原生终端、Python、包管理器和 .venv/bin/python：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pytest
```

通过发行版包管理器安装创建虚拟环境所需的系统包，例如 Debian/Ubuntu 的 python3-venv；不替换系统 Python。不同 Linux 发行版的包名和系统依赖可能不同。

## Windows 与 Linux 的依赖和行为差异

| 项目 | Windows + Git Bash | Linux 原生 |
| --- | --- | --- |
| Python 虚拟环境解释器 | .venv/Scripts/python.exe | .venv/bin/python |
| 创建虚拟环境 | py -3.12 -m venv .venv | python3 -m venv .venv |
| 激活环境 | source .venv/Scripts/activate | source .venv/bin/activate |
| 路径 | Bash 中常见 /c/Users/name/project；Windows API/原生工具可能需要 C:\... | /home/name/project 等 POSIX 路径 |
| 大小写 | 默认文件系统通常大小写不敏感，但目录可启用大小写敏感；不能依赖大小写混写 | 通常大小写敏感；Module.py 与 module.py 是不同路径 |
| 环境变量 | 当前 Bash 会话用 export NAME=value；Windows 原生进程还可能读取 Windows 用户/系统环境 | 当前 shell 用 export NAME=value；服务管理器/桌面启动器有独立环境配置 |
| 搜索路径 | Git Bash 的 $PATH 使用冒号分隔并转换 Windows PATH；Windows Python os.pathsep 为分号 | $PATH 与 Python os.pathsep 通常为冒号 |
| 二进制依赖 | 安装 Windows 对应 wheel/DLL；CUDA、ONNX Runtime、OpenCV 与 Python ABI 需匹配 Windows 驱动/运行时 | 安装 Linux wheel/共享库及发行版系统依赖；CUDA provider、驱动和动态链接库需匹配 Linux 环境 |
| 文件属性 | NTFS 权限、符号链接和行尾行为由 Git/Windows 配置共同影响 | POSIX 权限、符号链接和可执行位生效；文件名区分大小写 |

两个平台必须分别创建 .venv，不得复制虚拟环境目录或把另一平台的二进制轮子带入当前环境。依赖通过仓库 pyproject.toml、extras 和 requirements.txt 声明安装；带 GPU 的 PyTorch/ONNX 等应按平台、Python 版本和驱动能力选择兼容发行包。不要仅因包名相同就假设 wheel、CUDA provider 或原生库可跨平台复用。

环境变量在 Git Bash/Linux shell 中写为 export NAME="value"，仅对当前 shell 及其子进程生效。要让 GUI 读取变量，应从设置变量的同一 Git Bash/终端启动；服务或桌面快捷方式需要按各自启动机制单独配置。Windows 的 $env:NAME=value 是 PowerShell 语法，不应粘贴进 Bash。

路径应通过 pathlib 或操作系统 API 组合；避免硬编码分隔符、只在 Windows 成立的大小写假设或 shell 特有展开。提交 shell 脚本使用 LF。跨平台环境差异不能成为修改 Public API、身份阈值、模型契约或 SQLite schema 的理由。

## WSL 状态

本机原 WSL 项目副本 /home/wzige/cross-event-prototype-verifier 已删除。不要将其视作当前工作区，也不要从 Windows 与 WSL 两侧并行修改项目。若未来要重新启用 WSL，需由用户明确提出新的环境迁移任务。

## IDE

Windows IDE 应打开当前 Windows 项目目录，并将解释器指向该目录下的 .venv/Scripts/python.exe。Linux IDE 应打开 Linux 项目目录并选择该目录下的 .venv/bin/python。不要将一侧的虚拟环境配置指向另一侧项目文件系统。
