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

## Linux 生产运行准备（Ubuntu/Debian 示例）

本节对应仓库当前的生产视觉链路（YOLO、RTMPose、OSNet、GPGait++）、摄像头采集、Bleak BLE 和 SQLite 存储。包名和设备权限在其他发行版可能不同；先核对本机实际设备与发行版文档，不要把 Windows DLL、COM 端口或虚拟环境复制过来。

### NVIDIA CUDA 与 ONNX Runtime

当前 `requirements.txt` 固定 PyTorch `2.12.1+cu132`、Torchvision `0.27.1+cu132`，Linux 上安装 ONNX Runtime GPU `>=1.27,<1.28`。ONNX Runtime 1.27 的 PyPI GPU wheel 使用 CUDA 13.0 / cuDNN 9，并支持 CUDA 13.x 对应的 PyTorch；主机 NVIDIA 驱动、CUDA 动态库和 cuDNN 必须匹配。请先按 [NVIDIA Linux CUDA 安装指南](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-linux/) 为当前发行版安装兼容的 NVIDIA 驱动，再按本项目锁定依赖安装。预编译 pip wheel 场景通常不需要为编译而安装完整 CUDA Toolkit；只有要编译 CUDA 扩展时才需要相应 Toolkit、编译器和 `CUDA_HOME`。

```bash
nvidia-smi
python -m pip install -r requirements.txt
python -c 'import torch, onnxruntime as ort; print("torch", torch.__version__, "torch CUDA", torch.version.cuda, "available", torch.cuda.is_available()); print("ORT providers", ort.get_available_providers())'
python -m cross_event_verifier doctor
```

检查 `torch.cuda.is_available()` 为 `True`，ONNX Runtime provider 列表含 `CUDAExecutionProvider`，且 `doctor` 没有阻断生产运行的 issues。当前生产配置默认 `require_cuda=True`，缺少 CUDA 时会 fail closed；CLI 没有已支持的 CPU 生产切换选项。没有可用 GPU/provider 时可以运行 `demo`，但不要把它视为生产视觉链路。若 provider 因找不到 `.so` 而无法加载，先核对 ONNX Runtime 1.27 所需的 CUDA 13.x / cuDNN 9 库；仅当这些库安装在非标准目录时，才把其 `lib` 目录加入 `LD_LIBRARY_PATH`。不要把 CUDA 12 与 CUDA 13、或 cuDNN 8 与 9 的目录混用。ONNX Runtime 的版本对应关系见[官方 CUDA Execution Provider 说明](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)。

`CUDA_HOME` / `CUDA_PATH` 是编译工具链定位变量，不是一般运行时必需项；动态加载器通常通过发行版库路径查找共享库。临时调试时可在启动应用的同一 shell 中设置：

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

只有确认库确实安装于 `/usr/local/cuda/lib64` 后才使用该示例。终端中 `export` 的值只传给该终端启动的进程；桌面图标、systemd 服务有各自的环境配置。项目没有随仓库提供 Linux systemd 服务单元。

### 摄像头接入与权限

Linux V4L2 摄像头通常显示为 `/dev/videoN`。先检查设备节点与支持的格式：

```bash
sudo apt update
sudo apt install -y v4l-utils
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

如果当前用户没有读写权限，可加入发行版的 `video` 组，然后注销并重新登录以刷新组权限：

```bash
sudo usermod -aG video "$USER"
```

不要用 `chmod 666 /dev/video*` 或以 root 启动 GUI 来绕过权限。若有多个摄像头，可优先用 `/dev/v4l/by-id/` 下稳定的设备链接核对设备身份；GUI 当前按数字摄像头索引选择，实际索引应在目标主机验证。现有采集代码先尝试 OpenCV DirectShow，再回退到 OpenCV 默认 backend；DirectShow 是 Windows backend，Linux 上依赖回退路径和 OpenCV/V4L2 build，因此必须在目标 Linux 机器用实际摄像头验证 `doctor` 之外的打开与连续读帧行为。OpenCV、Qt 的系统共享库也需按发行版安装；如 Qt 提示 `xcb` plugin 或共享库缺失，应安装报错所指的发行版包，不要复制 Windows Qt DLL。

### ESP-IDF USB 串口

Windows 设备管理器显示 `COM7` 时可在 Git Bash 使用 `idf.py -p COM7 flash monitor`，并按现场设备更换 `COMx`。Linux 开发板常见串口节点为 `/dev/ttyUSB0` 或 `/dev/ttyACM0`：

```bash
ls /dev/ttyUSB* /dev/ttyACM*
idf.py -p /dev/ttyUSB0 flash monitor
```

若 Linux 普通用户无权限，在 Ubuntu/Debian 可加入 `dialout` 组并重新登录；其他发行版请检查设备节点所属组（常见为 `uucp`）：

```bash
sudo usermod -aG dialout "$USER"
```

不要以 root 运行应用或 `idf.py` 来绕过串口权限。USB-UART 与 USB-JTAG/CDC 接口枚举的设备名不同，以实际设备节点为准。

### BlueZ / Bluetooth LE

Linux 的 Bleak backend 通过 BlueZ 的 D-Bus API 工作；项目当前约束 `bleak>=0.22,<2`，请为该依赖版本安装发行版支持的 BlueZ 与 D-Bus 服务。Ubuntu/Debian 示例：

```bash
sudo apt update
sudo apt install -y bluez dbus rfkill
sudo systemctl enable --now bluetooth.service
systemctl status bluetooth.service --no-pager
rfkill list bluetooth
bluetoothctl show
```

若无线电被软/硬件禁用，先检查设备开关与 `rfkill` 状态；确认适配器后，可用 `bluetoothctl scan on` 扫描并以 `Ctrl+C` 结束。Bleak 使用当前系统的 BlueZ/D-Bus 服务；确保 `bluetooth.service` 正常、桌面用户 D-Bus 会话可用，并按发行版的 polkit/D-Bus 策略授予普通用户设备访问权限。不要为了方便全局放宽 D-Bus 权限，也不需要把用户盲目加入一个假设存在的 `bluetooth` 组。

本项目 BLE extra 为 `bleak>=0.22,<2`。设置扫描所得地址后，从同一终端启动 GUI：

```bash
python -m pip install -e ".[ble]"
export CROSS_EVENT_TRANSPORT="ble-light"
export CROSS_EVENT_BLE_ADDRESS="AA:BB:CC:DD:EE:FF"
python -m cross_event_verifier gui --vision-backend production
```

示例地址必须替换为当前 Linux 主机实际扫描到的设备地址。也可使用 CLI 参数 `--action-transport ble-light --ble-address ...`；服务 UUID 等沿用项目默认值。Bleak Linux 说明见[官方 Linux backend 文档](https://bleak.readthedocs.io/en/latest/backends/linux.html)。

### 数据库（SQLite）

当前 `SqliteStore` 使用 Python 标准库 `sqlite3` 和本地 SQLite 文件；**它不是 PostgreSQL/MySQL 一类客户端/服务器数据库，不需要安装或启动独立数据库 daemon，也没有 `sqlite.service`**。生产 GUI 默认使用仓库相对路径 `data/verifier-production-gpgaitpp-v1.sqlite3`，也可通过 `--database` 指向用户数据目录。SQLite CLI 只用于人工检查，是可选工具：

```bash
sudo apt install -y sqlite3  # 仅需要 sqlite3 命令行工具时安装
mkdir -p "$HOME/.local/share/cross-event-prototype-verifier"
chmod 700 "$HOME/.local/share/cross-event-prototype-verifier"
umask 077
python -m cross_event_verifier gui \
  --database "$HOME/.local/share/cross-event-prototype-verifier/verifier.sqlite3" \
  --vision-backend production
```

数据库父目录必须事先存在且由运行应用的用户可写。文件模式设为 WAL，SQLite 可能在数据库同目录创建 `-wal` 与 `-shm` sidecar；因此目录权限、备份和清理都应覆盖同目录文件。不要把活跃数据库放在不支持 SQLite WAL 锁语义的网络共享盘上。需要检查时，在应用关闭后执行 `sqlite3 "$HOME/.local/share/cross-event-prototype-verifier/verifier.sqlite3" '.tables'`；不要删除数据库来修复模型或身份协议问题。项目当前不提供 PostgreSQL/MySQL backend；接入服务器型数据库需要另行设计和实现，不能只靠配置切换。

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

环境变量在 Git Bash/Linux shell 中写为 export NAME="value"，仅对当前 shell 及其子进程生效。项目没有自动加载 `.env` 文件的依赖或入口。要让 GUI 读取变量，应从设置变量的同一 Git Bash/终端启动；服务或桌面快捷方式需要按各自启动机制单独配置。Windows 的 $env:NAME=value 是 PowerShell 语法，不应粘贴进 Bash。

路径应通过 pathlib 或操作系统 API 组合；避免硬编码分隔符、只在 Windows 成立的大小写假设或 shell 特有展开。提交 shell 脚本使用 LF。跨平台环境差异不能成为修改 Public API、身份阈值、模型契约或 SQLite schema 的理由。

## WSL 状态

本机原 WSL 项目副本 /home/wzige/cross-event-prototype-verifier 已删除。不要将其视作当前工作区，也不要从 Windows 与 WSL 两侧并行修改项目。若未来要重新启用 WSL，需由用户明确提出新的环境迁移任务。

## IDE

Windows IDE 应打开当前 Windows 项目目录，并将解释器指向该目录下的 .venv/Scripts/python.exe。Linux IDE 应打开 Linux 项目目录并选择该目录下的 .venv/bin/python。不要将一侧的虚拟环境配置指向另一侧项目文件系统。
