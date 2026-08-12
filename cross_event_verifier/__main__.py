"""``python -m cross_event_verifier`` 的模块入口。

命令行实现位于参与者 C 的 ``cli`` 模块。本文件只保留一个很小的转发入口，
使包遵循 Python 的标准模块执行约定，而不重复参数解析逻辑。
"""

from .participant_c.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
