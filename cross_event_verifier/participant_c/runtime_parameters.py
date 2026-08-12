"""桌面管线中经过校验、在帧边界生效的运行时调参。

GUI 提交一组映射并接收一个不可变快照。验证、
跨字段不变量以及验证器、自动化、校准器和
可选生产视觉适配器的更新都隐藏在本模块的小
接口之后。媒体工作线程在帧与帧之间调用它，因此每一帧始终看到
完整的旧参数集或新参数集。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Literal, Mapping

from .automation import AutomationPolicy
from ..participant_a.calibration import ScoreCalibrator

if TYPE_CHECKING:
    from .automation import AutomaticVerificationController
    from ..participant_a.engine import CrossEventVerifier
    from ..participant_b.vision import VisionAdapter


RuntimeScalar = int | float
ParameterKind = Literal["int", "float"]


@dataclass(frozen=True)
class RuntimeParameterSpec:
    """描述一个可调标量及其面向 GUI 的校验契约。

    将标签、范围、默认值和说明与键放在一起，可以避免 Tk 页面形成另一套
    细微不同的参数模式。``coerce`` 有意保持严格，防止无效表单提交部分进入
    验证器或视觉适配器。
    """

    key: str
    section: str
    label: str
    kind: ParameterKind
    minimum: RuntimeScalar
    maximum: RuntimeScalar
    default: RuntimeScalar
    description: str

    def coerce(self, raw_value: object) -> RuntimeScalar:
        """将表单文本值转换为声明类型并强制执行范围。"""
        if isinstance(raw_value, bool):
            raise ValueError(f"{self.label} 必须是数值")
        try:
            numeric = float(str(raw_value).strip())
        except (TypeError, ValueError) as error:
            raise ValueError(f"{self.label} 必须是数值") from error
        if not math.isfinite(numeric):
            raise ValueError(f"{self.label} 必须是有限数值")
        if self.kind == "int":
            if not numeric.is_integer():
                raise ValueError(f"{self.label} 必须是整数")
            value: RuntimeScalar = int(numeric)
        else:
            value = float(numeric)
        if value < self.minimum or value > self.maximum:
            raise ValueError(
                f"{self.label} 必须在 {self.minimum:g}～{self.maximum:g} 之间"
            )
        return value

    def format(self, value: RuntimeScalar) -> str:
        """返回适合 Tk 输入框显示的紧凑值。"""
        return str(int(value)) if self.kind == "int" else f"{float(value):g}"


def _float_spec(
    key: str,
    section: str,
    label: str,
    minimum: float,
    maximum: float,
    default: float,
    description: str,
) -> RuntimeParameterSpec:
    """构造带有共享元数据的浮点参数说明。"""
    return RuntimeParameterSpec(
        key,
        section,
        label,
        "float",
        minimum,
        maximum,
        default,
        description,
    )


def _int_spec(
    key: str,
    section: str,
    label: str,
    minimum: int,
    maximum: int,
    default: int,
    description: str,
) -> RuntimeParameterSpec:
    """构造带有共享元数据的整数参数说明。"""
    return RuntimeParameterSpec(
        key,
        section,
        label,
        "int",
        minimum,
        maximum,
        default,
        description,
    )


RUNTIME_PARAMETER_SPECS: tuple[RuntimeParameterSpec, ...] = (
    _float_spec(
        "verifier.gait_novelty_threshold",
        "开放集与步态锚点",
        "新人物步态上限",
        0.0,
        0.89,
        0.35,
        "高质量步态低于此概率时明确拒绝全部旧 ID；调高更易建新号。",
    ),
    _float_spec(
        "verifier.strong_gait_probability",
        "开放集与步态锚点",
        "强步态确认概率",
        0.01,
        1.0,
        0.90,
        "达到此概率才可把 Track 正式确认为旧 ID。",
    ),
    _float_spec(
        "verifier.strong_gait_margin",
        "开放集与步态锚点",
        "强步态 Top2 间隔",
        0.0,
        1.0,
        0.08,
        "最佳与次佳步态概率的最低差值。",
    ),
    _float_spec(
        "verifier.defer_threshold",
        "开放集与步态锚点",
        "疑似匹配阈值",
        0.0,
        0.99,
        0.62,
        "非步态锚定策略下进入等待区的融合概率。",
    ),
    _float_spec(
        "verifier.accept_threshold",
        "开放集与步态锚点",
        "融合接受阈值",
        0.01,
        1.0,
        0.82,
        "非步态锚定策略和全局指派使用的接受概率。",
    ),
    _float_spec(
        "verifier.margin_threshold",
        "开放集与步态锚点",
        "融合 Top2 间隔",
        0.0,
        1.0,
        0.08,
        "融合最佳项与次佳项的最低差值。",
    ),
    _float_spec(
        "verifier.strong_gait_quality",
        "质量门控",
        "强步态质量",
        0.0,
        1.0,
        0.70,
        "步态用于确认或新建身份前的质量下限。",
    ),
    _float_spec(
        "verifier.detection_confidence_floor",
        "质量门控",
        "检测质量下限",
        0.0,
        1.0,
        0.35,
        "低于此置信度的检测不进入可靠证据。",
    ),
    _int_spec(
        "verifier.minimum_frames",
        "质量门控",
        "最少轨迹帧",
        1,
        300,
        8,
        "验证器计算成熟质量所需的最少帧数。",
    ),
    _float_spec(
        "verifier.minimum_gait_cycles",
        "质量门控",
        "最少步态周期",
        0.0,
        5.0,
        1.0,
        "强步态证据要求的最少行走周期。",
    ),
    _float_spec(
        "verifier.minimum_matching_quality",
        "质量门控",
        "最低匹配质量",
        0.0,
        1.0,
        0.38,
        "外观和步态质量都低于此值时继续等待。",
    ),
    _float_spec(
        "verifier.maximum_write_occlusion",
        "质量门控",
        "最大写入遮挡",
        0.0,
        1.0,
        0.40,
        "遮挡超过此比例时禁止写入正式图库。",
    ),
    _int_spec(
        "automation.minimum_track_frames",
        "自动注册稳定性",
        "注册预热帧数",
        1,
        300,
        16,
        "自动采集新人物步态前需持续跟踪的帧数。",
    ),
    _int_spec(
        "automation.minimum_stable_gait_samples",
        "自动注册稳定性",
        "稳定步态样本数",
        2,
        64,
        8,
        "连续通过一致性门后才允许建号的嵌入数量。",
    ),
    _int_spec(
        "automation.gait_sample_window",
        "自动注册稳定性",
        "步态样本窗口",
        2,
        128,
        16,
        "保留用于稳定性统计的最近步态样本数量。",
    ),
    _float_spec(
        "automation.minimum_sample_similarity",
        "自动注册稳定性",
        "单样本一致性",
        -1.0,
        1.0,
        0.86,
        "新步态样本与当前中心低于此相似度时重新采集。",
    ),
    _float_spec(
        "automation.minimum_gait_stability",
        "自动注册稳定性",
        "聚合步态稳定度",
        -1.0,
        1.0,
        0.94,
        "整个采集窗口与中心的最低平均相似度。",
    ),
    _float_spec(
        "verifier.appearance_floor",
        "融合与外观响应",
        "外观相似度底线",
        -1.0,
        1.0,
        0.45,
        "低于底线的外观不贡献融合可靠性。",
    ),
    _float_spec(
        "verifier.gait_floor",
        "融合与外观响应",
        "步态相似度底线",
        -1.0,
        1.0,
        0.58,
        "低于底线的步态不贡献融合可靠性。",
    ),
    _float_spec(
        "verifier.conflict_probability",
        "融合与外观响应",
        "分支冲突概率",
        0.0,
        1.0,
        0.72,
        "外观与步态分别强指向不同 ID 时的冲突门。",
    ),
    _float_spec(
        "verifier.maximum_gait_weight",
        "融合与外观响应",
        "融合最大步态权重",
        0.0,
        1.0,
        0.35,
        "双分支同时存在时步态在融合分数中的上限。",
    ),
    _float_spec(
        "verifier.strong_appearance_probability",
        "融合与外观响应",
        "强外观响应概率",
        0.0,
        1.0,
        0.90,
        "已有步态令牌时，外观响应匹配目标的概率下限。",
    ),
    _float_spec(
        "verifier.strong_appearance_quality",
        "融合与外观响应",
        "强外观响应质量",
        0.0,
        1.0,
        0.70,
        "外观响应被吸收前的图像质量下限。",
    ),
    _float_spec(
        "calibration.appearance_scale",
        "分支概率校准",
        "外观校准斜率",
        0.1,
        30.0,
        8.0,
        "外观余弦相似度映射到概率时的陡峭程度。",
    ),
    _float_spec(
        "calibration.appearance_midpoint",
        "分支概率校准",
        "外观校准中点",
        -1.0,
        1.0,
        0.56,
        "映射为 0.5 概率的外观余弦相似度。",
    ),
    _float_spec(
        "calibration.gait_scale",
        "分支概率校准",
        "步态校准斜率",
        0.1,
        30.0,
        8.0,
        "步态余弦相似度映射到概率时的陡峭程度。",
    ),
    _float_spec(
        "calibration.gait_midpoint",
        "分支概率校准",
        "步态校准中点",
        -1.0,
        1.0,
        0.68,
        "映射为 0.5 概率的步态余弦相似度。",
    ),
    _float_spec(
        "vision.detector_confidence",
        "生产视觉前端",
        "YOLO 跟踪置信度",
        0.0,
        1.0,
        0.25,
        "送入 ByteTrack 的 YOLO 最低检测置信度。",
    ),
    _float_spec(
        "vision.output_confidence",
        "生产视觉前端",
        "人物输出置信度",
        0.0,
        1.0,
        0.45,
        "低于此值的已跟踪人物不输出到验证器。",
    ),
    _float_spec(
        "vision.detector_iou",
        "生产视觉前端",
        "YOLO NMS IoU",
        0.0,
        1.0,
        0.50,
        "YOLO 非极大值抑制使用的 IoU 阈值。",
    ),
    _float_spec(
        "vision.keypoint_confidence",
        "生产视觉前端",
        "关键点置信度",
        0.0,
        1.0,
        0.45,
        "RTMPose 关键点进入步态序列的置信度下限。",
    ),
    _int_spec(
        "vision.minimum_pose_frames",
        "生产视觉前端",
        "有效姿态帧数",
        8,
        60,
        25,
        "GaitGraph2 开始输出步态前要求的有效姿态帧数。",
    ),
    _int_spec(
        "vision.appearance_stride",
        "生产视觉前端",
        "外观提取间隔",
        1,
        30,
        3,
        "每隔多少 Track 帧刷新一次 OSNet 外观嵌入。",
    ),
    _float_spec(
        "vision.low_light_threshold",
        "生产视觉前端",
        "低照度阈值",
        0.0,
        255.0,
        100.0,
        "画面平均灰度低于此值时启用 CLAHE。",
    ),
)

_SPECS_BY_KEY = {item.key: item for item in RUNTIME_PARAMETER_SPECS}


@dataclass(frozen=True)
class RuntimeParameterState:
    """当前后端可见参数在某一时刻的不可变视图。

    ``revision`` 让 GUI 能报告当前生效的事务；``available_keys`` 则区分生产专用
    控件与有意暴露较少参数的 demo 后端。
    """

    revision: int
    values: Mapping[str, RuntimeScalar]
    available_keys: tuple[str, ...]


class RuntimeParameterController:
    """原子校验并应用管线中所有可热更新参数。

    控制器是表单提交与可变运行时状态之间的唯一接口。它先构造候选验证器、
    策略和校准值，再在帧边界提交。校验失败时，之前的快照保持不变。
    """

    def __init__(
        self,
        verifier: CrossEventVerifier,
        automation: AutomaticVerificationController,
        vision: VisionAdapter,
    ) -> None:
        """将控制器绑定到实时验证器、自动化和视觉接口。"""
        self.verifier = verifier
        self.automation = automation
        self.vision = vision
        self._revision = 0

    @property
    def specs(self) -> tuple[RuntimeParameterSpec, ...]:
        """返回 GUI 使用的完整有序参数模式。"""
        return RUNTIME_PARAMETER_SPECS

    @staticmethod
    def defaults() -> dict[str, RuntimeScalar]:
        """返回代码默认值，不修改任何实时组件。"""
        return {item.key: item.default for item in RUNTIME_PARAMETER_SPECS}

    def _vision_values(self) -> dict[str, RuntimeScalar]:
        """读取所选视觉后端公开的可选热更新控件。"""
        reader = getattr(self.vision, "runtime_parameters", None)
        if not callable(reader):
            return {}
        values = reader()
        return {
            f"vision.{name}": value
            for name, value in dict(values).items()
            if f"vision.{name}" in _SPECS_BY_KEY
        }

    def state(self) -> RuntimeParameterState:
        """从所有参与组件组装一致的运行时快照。"""
        config = self.verifier.config
        policy = self.automation.policy
        values: dict[str, RuntimeScalar] = {}
        for spec in RUNTIME_PARAMETER_SPECS:
            prefix, field_name = spec.key.split(".", 1)
            if prefix == "verifier":
                values[spec.key] = getattr(config, field_name)
            elif prefix == "automation":
                values[spec.key] = getattr(policy, field_name)
        values.update(
            {
                "calibration.appearance_scale": self.verifier.appearance_calibrator.scale,
                "calibration.appearance_midpoint": self.verifier.appearance_calibrator.midpoint,
                "calibration.gait_scale": self.verifier.gait_calibrator.scale,
                "calibration.gait_midpoint": self.verifier.gait_calibrator.midpoint,
            }
        )
        values.update(self._vision_values())
        available = tuple(item.key for item in RUNTIME_PARAMETER_SPECS if item.key in values)
        return RuntimeParameterState(self._revision, dict(values), available)

    def apply(self, updates: Mapping[str, object]) -> RuntimeParameterState:
        """在提交任何部分之前校验完整的变更集。

        校准对象和数据类替换对象会在赋值前构造。这样跨字段检查会在实时验证器
        修改前失败；视觉更新只有在其适配器接受候选值后才会委托执行。
        """

        before = self.state()
        unknown = sorted(set(updates) - set(_SPECS_BY_KEY))
        if unknown:
            raise ValueError(f"未知运行时参数：{', '.join(unknown)}")
        unavailable = sorted(set(updates) - set(before.available_keys))
        if unavailable:
            raise ValueError(f"当前视觉后端不支持：{', '.join(unavailable)}")

        parsed: dict[str, RuntimeScalar] = {}
        for key, raw_value in updates.items():
            value = _SPECS_BY_KEY[key].coerce(raw_value)
            if value != before.values[key]:
                parsed[key] = value
        if not parsed:
            return before

        verifier_changes = {
            key.split(".", 1)[1]: value
            for key, value in parsed.items()
            if key.startswith("verifier.")
        }
        automation_changes = {
            key.split(".", 1)[1]: value
            for key, value in parsed.items()
            if key.startswith("automation.")
        }
        vision_changes = {
            key.split(".", 1)[1]: value
            for key, value in parsed.items()
            if key.startswith("vision.")
        }
        next_revision = self._revision + 1
        candidate_config = replace(
            self.verifier.config,
            **verifier_changes,
            threshold_version=f"runtime-v{next_revision}",
        )
        candidate_policy = replace(self.automation.policy, **automation_changes)

        appearance_values = {
            "scale": parsed.get(
                "calibration.appearance_scale",
                self.verifier.appearance_calibrator.scale,
            ),
            "midpoint": parsed.get(
                "calibration.appearance_midpoint",
                self.verifier.appearance_calibrator.midpoint,
            ),
        }
        gait_values = {
            "scale": parsed.get(
                "calibration.gait_scale",
                self.verifier.gait_calibrator.scale,
            ),
            "midpoint": parsed.get(
                "calibration.gait_midpoint",
                self.verifier.gait_calibrator.midpoint,
            ),
        }
        candidate_appearance = ScoreCalibrator(
            scale=float(appearance_values["scale"]),
            midpoint=float(appearance_values["midpoint"]),
            name=self.verifier.appearance_calibrator.name,
        )
        candidate_gait = ScoreCalibrator(
            scale=float(gait_values["scale"]),
            midpoint=float(gait_values["midpoint"]),
            name=self.verifier.gait_calibrator.name,
        )

        # 可选视觉适配器先校验并提交。从此处开始的操作都是不会失败的进程内赋值。
        # 调用发生在媒体线程的两次 process_frame 之间。
        if vision_changes:
            updater = getattr(self.vision, "update_runtime_parameters", None)
            if not callable(updater):
                raise ValueError("当前视觉后端不支持动态视觉参数")
            updater(vision_changes)
        self.verifier.config = candidate_config
        self.verifier.appearance_calibrator = candidate_appearance
        self.verifier.gait_calibrator = candidate_gait
        if candidate_policy != self.automation.policy:
            self.automation.update_policy(candidate_policy)
        else:
            # 旧阈值/校准下收集的证据不能与新事务下的证据混合。有效的外观令牌
            # 和已经确认的轨迹身份保持不变。
            self.automation.reset_gait_samples()
        self._revision = next_revision
        return self.state()


__all__ = [
    "RUNTIME_PARAMETER_SPECS",
    "RuntimeParameterController",
    "RuntimeParameterSpec",
    "RuntimeParameterState",
]
