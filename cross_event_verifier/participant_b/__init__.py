"""参与者 B：视觉感知、跟踪和姿态质量。

本包把视频帧转换为稳定的 ``VisionTrack`` 值。生产适配器负责
YOLO/ByteTrack/RTMPose 的编排，并把身份嵌入交给参与者 A 的编码器；当生产
权重不可用时，轻量 OpenCV 适配器可供 GUI 诊断使用。
"""
