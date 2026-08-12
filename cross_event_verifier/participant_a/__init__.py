"""参与者 A：身份表示和证据策略。

本包负责与模型无关的身份决策部分，包含外观/ReID 和步态编码器、校准分数
融合、开放集策略、候选人记忆以及 :class:`CrossEventVerifier` 深模块。它接
收其他包提供的 ``TrackQuality`` 和 ``FeatureBundle``，不会打开摄像头或导入
GUI 代码。
"""
