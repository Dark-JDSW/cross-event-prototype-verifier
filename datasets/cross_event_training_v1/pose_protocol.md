# RTMPose / GPGait++ 姿态协议

每个事件下按 clean sub-tracklet 输出一个 `.npz`，路径形如：

```text
poses/<event_id>/<event_id>_<track_id>_<subtracklet_id>.npz
```

必须保留以下数组：

| 数组 | 形状 | 语义 |
|---|---|---|
| `keypoints_xy` | `[T,17,2]` | RTMPose 结果映射回原视频的像素坐标，COCO17 顺序 |
| `keypoint_scores` | `[T,17]` | RTMPose 原始置信度 |
| `valid_mask` | `[T,17]` | 关键点是否可用；缺失不能用插值伪装 |
| `boxes_xyxy` | `[T,4]` | YOLO/项目固定 detector 的真实框 |
| `frame_indices` | `[T]` | 原视频帧号，不能重编号成连续“有效帧” |
| `timestamp` | `[T]` | 原视频时间戳，优先来自视频/容器时间基 |
| `track_id` | `[T]` | ByteTrack 源轨迹标识 |
| `missing_mask` | `[T]` | 该帧是否因丢帧/检测缺失而不可用 |

建议附加字段：`posture_condition`、`posture_confidence`、`posture_window_purity`、`transition_flag`、`track_id_switch`、`track_gap`、`detector_gap`、`box_truncated`、`feet_visible`、`leg_visibility`、`occlusion_level`、`valid_pose_frame_count`、`valid_leg_frame_count`、`gait_cycle_count`、`pose_jitter_level`、`mixed_identity_flag`。

## 硬性处理规则

1. 先固定 detector 和 tracker，再运行 RTMPose；不要用 MEVID 已有 cropped JPEG 的框和分数冒充项目 detector/pose 输出。
2. 关键点保持原图坐标。进入项目 GPGait++ 时，由编码器执行官方 Affine → RescaleCenter → HOD_MultiInput → SkeletonInput 预处理，不在数据集阶段偷偷改成另一套坐标协议。
3. 长 gap、ID switch、混入他人或框截断必须形成 clean sub-tracklet 边界；禁止跨污染区间拼接一个“更长”的样本。
4. 缺帧保留 `frame_indices`、`timestamp` 和 `missing_mask`。重采样到 60 帧不增加真实有效帧数。
5. `front`/`back` 需要人工或可靠方向模型确认。不能仅依据肩宽/躯干比例把正面和背面硬标成不同类别。
6. 训练、calibration 和 locked-test 的身份清单在构建报告写入后冻结；任何修改都要生成新版本和新 manifest hash。

