"""Self-contained MMAction2 inference config for the released HMDB51 PoseC3D model."""

default_scope = "mmaction"

model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="ResNet3dSlowOnly",
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(3, 4, 6),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
    ),
    cls_head=dict(
        type="I3DHead",
        in_channels=512,
        num_classes=51,
        spatial_type="avg",
        dropout_ratio=0.5,
        average_clips="prob",
    ),
    train_cfg=None,
    test_cfg=None,
)

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# The live gate uses one deterministic clip to keep per-track latency bounded.
# The official model-zoo score uses ten test clips and must not be compared to
# this runtime protocol without a separate evaluation.
test_pipeline = [
    dict(
        type="UniformSampleFrames",
        clip_len=48,
        num_clips=1,
        test_mode=True,
        seed=255,
    ),
    dict(type="PoseDecode"),
    dict(type="PoseCompact", hw_ratio=1.0, allow_imgpad=True),
    dict(type="Resize", scale=(-1, 56)),
    dict(type="CenterCrop", crop_size=56),
    dict(
        type="GeneratePoseTarget",
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=False,
        left_kp=left_kp,
        right_kp=right_kp,
    ),
    dict(type="FormatShape", input_format="NCTHW_Heatmap"),
    dict(type="PackActionInputs"),
]
