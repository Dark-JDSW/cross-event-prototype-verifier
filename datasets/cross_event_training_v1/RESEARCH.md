# 数据源核验记录

核验日期：2026-08-26

本记录只引用数据集作者/机构项目页、官方代码仓库和论文页面。它用于解释为什么本构建包选择 MEVID，以及为什么没有把现有本地数据或其他公开数据研究 伪装成目标域生产训练集。

## 1. 主来源：MEVID

MEVID 官方仓库报告 158 个身份、598 套服装、8,092 条轨迹，平均轨迹约 590 帧，覆盖 33 个摄像机视角。其训练部分为 6 个日期/9 天、104 个身份，测试部分为 3 个日期/5 天、54 个身份；官方仓库还提供 RGB bounding-box 包、身份/服装/摄像机标注和原始视频 URL 清单。

- 官方代码与数据说明：[Kitware/MEVID](https://github.com/Kitware/MEVID)
- 官方论文：[MEVID: Multi-view Extended Videos with Identities for Video Person Re-Identification](https://openaccess.thecvf.com/content/WACV2023/papers/Davila_MEVID_Multi-View_Extended_Videos_With_Identities_for_Video_Person_Re-Identification_WACV2023_paper.pdf)
- 官方数据发布消息：[MEVID release](https://groups.google.com/g/meva-data-users/c/WMT57QwlpKc)
- MEVA 数据注册与许可证入口：[AWS Open Data Registry](https://github.com/awslabs/open-data-registry/blob/main/datasets/mevadata.yaml)

MEVID 的官方五列轨迹索引只提供图像名区间、身份、服装和摄像机 ID；它没有自动满足本项目所需的原始视频 SHA-256、准确源视频帧区间、walking 人工确认、姿态条件和 RTMPose 输出。因此 `source_metadata.jsonl` 把这些字段设为强制输入，生成器缺任何一个就拒绝产出事件清单。

## 2. 辅助来源：FVG-B

FVG-B 官方页面报告 226 个身份、2,856 个视频、15 FPS、1,920×1,080、3 个前向角度和 3 个采集 session；其中 12 个身份在 2017 与 2018 重复采集。它适合测试正面/斜向、慢快走、携物和换衣，但绝大多数身份没有足够跨日期独立 session，不能满足本项目 P1 的每身份 4 session 要求。

- 官方数据页：[Front-View Gait Blurred Database](https://cvlab.cse.msu.edu/frontal-view-gaitfvg-database.html)
- 官方技术报告：[FVG-B Report](https://cvlab.cse.msu.edu/pdfs/FVG_B_Report.pdf)
- 原始研究论文：[Gait Recognition via Disentangled Representation Learning](https://cvlab.cse.msu.edu/pdfs/Zhang_Tran_Liu_Liu_arxiv2019.pdf)

## 3. 排除来源：DeepChange

DeepChange 具有真实长时跨度和换衣变化，官方页面报告 17 个摄像头、1,121 个身份、178,407 个 bounding boxes 和 12 个月采集跨度；但其官方条款限制为非商业研究、禁止复制或再分发，并要求取得访问协议。因此它不能直接作为当前项目生产训练资产，只能在单独获批的研究环境中作为外部评测参考。

- 官方项目页：[PengBoXiangShang/deepchange](https://github.com/PengBoXiangShang/deepchange)
- 官方论文：[DeepChange](https://openaccess.thecvf.com/content/ICCV2023/papers/Xu_DeepChange_A_Long-Term_Person_Re-Identification_Benchmark_with_Clothes_Change_ICCV_2023_paper.pdf)

## 4. 本地数据为何不纳入

项目现有 `train/multi_action_dataset_v2.zip` 的本地 README 和质量报告已经明确：它主要是 Kinect skeleton proxy，没有原始 RGB、原始 RTMPose 输出、衣着标签和可靠日期。`val/` 中四个视频只有同一人、两套衣服和有限视角，不能形成 60/80 身份的开放集协议。为了保留数据血缘，本构建包不复制、不改名、不修改这些文件。

## 5. 决策

MEVID 是最适合作为本项目“外部研究源 + 目标域处理模板”的候选，但只有经过授权、原始视频映射、walking 复核和项目姿态链路重算后，才可晋级。公开数据研究 的 Rank-1 或已有 gait/skeleton 文件不能替代本项目要求的真实 RGB → detector → tracker → RTMPose → GPGait++ 链路，也不能用于正式 FPIR/FNIR 校准。

