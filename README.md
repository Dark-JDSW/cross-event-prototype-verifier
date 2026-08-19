# cross-event-prototype-verifier

一个独立运行的跨事件人物验证系统：先用 OSNet 外观特征确认视觉身份并绑定 Track，再用 RTMPose 与 GaitGraph2 学习该身份的步态原型，最终以步态作为跨事件身份检索依据，并提供摄像头/视频 GUI、开放集拒识、隔离图库和可回滚持久化。

```text
摄像头 / 视频
      ↓
低照度检测与 CLAHE（按需）
      ↓
YOLO11L 人物检测 → ByteTrack 多目标轨迹
      ├──────────────→ OSNet-AIN 512 维外观特征 → 视觉身份 P1/P2 绑定
      └→ RTMPose 17 点姿态 → GaitGraph2 384 维时序步态特征 → 质量加权步态样本
                                  ↓                         ↓
                    视觉身份全局一对一指派       独立步态事件/事件修订历史
                                  ↓                         ↓
                 多证据开放集判决与步态就绪
                                  ↓
                 步态主检索 + OSNet 候选关联 / formal 多原型 SQLite 图库
```

## 技术栈

| 层次 | 技术 | 在项目中的职责 |
|---|---|---|
| 语言与数值计算 | Python 3.10+、NumPy | 业务编排、值对象、质量计算和向量处理 |
| 视频输入与预处理 | OpenCV、CLAHE | 摄像头/视频读取、图像转换、低照度增强和画面绘制 |
| 人物检测 | Ultralytics YOLO11L（`yolo11l.pt`） | 只检测和保留 person 类目标 |
| 多目标跟踪 | Ultralytics ByteTrack（`bytetrack-cross-event.yaml`） | 维护连续 Track；ByteTrack 本身不承担外观 ReID |
| 视觉身份 | 项目内 PyTorch OSNet-AIN，512 维 L2 归一化特征 | 通过外观匹配、人工确认或外部任务绑定 `P1`、`P2` 等视觉身份 |
| 姿态估计 | MMPose RTMPose-S 导出的 ONNX 模型（17 点 COCO 关键点） | 为已绑定视觉身份采集逐帧姿态序列 |
| 步态编码 | 项目内 PyTorch GaitGraph2 GREW 兼容编码器（60 帧、384 维 L2 向量） | 将姿态序列编码为步态样本，并形成步态事件与步态原型 |
| 推理运行时 | PyTorch CUDA、ONNX Runtime GPU（CUDA/CPU Execution Provider） | YOLO、OSNet 和 GaitGraph2 使用 PyTorch；RTMPose 使用 ONNX Runtime；允许少量 CPU 辅助节点 |
| 身份决策 | Python 领域模块、质量门、分支校准、全局一对一指派 | 管理视觉身份绑定、步态事件审核、开放集和步态就绪状态 |
| 持久化与检索 | SQLite、NumPy 向量索引；可选 FAISS | 保存 formal/quarantine 图库、模型协议、审核记录和多原型检索数据 |
| 交互与命令行 | Tkinter、`python -m cross_event_verifier` | GUI、`doctor`、模型下载、演示和生产管线入口 |

生产链路中，OSNet 的 512 维外观向量不会直接输入 GaitGraph2。OSNet 负责确认视觉身份和关联 Track；RTMPose 的时序姿态才是 GaitGraph2 的输入。GaitGraph2 权重只读加载，系统学习的是写入 SQLite 的步态样本、步态事件和由事件派生的身份模型，而不是在线修改模型参数。

## 更新日志
- V1.00
  - 基础框架，效果差，好在已跑通技术栈

- V1.05
  - 修复了GUI的小BUG
  - 取消项目分工设计
  - 优化视频流逻辑，流畅度大幅提升！
  - 重构算法，一定程度解决了强步态Pg得分过低，导致自动注册功能失效的问题(但还是不好用！)
  - 试着用轮廓识别方案替换GaitGraphic2，但是效果很差，又换回来了（

- V1.10
  - 在保留当前技术栈的前提下，完全重写了算法，效果拔群
  - 给GUI加了个一键删库按钮，方便删库跑路
  - 可能(?)小幅优化了性能开销

- V1.15
  - 针对换衣问题增加步态回连机制与身份创建等待窗口，现在它真能用了
  - 加入了重复学习功能，不用反复操作那个死难看的GUI了
  - 增加动作路由和隔离机制
  - 增加外观多原型学习机制
  - 预留了学习更多姿态的入口，但缺少能用的数据集
  - 优化性能，现在真不卡了
## 先运行起来

```powershell
.\.venv\Scripts\Activate.ps1
python -m cross_event_verifier doctor
python -m cross_event_verifier gui
```

GUI 选择视频文件时，“视频重复学习”默认为 1；设置为大于 1 的正整数后，系统会自动
重复播放同一视频。重复播放共用同一个采集会话键，因此会累计该事件的步态样本，但不会
伪造多个独立事件；摄像头输入不支持重复播放。

如果模型文件不在本机，先看“模型资产与云端迁移”一节，从私有云端同步完成后再运行 `doctor`。

## 证据规则

- 视觉身份由 OSNet 外观匹配、人工确认或外部采集任务确定并获得 `P1`、`P2` 标签；视觉身份已确认不等于步态已经就绪。
- 自动生成的 `P` 是系统内部视觉身份标签，不是 OSNet 从图像中推断出的真实姓名；需要语义姓名时仍应由人工或外部采集任务提供映射。
- 已确认视觉身份的 Track 才能贡献步态样本。RTMPose 姿态序列经 GaitGraph2 编码后，按质量和稳定性审核为步态样本。
- 同一次采集会话/挑战中的多个窗口只计为一个步态事件；通过独立性、稳定性和模型协议检查后，事件事实会追加到 SQLite 修订历史。同一会话后续窗口可以生成新的事件修订，但不会增加独立事件数；该视觉身份的步态模型由当前事件集合重新派生。
- 步态原型不会在线修改 GaitGraph2 权重；系统学习的是该视觉身份的事件事实和原型图库，后续可用 GaitGraph2 向量检索该身份。
- 步态就绪需要足够的独立事件、条件覆盖、内部一致性、留出验证和开放集检查；在此之前状态分别为“步态学习中”或“步态暂可用”。
- 步态冲突会暂停写入并清空当前窗口，不会把冲突事件强行并入视觉身份。
- OSNet 可以负责 Track 的初始身份绑定和候选续接，但不能把任意外观向量直接输入 GaitGraph2；两个模型仍使用各自的输入契约。
- 旧版强步态建号和一次性外观请求 API 仍保留用于兼容与人工复核，生产 GUI 默认使用上述视觉优先流程。
- 旧版步态优先 API 中，高质量步态若明确低于“新人物步态上限”，会拒绝旧 ID 并进入稳定步态采集；生产 GUI 不用步态来创建视觉身份。
- `DEFERRED` 只表示“尚未判定”，不会携带或显示正式身份；OSNet 未达到视觉身份门时，系统继续采集外观样本。
- 视觉身份已确认后，GaitGraph2 的稳定输出才会进入该身份的步态事件审核；开放集和留出验证用于判定“步态就绪”，不是用来生成视觉身份编号。
- 高质量外观明确反对当前视觉身份时，GaitGraph2 不得覆盖该反证；未绑定 Track 也不能凭一个步态窗口直接重新绑定正式身份。
- 25 个真实有效姿态帧只允许进入步态事件学习候选；只有达到更长真实窗口、运动周期和姿态覆盖门后，步态才可承担正式身份检索。
- 只有视觉绑定或步态事件两侧都达到各自质量门且无法解释时才算“证据冲突”。单帧异常会清空当前窗口并重采集，独立事件冲突则暂停该身份的步态写入。
- 步态质量 `Q_gait` 与身份相似度 `S_gait` 分开处理：质量分为 `INVALID`、`PARTIAL`、`STRONG`。默认允许 `PARTIAL` 以较低权重参与窗口聚合，但不能作为“不是该身份”的负证据；严重 ID switch、腿部完全不可见、框截断或序列不足仍会硬拒绝并清空窗口。
- 正式验证会综合 Top-2 margin、类内步态离散度、独立事件支持数、质量、视角和分支冲突；结果可通过 `Decision.open_set_label` 映射为 `KNOWN`、`AMBIGUOUS` 或 `UNKNOWN`。冲突向量不能污染该视觉身份。
- 同一视觉身份的步态原型写入受事件键、模型协议、稳定性和近重复门控保护；同会话重复窗口只更新当前采样窗口，不重复增加事件。
- 触碰画面边界、人体框异常、下肢关键点不足、遮挡严重或轨迹跳变时，不允许产生强步态/强外观证据。
- 多人同框使用全局一对一身份指派，一个正式身份不会在同一帧分配给多个 Track。

默认自动流程：

```text
T 临时 Track
  → ByteTrack 稳定跟踪
  → 连续稳定 OSNet 外观样本
  → 确认/创建视觉身份 P（仅 appearance）
  → 该 Track 的 RTMPose → GaitGraph2 步态样本
  → 质量加权窗口聚合与独立事件审核
  → 事件修订历史 → 该视觉身份的 gait model
  → 多个独立事件、覆盖度、留出和开放集检查
  → 步态暂可用 / 步态就绪，继续以 GaitGraph2 主检索
```

## Windows下环境安装
#这里可以根据自身GPU型号适当改变CUDA版本喵

建议使用 Python 3.12：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m cross_event_verifier download-models
```

安装依赖并下载模型后，再执行环境与模型自检：

```powershell
python -m cross_event_verifier doctor
```

成功时输出中应包含：

```json
{
  "cuda_available": true,
  "cuda_device": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "production_ready": true,
  "issues": []
}
```

`doctor` 会检查运行依赖、CUDA、ONNX Runtime provider、模型文件大小及 SHA-256。返回码非零表示生产 GUI 不应启动。

## 启动 GUI

```powershell
python -m cross_event_verifier gui
```

也可使用安装后的命令：

```powershell
cross-event-verifier gui
```

首次打开摄像头/视频时需要把四个模型加载到内存，RTX 5060 实测约需 8～15 秒；这段时间窗口仍会显示加载状态。完成预热后，单帧耗时取决于分辨率和同框人数。

GUI 操作要点：

1. 选择摄像头设备号（通常为 `0`），或选择 `mp4`、`avi`、`mov`、`mkv` 视频。
2. 点击开始后，让人物从头到脚完整出现在画面中；脚踝被裁掉时系统会继续跟踪，但不会把代理信息误当成强步态。
3. 人物需要实际行走，而不是只站立。界面会依次显示视觉身份确认、步态学习、步态暂可用和步态就绪等阶段。
4. “自动注册新人物”默认开启。关闭后只禁止创建新的视觉身份 `P`，已有身份识别和已绑定 Track 的步态学习仍继续。
5. 人工登记用于明确建立视觉身份标签，优先使用当前 Track 最近窗口内质量最高的 OSNet 外观快照；登记后仍需让目标连续行走，系统才会收集步态样本。
6. 已有视觉身份会由 OSNet 跨源重新绑定；如果视觉身份尚未完成、需要让同一个临时 Track 跨视频/摄像头继续收集外观样本，才在“人物注册”面板填写稳定的“跨事件候选键”，并在换源后保持不变。
7. 切换到“实时参数”页，可以在不中断视频、不重载模型的情况下调整开放集、质量、自动注册、校准和生产视觉参数。
8. 需要重新建立身份图库时，点击“当前正式身份”区域的“清除现有 ID（重新建库）”；确认后系统会先备份 SQLite，再清除视觉身份、步态原型、事件和审计记录。

### 实时参数页

参数页会显示当前实际生效值、允许范围和用途。点击“应用到运行中”后，整组参数先完成类型及跨字段校验，再通过采集线程在下一帧边界一次性生效；因此不会出现一帧读取到半组新值的情况。无效组合会整组拒绝，原参数保持不变。参数真正改变时会清空尚未完成的步态采样窗口，避免把调参前后的证据混在一次自动建号中；已确认身份和待响应外观令牌不受影响。

视觉优先流程中，最直接影响“视觉身份绑定”和“步态就绪”的参数是（旧版步态优先参数仍保留兼容）：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| 新人物步态上限 | `0.35` | 高质量步态对图库最佳 ID 的校准概率低于此值时，明确视为新人物候选；调高会更容易建新号，也会增加重复建号风险 |
| 强步态确认概率 | `0.90` | 步态达到此值且通过质量、Top2 间隔后，才确认已有 ID |
| 强步态 Top2 间隔 | `0.08` | 防止两个已有 ID 分数过近时强行确认 |
| 开放集最大冒认相似度 | `0.90` | 任一 formal gait 原型达到此原始余弦上限时，禁止自动创建新身份；部署前必须用目标域冒认数据标定 |
| 强步态质量 | `0.70` | 新人物拒识、旧人物确认和自动建号共同使用的质量门 |
| 部分步态质量下限 | `0.35` | `Q_gait` 低于此值为 `INVALID`；达到此值但未到强门时为 `PARTIAL` |
| 视觉身份稳定样本数 | `8` | OSNet 自动生成视觉身份编号前需要的连续稳定外观样本数量 |
| 视觉身份稳定度 | `0.90` | OSNet 样本与视觉身份外观中心的最低一致性 |
| 视觉身份开放集相似度 | `0.90` | 新建视觉身份前，若 OSNet 与已有身份的原始相似度达到此值则继续等待绑定，避免重复编号 |
| 质量加权最小样本质量总量 | `0.70` | 窗口质量权重总量至少达到稳定样本数乘此比例，才允许形成稳定步态中心 |
| 允许 PARTIAL 样本贡献 | `true` | 允许 PARTIAL 以较低权重参与聚合；严重断轨/缺失帧仍会清窗 |
| 正式步态最大类内离散度 | `0.60` | 事件间离散度高于此值时不作为强步态确认依据 |
| 强确认最少事件支持数 | `1` | 强步态确认需要的独立事件/可信 formal 模板支持数 |
| 最低视角证据 | `0.30` | 当前视角证据低于此值时不允许强步态确认 |
| 外观冲突余弦门 | `0.35` | 高质量外观低于此原始余弦时，禁止 GaitGraph2 覆盖当前绑定；跨视角恢复依靠连续 Track 和多原型 |
| 步态暂可用事件数 | `2` | 同一视觉身份进入“步态暂可用”前需要的独立步态事件数 |
| 步态就绪事件数 | `3` | 进入“步态就绪”前至少需要的独立步态事件数，之后还要检查覆盖度、留出和开放集 |
| 步态条件覆盖数 | `2` | 步态就绪要求覆盖的摄像头/视角条件数量 |
| 步态事件最少真实帧 | `25` | 允许步态样本进入事件学习的真实有效姿态帧数；重采样帧不计入此数 |
| 步态身份检索真实帧 | `45` | GaitGraph2 承担正式身份检索前需要的真实有效姿态帧数 |
| 步态真实姿态覆盖率 | `0.75` | 有效姿态帧占 Track 窗口的最低比例，防止插值帧伪装成完整运动 |
| 步态事件一致性下限 | `0.70` | 独立步态事件与同身份事件的最低一致性，低于此值进入步态冲突 |
| 步态留出相似度下限 | `0.70` | leave-one-event-out 留出验证的 genuine 相似度下限 |
| 步态事件近重复上限 | `0.985` | 同条件新窗口达到此相似度时不重复增加事件 |
| 单图库步态近重复上限 | `0.985` | 只有步态余弦相似度低于此值、且强质量外观明确拒绝唯一旧 ID 时，才允许从单身份图库引入新号 |
| 单图库外观拒识阈值 | `0.30` | 单身份图库中，强质量外观低于此概率才可作为新人物的负证据；没有可靠外观时保持保守等待 |

生产视觉页还提供三项吞吐量参数：

- “检测推理间隔”默认值为 `1`，即每帧刷新 YOLO/ByteTrack。性能预算不足时可
  调大该值；中间帧使用短时运动预测框，但 ByteTrack 的遮挡恢复和 ID 稳定性
  必须按目标 FPS 实测，不能按调用次数推断。
- “外观提取间隔”默认值为 `6`。OSNet 首次取得外观后每六个 Track 帧刷新，
  没有需要刷新的目标时不会发起空推理调用。周期性刷新会优先安排在不执行
  YOLO 和 GaitGraph2 的帧，减少多个重模型同帧运行造成的卡顿。
- “步态推理间隔”默认值为 `3`。RTMPose 姿态仍逐帧采集，但 GaitGraph2 只在
  轨迹首次成熟或距离上次推理达到该间隔时刷新；同一帧到期的所有 Track 会合并
  为一个 GPU 批次。周期性刷新同样优先避开 YOLO 帧；首次成熟特征不会延迟。

生产 GaitGraph2 不接收整幅 RGB 图像、YOLO crop 或人体掩码。YOLO/ByteTrack
在整幅画面上生成并关联人体框；RTMPose 使用每个框的仿射 crop 做 top-down
姿态推理，再把 COCO17 关键点映射回原图坐标。当前 canonical pose 只做有限值和
置信度过滤，保留原始全帧坐标，随后以 `(T, 17, 3)` 姿态序列输入 GaitGraph2。
因此 `image_size=736` 是 detector 输入尺寸，不是 gait 输入尺寸；更换姿态坐标
契约或继续使用旧库时，必须使用新模型/新版本标识并隔离旧 gait gallery。

生产适配器会把长时间 Track gap、突发框跳变和运行时步态参数变化视为硬时序边界，
清空旧姿态/步态窗口并递增 `segment_id`。每个 `VisionTrack` 同时提供
`clean_subtracklet` 摘要，包含有效帧区间、真实姿态帧数、下肢帧数、视角覆盖和边界
原因；因此后续 gait 分支只能在当前 clean sub-tracklet 内取样，不能跨越污染区间复用
旧序列。该摘要不改变 GaitGraph2 的 raw full-frame `(T, 17, 3)` 输入契约。

步态学习和步态身份检索使用不同的真实帧门：GaitGraph2 可以在至少 25 个真实有效
姿态帧后生成一个学习候选，但固定长度重采样不会增加真实运动信息；当前查询至少需要
45 个真实有效姿态帧、`0.75` 真实姿态覆盖率和完整行走周期，才允许作为正式步态身份
检索。高质量外观与已绑定视觉身份明显冲突时，系统返回冲突/隔离，不让 gait 分支
越权改写身份。

错峰只是短暂顺延周期性刷新，并设置了最大延迟。如果把检测间隔调为 `1`，系统
仍会在达到延迟上限后执行外观和步态刷新，不会因为每帧检测而饿死特征分支。

检测和外观间隔调大可以提升吞吐量，但快速运动时复用框的滞后会增加，外观更新
也会变慢。步态间隔调大则会延长自动注册收集足够稳定步态样本所需的时间。

建议先观察默认值。若陌生人仍长时间显示“疑似 P…”，可小幅提高“新人物步态上限”；若同一人容易重复建号，则应降低该值，并先用目标摄像头数据检查步态概率分布后再考虑小幅降低“强步态确认概率”。校准中点和斜率会改变概率含义，不应只凭单段视频随意修改。

“填入默认值（未应用）”只会填写表单，仍需点击“应用到运行中”。当前版本的动态参数仅对本次进程生效，重启后恢复代码默认值；审计中的 `threshold_version` 会更新为 `runtime-vN`，便于区分调参前后的观测。诊断后端不支持的生产视觉参数会在页面中禁用。

默认数据库是：

```text
data/verifier-production-v1.sqlite3
```
需要指定其他数据库时：
```powershell
python -m cross_event_verifier gui --database data/my-gallery.sqlite3
```

## 视觉后端

对了，生产后端是默认值；缺依赖或权重时会明确报错
```powershell
#一般正常使用这一条即可(即默认)：
python -m cross_event_verifier gui

# 默认：允许自动注册
python -m cross_event_verifier gui --vision-backend production

# 自动回退：生产链路不可用时进入 HOG 诊断模式，并强制关闭自动注册
python -m cross_event_verifier gui --vision-backend auto

# 明确使用 HOG/前景轮廓，仅用于 GUI、摄像头和数据流排障
python -m cross_event_verifier gui --vision-backend demo
#其实大部分时候你可以直接在GUI里改
```

诊断后端的检测框和轮廓特征不具备身份注册精度，所以代码层面的 `supports_automatic_registration=False` 无法由 GUI 勾选绕过。

## 本地模型

生产链路要求以下文件位于 `models/`：

| 文件 | 用途 | 输出/说明 |
|---|---|---|
| `yolo11l.pt` | 人物检测 | YOLO11L，仅保留 person 类 |
| `bytetrack-cross-event.yaml` | 多目标跟踪 | 为跨事件场景设置的 ByteTrack 门限 |
| `rtmpose-s.onnx` | 17 点人体姿态 | ONNX Runtime CUDA 主图 + CPU 辅助节点的批量 top-down 推理 |
| `osnet_ain_x1_0_dg.pth` | 外观 ReID | 512 维 L2 归一化向量 |
| `gaitgraph2_grew_state.pt` | 时序步态 | 60 帧重采样、384 维 L2 归一化向量 |


## 模型资产与云端迁移

四个二进制模型合计约 154 MB，默认保存在项目的 `models/` 目录，并被 `.gitignore` 忽略；`models/manifest.json` 保存直接文件的字节数/SHA-256、转换 checkpoint 的 tensor 指纹和下载源。这样代码仓库不会被大文件和模型版本变更拖慢，但部署机器仍需要在启动前取得模型。

`requirements` 负责 Python 包依赖，模型属于外部大文件和许可资产，需要单独下载。所以克隆项目后的标准部署顺序就是：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m cross_event_verifier download-models
python -m cross_event_verifier doctor
```

`download-models` 默认从 manifest 中的固定 HTTPS 地址下载，并在写入 `models/` 前校验源文件 SHA-256；OSNet-AIN 和 GaitGraph2 会在源文件校验通过后转换为本项目运行时需要的 tensor-only state dict，再校验最终文件。已存在且校验通过的文件不会重复下载。

也可以只下载某一个文件或强制重新下载：

```powershell
python -m cross_event_verifier download-models --model yolo11l.pt
python -m cross_event_verifier download-models --model rtmpose-s.onnx --force
```

建议把模型放到私有对象存储、私有 GitHub Release 或私有 Git LFS。不要在没有确认许可和访问控制的情况下，把 YOLO/OpenGait 权重上传到公开仓库。当前工作区检测到 GitHub 远程地址，但没有可用的 Git LFS 上传凭据，因此本项目没有自动执行远程推送。

如果部署环境不能访问这些公共源，本项目也提供两个不依赖云厂商 SDK 的 PowerShell 工具，用于把同一份 manifest 资产切换到你自己的私有对象存储：

```powershell
# 1. 生成待上传的模型包，并在打包前校验 manifest 哈希
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_model_bundle.ps1 `
    -Output .\dist\model-bundle-v1.zip

# 2. 上传 model-bundle-v1.zip 解压后的 models/下四个二进制文件，
#    以及 manifest.json；然后用 HTTPS 前缀同步到本机并再次校验
$env:CROSS_EVENT_MODEL_BASE_URL = "https://your-private-bucket.example/models"
powershell -ExecutionPolicy Bypass -File .\scripts\sync_model_assets.ps1
python -m cross_event_verifier doctor
```

`CROSS_EVENT_MODEL_BASE_URL` 可以指向私有对象存储的临时签名 URL 前缀、私有 CDN，或 GitHub Release 的资产目录。同步脚本只会接受 manifest 中列出的文件，下载到临时文件并在 SHA-256 校验成功后替换目标；已正确存在的文件会跳过，哈希不匹配时默认停止，不会静默覆盖。

默认公共源与用途如下；下载源和上游许可随模型版本记录在 `models/manifest.json`：

| 模型 | 默认来源 |
|---|---|
| YOLO11L | Ultralytics 官方 assets release |
| RTMPose-s ONNX | Hugging Face `ziq/rtm` 仓库中的 ONNX 导出 |
| OSNet-AIN DG | deep-person-reid 使用的官方 Google Drive checkpoint |
| GaitGraph2 GREW | OpenGait 官方 Hugging Face checkpoint，下载后提取 model state |

如果选择 GitHub LFS，必须确认仓库可用 LFS 配额、模型许可允许上传，并在确认远程目标后执行：

```powershell
git lfs install
git lfs track "models/*.pt" "models/*.pth" "models/*.onnx"
git add .gitattributes models/manifest.json models/README.md
git add -f models/yolo11l.pt models/rtmpose-s.onnx `
    models/osnet_ain_x1_0_dg.pth models/gaitgraph2_grew_state.pt
git commit -m "Store production model assets with Git LFS"
git lfs push --all origin main
git push origin main
```

上面是有外部写入影响的人工确认步骤，脚本不会替你执行。更适合生产部署的做法是使用私有对象存储，并给运行机器发放只读、短期有效的下载 URL；密钥不要写进 README、代码或 SQLite。

## 离线开放集评测

生产阈值不应只依据闭集 Rank-1 或单个余弦分数调节。`evaluation.py` 提供
`evaluate_open_set_protocol()`，用于在 gallery、known probes 和 unknown probes
之间计算 Rank-k、FNIR、FPIR、known rejection 与 unknown acceptance。阈值应在独立
calibration unknown 集上确定；如果省略该集合，函数会把 unknown probes 作为小规模
探索性校准集，不应直接当作最终验收结果。

建议固定 identity/event/session-disjoint 的拆分，并至少报告 FNIR@1%FPIR、
FNIR@0.1%FPIR、Rank-1/Rank-20、最大冒认相似度、UNKNOWN→KNOWN、KNOWN→UNKNOWN、
AMBIGUOUS 比例、冲突率和自动注册误接纳率。连续帧或同一采集会话的多个窗口不能
被当作相互独立的注册事件。

```python
from cross_event_verifier import evaluate_open_set_protocol

report = evaluate_open_set_protocol(
    gallery_embeddings={"P1": (gallery_p1,), "P2": (gallery_p2,)},
    known_probes={"P1": (probe_p1,), "P2": (probe_p2,)},
    unknown_probes=(unknown_probe,),
    calibration_unknown_probes=(calibration_unknown_probe,),
    target_fpir=0.01,
    rank=1,
)
print(report.fnir, report.fpir, report.threshold)
```

## Python API

核心验证器仍可脱离 GUI 和重模型独立调用：

帧管线若要使用与生产 GUI 相同的视觉优先流程，应创建
`VideoVerifierPipeline(..., appearance_first=True)`；它会先绑定视觉身份，再
把同一 Track 的 GaitGraph2 输出登记为步态事件和步态原型。

非 GUI 调用方可以用 `verifier.evaluate_gait_readiness("P1")` 查看该视觉身份
的事件数、原型数、覆盖度、留出结果和当前步态状态。

```python
import numpy as np

from cross_event_verifier import (
    CrossEventVerifier,
    FeatureBundle,
    Observation,
    TrackQuality,
)

verifier = CrossEventVerifier()
verifier.register_identity(
    "P1",
    FeatureBundle(
        appearance=np.array([1.0, 0.0, 0.0]),
        gait=np.array([0.0, 1.0, 0.0]),
    ),
)

decision = verifier.verify(Observation(
    camera_id="cam-a",
    capture_session_id="session-a",
    track_id="track-7",
    features=FeatureBundle(
        appearance=np.array([0.98, 0.10, 0.0]),
        gait=np.array([0.0, 0.98, 0.10]),
    ),
    quality=TrackQuality(
        detection_confidence=0.92,
        box_height=180,
        frame_count=30,
        gait_cycles=2,
        walking_ratio=0.90,
        keypoint_visibility=0.88,
        occlusion=0.05,
    ),
))

print(decision.kind, decision.identity_id, decision.score)
```

真实视频管线应使用稳定的 `candidate_id` 关联跨事件候选；否则系统会为每个事件创建独立候选，避免未经授权的自动合并。使用 Python 管线时，在每次换源时显式传入同一个候选键：

```python
pipeline.set_source(
    "camera-b",
    capture_session_id="event-2",
    candidate_id="enrollment-task-42",
)
```

该键必须来自采集任务或操作员确认，不能由 ByteTrack 的短期 `track_id` 推导。多人同框或 Track 发生不确定切换时，管线会回退到会话级候选键并等待新的安全关联。

## 目录结构

源码现在按职责组织在同一个 Python 包中；模块之间通过公开值对象、视觉适配器、验证器和帧管线的稳定 seam 协作。完整的职责、依赖方向和稳定接口见：

- [模块地图](docs/architecture/module-map.md)
- [组内协作约定](docs/architecture/contributor-guide.md)
- [共同研究入口](CONTRIBUTING.md)

```text
cross_event_verifier/
├── engine.py、config.py、calibration.py、fusion.py、assignment.py、evaluation.py
│                         # 身份验证、多证据开放集决策与离线评测
├── gait_graph.py、aggregation.py、osnet_ain.py、memory.py、stability.py
│                         # GaitGraph2、质量聚合、外观表征与派生原型记忆
├── production_vision.py、vision.py、vision_factory.py、adapters.py
│                         # YOLO、ByteTrack、RTMPose 与视觉适配器
├── pipeline.py、automation.py、appearance_first.py、gait_readiness.py
│                         # 帧编排、视觉身份绑定、步态事件与就绪判定
├── runtime_parameters.py、media.py
│                         # 线程与运行时参数管理
├── gui.py、cli.py、storage.py、vector_index.py、model_assets.py
│                         # GUI、CLI、SQLite、向量索引与模型资产
├── types.py             # 公开值对象和模块间接口契约
├── __init__.py          # 公开 Python 门面
└── __main__.py          # python -m cross_event_verifier 入口
docs/architecture/
├── module-map.md        # 模块分组、依赖方向、稳定 seam 和测试入口
└── contributor-guide.md # 分工、实验记录和合并检查
models/
├── manifest.json
├── bytetrack-cross-event.yaml
└── README.md
scripts/
├── prepare_model_bundle.ps1 # 校验并生成待上传 zip
└── sync_model_assets.ps1    # 从 HTTPS/私有对象存储同步并校验模型
```

## 验证

```powershell
python -m pytest -q
python -m unittest discover -s tests -v
python -m compileall cross_event_verifier tests
python -m cross_event_verifier demo
python -m cross_event_verifier doctor
```

测试覆盖质量门控、质量加权聚合、校准与动态融合、多证据开放集拒识（含 FNIR/FPIR）、全局指派、OSNet 建议权、步态样本/事件/修订/派生模型、步态就绪判定、新人物不粘连旧 ID、运行时参数事务、GUI 参数页、人工登记、防污染图库、SQLite 回滚、混合特征维度迁移，以及生产适配器的批量步态编码、推理间隔和时序特征输出。

## 常见问题

- `doctor` 报模型哈希不符：不要继续使用该文件；重新从可信资产源同步，或恢复 `models/manifest.json` 对应版本。
- 只有上半身或脚踝出画：系统会保留检测框和 Track，但步态质量为零，不会自动建号。
- 首次启动较慢：YOLO、RTMPose、OSNet-AIN 和 GaitGraph2 会在采集线程首次使用时加载；RTX 5060 冷启动通常需要数秒，之后才进入稳定帧率。
- 仍看到 CPU 占用：摄像头读取、OpenCV 预处理、ByteTrack/SQLite/Tk 以及 RTMPose 少量 shape 辅助节点会使用 CPU；用 `doctor` 查看 `cuda_available`、`onnx_providers`，并确认生产后端不是 `demo` 或 `auto`。
- 画面能显示但视觉身份没有进展：检查 OSNet 外观是否稳定、人物框是否完整、检测质量是否达到强外观门；稳定样本数达到“视觉身份稳定样本数”后才会生成 `P` 编号。
- 视觉身份已确认但步态没有进展：检查人物是否实际行走、是否持续产生至少 25 个有效姿态帧、下肢是否完整，并观察 GUI 中的步态质量提示。
- 步态已有向量但没有用于身份确认：这是安全门控的预期行为；确认正式身份检索还需达到 45 个真实有效姿态帧和 0.75 姿态覆盖率，插值到 60 帧不算真实帧。
- 高质量外观与步态指向不同身份：系统会进入冲突/隔离，不会自动选择分数更高的一侧；先检查 Track 是否发生 ID switch，再进行人工复核。
- 同一人转为侧面/背面后没有生成新 P：连续可靠 Track 会保持视觉身份，并在累计稳定样本后将新视角写入该 P 的多原型外观图库。
- 步态事件显示为“已计数”：表示当前会话已经产生一个事件；后续窗口可以追加该事件的修订，但不会增加独立事件数。该视觉身份的步态模型会依据事件事实重新派生。
- 步态状态为“冲突”：独立事件之间的离散度或分支证据不足，系统会暂停写入并清空当前窗口，需要复核 Track 是否发生切换以及姿态输入契约是否一致。
- 需要只排查摄像头或 GUI：使用 `--vision-backend demo`；该模式强制禁止自动注册，不应拿来建立正式图库。

## 精度与许可边界

- 默认阈值和开放集 max-impostor 上限只能作为可运行起点。部署前应使用目标摄像头数据按 FAR/FPIR 和 FNIR@FPIR 协议拟合分支校准器，并评估换衣、遮挡、逆光、拥挤和跨镜头条件；生产可设置 `VerifierConfig(require_calibrated_scores=True)` 强制拒绝启发式校准。
- 当前不通过替换模型解决 GaitGraph2 的域差异。`cross_event_verifier.evaluation` 提供 `compare_encoder_embeddings()` 和 `evaluate_open_set_protocol()`，可记录 genuine/impostor similarity、max-impostor、d-prime、Rank-k、FNIR@FPIR 和 FPIR；当前生产模型不会因此自动改变。
- 方案二的姿态归一化和 2D→3D 姿态迁移暂不进入生产链路；当前 GaitGraph2 继续使用版本化的 2D raw full-frame 坐标契约。方案一的 GPGait++、SkeletonGait++、PSGait-like 分支属于后续 P2 研究，待模型资产、输入链路和目标域 A/B 协议齐备后再评估。
- 视觉身份自动编号要求连续稳定的 OSNet 外观样本；步态学习还要求完整人体和有效行走序列。只看到上半身时不会产生可靠步态原型。
- Ultralytics 代码/模型涉及 AGPL-3.0 或企业许可；OpenGait 官方资源声明限学术用途。商业部署前必须完成对应授权或替换模型。完整第三方归属与许可提示见 `THIRD_PARTY_NOTICES.md`。

## 仍需目标数据验证
- RTMPose-S 姿态分布是否适合 GREW GaitGraph2，以及默认开放集上限是否满足目标 FAR/FPIR。
- 24/30/60 FPS、遮挡、拥挤和低照度下的真实 P95 延迟、ID switch 与 FNIR。
- GUI 外观与交互仍待下一版本优化。
- 遮挡场景仍需继续评估吞吐量与轨迹精度的平衡。
- 本地硬件差异可能影响冷启动和吞吐量，部署前应按目标设备重新测量。

## 写在最后
- 本项目曾在华硕天选 Air 2025 H350 版本上运行；不同设备的加载时间和吞吐量请以 `doctor` 与目标数据实测为准。
