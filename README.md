# FloorFlow：外轮廓条件下的家装图纸逆向生成

本项目的目标不是识别户型外轮廓，而是**把已知的户型外轮廓作为生成条件，直接生成轮廓内部的完整家装图纸**，包括房间分区、内墙、门窗和固定家具。

形式化定义为条件 flow matching：

```text
c   = 已知户型外轮廓
x1  = 目标语义家装图纸
x0  ~ N(0, I)
xt  = (1 - t) x0 + t x1
vθ(xt, t, c) ≈ x1 - x0
```

这与原 `multiflow` 中“蛋白质口袋条件 → 配体生成”的关系一致：

| 药物设计 | 家装图纸逆向设计 |
|---|---|
| protein pocket | 房间外轮廓 |
| ligand coords / topology | 房间分区 / 内墙 / 开洞 / 家具 |
| pocket-conditioned velocity | contour-conditioned velocity |
| Vina / PoseBusters | room IoU / opening F1 / envelope validity |

外轮廓是已知输入，因此生成后会用确定性后处理回填外墙并清空轮廓外区域，模型容量集中在内部设计，而不是边界检测。

> 当前状态（2026-11-02）：分层多阶段 pipeline 已通过 frozen 503 发布 gate。
> 主口径是 **contour + room program + coarse room/structure/furniture layout
> control**，不是 contour-only 精确重建；contour-only 继续作为诊断口径单独报告。
> 下表是 layout-conditioned 诊断结果，不应改写为 target-free contour-only
> 重建或 CVPR 主 claim；当前 target-free 证据与主张边界见文末实验状态链接。
>
> | 指标 | Frozen 503 |
> |---|---:|
> | Room macro IoU | 0.9718 |
> | Instance geometric IoU / F1 / type | 0.9711 / 0.9740 / 1.0000 |
> | fpeval wall / opening F1 @0.05 | 0.9637 / 0.9724 |
> | fpeval polygon-room F1 | 0.9818 |
> | Furniture IoU / type accuracy | 0.9672 / 0.9774 |
> | Room-support IoU | 0.9741 |
> | Validity / conditional watertight | 1.0000 / 0.9595 |
>
> GT 自身在 fpeval verifier 下只有 0.8827 watertight，因此同时报告原始
> watertight 0.8986 和只在 GT 可闭合样本上的条件值。完整证据见
> [`outputs/reports/a5_layered_pipeline_gate_test503_v2.json`](outputs/reports/a5_layered_pipeline_gate_test503_v2.json)。
> skv4 296 独立复现中，284 个带 GT room 的 plan 上
> room/instance/F1/type 为 0.9857/0.9675/0.9700/0.9894；其余 12 个 plan
> 没有 GT room，room 指标不参与平均。wall/opening 为 0.9799/0.9648。
>
> 当前 CVPR 提交口径见
> [`docs/cvpr_experiment_status_20261102.md`](docs/cvpr_experiment_status_20261102.md)。
> Oracle-control 编辑协议下，五个操作、五个 seed 的 Delta-IoU 为
> `0.956703-0.999801`、Delta-MSE 为 `0.000019-0.000096`，数值上超过
> HouseMind-E 公开协议锚点 `0.608/0.0019`；但数据与输入不同，且该轨道不可部署。
> 完整协议与 claim 边界见
> [`docs/fp_oracle_control_protocol_status_v1.md`](docs/fp_oracle_control_protocol_status_v1.md)。

当前发布 pipeline 包含四个独立阶段：

1. room-program + 224 coarse room layout → 256 room refiner；
2. 128 coarse wall/opening/type layout → 256 structural refiner；
3. 192 coarse furniture layout → 256 furniture refiner；
4. 192 structural layout → 独立 room-support refiner，再显式组合 room、wall、
   opening、furniture 四个输出层。

最后的多层组合只负责渲染，不覆盖 room 层的存储。MiTA 已迁移到语义流
cross-attention；191-val 同预算对照中 room/instance/F1/type 全部优于 MHA。

当前仍有明确的科学边界：去掉 target-derived coarse layout 后，noise CFM、
contour+program predictor 和 guided CFM 的 room IoU 分别只有
0.0607/0.1630/0.7196；train-only 轮廓邻域审计中严格 `IoU>=0.95` 的可识别
比例只有 0.596%。因此当前 0.95 系列是 **layout-conditioned refinement**
结果，不是纯轮廓精确重建。证据见
[`outputs/reports/b1_contour_cfm_blocker.json`](outputs/reports/b1_contour_cfm_blocker.json)
和 [`outputs/reports/b5_cvpr_readiness.json`](outputs/reports/b5_cvpr_readiness.json)。

## 当前实现

- CubiCasa5K SVG 原生解析，不依赖原仓库的旧训练栈。
- `BoundaryPolygon` 同时支持 `<polygon>` 和 `<path>`；path 曲线以分段端点近似，恢复此前漏掉的卫浴、马桶和 `furniture:Misc` 标注。
- 65 类语义空间：背景、墙、8 种门、3 种窗、24 个房间组、28 个固定家具组和结构。
- 轮廓条件 U-Net，多尺度条件融合，以及迁移自 `multiflow` 思路的 gated cross-attention。
- 64 个前景语义通道 token；类别由真实数据频次自动构建，低于阈值的稀有类型有显式 rare 通道。
- 跨通道注意力使用建筑语义关系初始化，例如门/窗到墙、卫浴对象到卫生间、柜体到厨房/储藏、壁炉到起居空间。
- 从 `multiflow` 迁移 batched `soft_slot_decompose` 与双向 gated slot cross-attention，在空间特征和轮廓特征之间动态形成 16 个语义槽。
- endpoint 分类采用按通道正样本比例加权的 object-balanced loss，避免马桶、淋浴、壁炉等小目标被大面积房间淹没。
- Heun / Euler conditional flow matching 采样。
- 支持 standard velocity 与 Dirichlet simplex-categorical flow 两种目标。
- 动态多尺度输出头：3/5/7 等并行感受野分支由 router 产生逐位置、逐语义通道的 softmax 权重；尺度选择完全由模型学习。
- 外墙与轮廓外区域硬约束。
- SDF/inside-outside depth、楼层 ID 条件和 classifier-free guidance。
- semantic endpoint 与 condition-only endpoint 同步头。
- Mask2Former 风格 room query-mask 解码器：多尺度 memory、masked cross-attention、逐层监督、Hungarian mask loss 和 boundary-aware Dice/BCE。
- coarse partition 作为强先验，query-mask 通过零初始化的 learned scale 只学习残差，初始输出不会破坏现有 room baseline。
- 从 MIT LayoutBridge 适配的 DDI 双流结构注入、各向异性方向修正和高/低频重建损失。
- coarse-to-fine semantic anchor：condition-only 输出作为稳定先验，flow 通过 learned gate 只学习残差；内部墙、开洞和家具使用独立 group heads。
- wall、wall-direction、door-wall contact、furniture-room containment、room-boundary、room-adjacency 辅助头。
- warmup-cosine、梯度累积、noise repeats、loss/model EMA、全局类别权重与数据预加载。
- dense partition 与 object heads 使用 gradient isolation；object 训练不会覆盖
  room backbone。
- room coverage 使用 target-room recall，另报 generated/target occupancy；
  furniture containment 在 room-support 通道上计算。
- 室内房间 IoU、内墙 IoU、门/窗/家具 IoU、开洞 F1、房间面积覆盖率等评估。
- 楼层级全量预处理：train 5,126、val 491、test 503，0 个解析失败。

## CAD08 结构化编辑（实验性）

仓库还提供一个独立的 CPU CAD 状态事务接口，支持有界房间重分类、拆分、合并，以及绑定到已注册墙宿主的开口新增和替换。接口维护房间实例、谱系、别名和开口—墙宿主关系，并可将有明确历史收据的二维状态写为 R2010 DXF 与 sidecar 后读回。它不调用或训练上述生成模型。

该接口不会从预测图推断用户身份：初始化生成的对象 ID 是运行时 ID；开口操作必须收到当前状态中的明确 host ID；无法唯一绑定时保留失败或 unknown。R2010 校验使用仓库自己的 reader，不代表独立 CAD 软件兼容性或真实 CAD 安全认证。安装可选依赖和运行范围说明见 [`docs/cad08_structured_editing.md`](docs/cad08_structured_editing.md)。

```bash
uv sync --extra dev --extra cad
uv run pytest tests/test_cad08_cpu_import.py tests/test_cad08_integrated_v1.py \
  tests/test_cad08_opening_lifecycle.py \
  tests/test_cad08_history_export.py tests/test_cad08_history_dxf_readback.py -q
```

## 数据

项目中的 `.data/cubicasa5k` 用于放置本机完整解压数据：

```text
./.data/cubicasa5k
```

`.data/cubicasa5k_source` 是同一目录的兼容链接。`.data/` 和生成的
`outputs/` 已被 Git 忽略。需要从原始 ZIP 重新解压时，将数据放到：

GitHub 仓库只提交源码、测试、配置、论文、文档和体积较小的冻结提交证据。
`third_party/` 中的第三方源码树、数据集和 checkpoint、`.data/` 下的大文件、
`outputs/` 中的缓存与训练产物、虚拟环境及 wandb 缓存均不进入版本控制；所需外部依赖按
[`third_party/README.md`](third_party/README.md) 列出的固定 revision 单独获取。

投稿证据的唯一来源索引为
[`docs/floorplan_submission_source_registry_v2.json`](docs/floorplan_submission_source_registry_v2.json)，
冻结配置为
[`configs/floorplan_submission_evidence_freeze_v2.json`](configs/floorplan_submission_evidence_freeze_v2.json)。
对应的关键审计结果以显式路径提交到 `outputs/reports/`；完整匿名包、缓存、checkpoint
和逐案例大型 JSON 仍保留在本机 `outputs/` 中。

```bash
mkdir -p .data
unzip /path/to/cubicasa5k.zip -d .data/cubicasa5k
```

每个样本缓存为：

```text
target:    uint8 [256, 256]  # 65 类语义标签
footprint: uint8 [256, 256]  # 已知建筑外轮廓
floor id:  1..4              # 楼层条件
```

## 环境

本机训练使用现成的 `pocket311` 环境。执行下文命令前先激活该环境；如使用其他
Python 3.11 环境，请确保其具备相同的 PyTorch/CUDA 依赖并直接替换 `python`：

```bash
python --version
```

该环境提供 Python 3.11、PyTorch 2.11、CUDA 12.8、NumPy 和 Pillow。脚本已自动加入 `src/`，不需要先安装 Python 包。

## 运行

### 1. 预处理

全量 path-fix 缓存位于 `outputs/cache/cubicasa5k_full_v2`（5126/491/503，0 解析失败）；
`outputs/reports/data_audit.json` 记录旧 cache 到新 cache 的类别像素变化。重新生成：

```bash
python scripts/preprocess.py \
  --cache-root outputs/cache/cubicasa5k_full_v2 \
  --splits train val test \
  --image-size 256 \
  --workers 8 \
  --room-min-count 100 \
  --furniture-min-count 100
```

### 2. 训练

当前完整配置：

```bash
python scripts/train.py \
  --cache-root outputs/cache/cubicasa5k_full_v1 \
  --image-size 256 \
  --epochs 8 \
  --batch-size 32 \
  --workers 8 \
  --base-channels 48 \
  --levels 3 \
  --num-slots 16 \
  --objective-mode simplex \
  --gradient-accumulation-steps 1 \
  --noise-repeats 2 \
  --topology-aux-weight 0.10 \
  --semantic-head-weight 1.0 \
  --direct-semantic-weight 1.0 \
  --output-kernels 3 5 7 \
  --preload-data \
  --amp \
  --output-dir outputs/train/floorflow_full_v1
```

如果显存不足，先降 `--batch-size`；不要先降分辨率，因为 256 下细墙已经接近像素级瓶颈。训练产物：

```text
outputs/train/floorflow_full_v1/best.pt
outputs/train/floorflow_full_v1/last.pt
outputs/train/floorflow_full_v1/ema.pt
outputs/train/floorflow_full_v1/train.jsonl
```

语义通道跨注意力和 slot mixing 默认开启。做消融时分别加
`--disable-semantic-channel-attention`、`--disable-slot-mixing`，其余配置保持一致。
动态输出头可通过 `--disable-dynamic-output-head` 回退到固定 3×3 输出。

## 模型调优实验顺序

不要先跑 503-case。当前 `todo.json -> model_tuning_sequence` 定义了固定的门控顺序：

1. 检查数据/标注覆盖率与 loss/head 是否产生非零梯度；
2. 验证 room partition 和 query-mask head 在小样本上不坍塌；
3. 用 `--room-partition-weight`、`--room-mask-weight`、
   `--room-mask-partition-weight` 逐项接入主模型；
4. 用 `--layoutbridge-refinement --frequency-low-weight ... --frequency-high-weight ...`
   加入方向与边界高频约束；
5. 用 `--room-balanced-sampling` 处理稀有房间；
6. 只有每一步的 val 诊断通过后，才运行 frozen 503-case。

例如先做 256/96 短训练，而不是直接全量：

```bash
python scripts/train_room_mask_decoder.py \
  --checkpoint outputs/train/floorflow_full_v1_direct/best.pt \
  --cache-root outputs/cache/cubicasa5k_full_v2 \
  --limit-train 256 --limit-val 96 --epochs 3 \
  --partition-weight 1.0 \
  --output-dir outputs/train/room_mask_decoder_256_dense
```

### 3. 当前分层 pipeline 的验证与冻结测试

```bash
PYTHONPATH=src python \
  scripts/evaluate_layered_pipeline.py \
  --room-checkpoint outputs/train/layout_refiner_full_s224/best.pt \
  --structure-checkpoint outputs/train/coarse_structure_type_full_v1/best.pt \
  --furniture-checkpoint outputs/train/coarse_furniture_full_s192_v1/best.pt \
  --support-checkpoint outputs/train/coarse_support_full_v1/best.pt \
  --split test \
  --layout-size 224 --structure-size 128 \
  --furniture-size 192 --support-size 192 \
  --wall-threshold 0.6 --opening-threshold 0.5 \
  --furniture-threshold 0.5 \
  --output-dir outputs/eval/layered_pipeline_test503_frozen_v2
```

生成发布 gate：

```bash
PYTHONPATH=src python \
  scripts/build_layered_pipeline_gate.py \
  --metrics outputs/eval/layered_pipeline_test503_frozen_v2/metrics.json \
  --per-case outputs/eval/layered_pipeline_test503_frozen_v2/per_case.json \
  --output outputs/reports/a5_layered_pipeline_gate_test503_v2.json
```

### 4. 旧版单模型评估

```bash
python scripts/evaluate.py \
  --checkpoint outputs/train/floorflow_full_v1_direct/best.pt \
  --cache-root outputs/cache/cubicasa5k_full_v1 \
  --split test \
  --batch-size 8 \
  --workers 8 \
  --steps 12 \
  --semantic-decode-weight 1.0 \
  --output-dir outputs/eval/full_v1_direct_claim
```

结果在 `metrics.json`、`per_case.json` 和 `examples/`。主指标应优先看：

- `room_macro_iou_mean`
- `room_area_coverage_mean`
- `interior_wall_iou_mean`
- `opening_f1_mean`
- `furniture_iou_mean`
- `outside_footprint_fraction_mean`，应恒为 0

当前 smoke 训练只用于验证代码闭合，低指标没有科学意义。

### 5. 任意外轮廓推理

```bash
python scripts/infer.py \
  --checkpoint outputs/train/floorflow_full_v1_direct/best.pt \
  --footprint /path/to/outer_contour.png \
  --samples 8 \
  --steps 16 \
  --semantic-decode-weight 1.0 \
  --output-dir outputs/infer/case_001
```

默认把白色像素视为建筑内部、黑色视为外部。若输入是黑白线条轮廓，加
`--fill-outline`；若前景为黑色，加 `--foreground black`。

解码默认使用模型学到的多尺度路由，`--smooth-kernel` 默认为 1；只把它当独立的后处理消融项。

检查每个类别学到的尺度选择：

```bash
python scripts/inspect_scale_router.py \
  --checkpoint outputs/train/floorflow_full_v1_direct/best.pt \
  --limit 32
```

该诊断报告会输出每个类别在 3/5/7 核上的平均路由权重和当前偏好核。

### 6. 合成长尾预训练

`third_party/fpvec-lab/fpgen` 可生成带像素对齐 GT 的合成平面图。安装
`.[synthetic]` 后生成并转换：

```bash
PYTHONPATH=third_party/fpvec-lab python -m fpgen generate \
  -n 2048 --seed 910 --style clean_cad:0.4,marketing:0.4,brochure:0.2 \
  --degrade light --out outputs/synthetic/fpgen --run-id pretrain2k --workers 4

python scripts/preprocess_fpgen.py \
  --input outputs/synthetic/fpgen/pretrain2k \
  --output-cache outputs/cache/fpgen_pretrain2k_v1
```

同一训练脚本通过 `--synthetic-cache outputs/cache/fpgen_pretrain2k_v1`
使用合成 split。当前 2K 合成预训练在 fine-tune CubiCasa5K 后未超过真实数据
baseline，因此只保留为实验路径，不是默认主模型。

### 生成评估与 best-of-N 边界

外轮廓到室内布局是一对多映射。主评估使用固定种子单样本，或同一条件下全部生成
样本的均值/中位数、standard error、MMD/energy、bounded coverage/density；
不对候选取 max。固定预算的 best-of-N 只用于 appendix ceiling 诊断，learned
selector 会形成 reward-tilted distribution，必须单独声明和评估。数学定义与
gate 约束见 [`docs/sampling_math.md`](docs/sampling_math.md)。

## 代码结构

```text
src/floorflow/
  svg.py          CubiCasa SVG → target / exterior contour
  data.py         split、缓存和数据集
  model.py        条件 U-Net + gated condition attention
  slots.py        multiflow semantic slot decomposition / gated attention
  flow.py         CFM loss、Heun 积分、解码
  sampling_policy.py 单样本/分布/ceiling 预算与 fail-closed 审计
  room_partition.py  dense room partition 头、目标、损失与语义覆盖
  room_groups.py  fine-to-functional room group 指标与消融损失
  room_program.py  可选 room presence/program 先验
  room_segmenter.py  ResNet18-FPN 容量对照基线
  contour_memory.py  train-only top-1 轮廓检索与确定性 room prior
  condition_evaluation.py  validation 指标、room-support 与 checkpoint 选模
  room_mask_decoder.py  Mask2Former 风格 room query-mask head
  layoutbridge_adapters.py  LayoutBridge DDI、anisotropic 与 frequency-loss 适配
  anchor_fusion.py  semantic anchor 与 flow 的 learned residual fusion
  semantic_attention.py  多通道语义注意力与 slot mixer
  structured_decode.py  room mask + object heads 的统一推理合成
  geometry.py     外墙回填与轮廓硬约束
  masks.py        外部轮廓图像读取
  metrics.py      内部结构评估
scripts/
  preprocess.py
  train.py
  evaluate.py
  infer.py
  train_room_partition.py
  evaluate_room_partition.py
  train_room_mask_decoder.py
  inspect_scale_router.py
  sample.py
tests/
```

外部参考代码、commit 和许可证记录在 [`third_party/README.md`](third_party/README.md)。

设计取舍、与原项目的对应关系以及后续拓扑增强路线见
[`docs/architecture.md`](docs/architecture.md)。

早期固定头与动态多尺度头的 400-case 探索对照见
[`docs/results.md`](docs/results.md)。

楼层级、65 类、simplex flow 和完整 503-case 发布 gate 结果见
[`docs/full_v1_results.md`](docs/full_v1_results.md)。

最新 structured dense-partition 结果见
[`docs/full_v2_results.md`](docs/full_v2_results.md)。

下一阶段的数据补全、训练调优和可发表性门槛见根目录的
[`todo.json`](todo.json)。
