# FloorFlow 架构与迁移说明

## 1. 任务边界

输入是户型外轮廓 `c`，输出是完整的内部语义图纸 `x1`。轮廓在推理时由用户或 CAD 前端提供，不是模型预测结果。

模型需要生成：

- 房间功能分区；
- 内部隔墙；
- 门和窗的位置；
- 固定家具；
- 各语义对象之间的几何一致性。

生成后必须满足：

```text
x1(u) = background,   u ∉ c
x1(u) = exterior_wall, u ∈ boundary(c)
```

因此外墙和轮廓外区域由 `compose_known_exterior` 确定性处理，主损失在外墙带降权。模型的核心监督区域是轮廓内部。

## 2. 与原 multiflow 的对应

本次迁移保留原项目的建模逻辑，而不是直接复制分子张量、RDKit、GNINA 和蛋白质物理约束：

| multiflow 组件 | FloorFlow 迁移 |
|---|---|
| `pocket_coords` | `footprint [B,H,W]` |
| ligand coordinate / topology state | semantic floorplan channel map |
| `FlowState(coords, x0_coords, t)` | conditional interpolation `xt` |
| pocket encoder + gated cross-attention | contour encoder + gated condition attention |
| `sample_flow` Euler/Heun integration | `sample_floorplans` |
| contact / wall constraints | outside masking、exterior-wall composition、room coverage |
| PoseBusters / Vina | room IoU、opening F1、envelope validity |

没有迁移的部分包括配体原子词表、化学键、Vina/GNINA、ADMET、蛋白口袋点云和分子后处理。这些组件依赖的具体语义在家装任务中不存在。

直接参考并适配的源码位置：

```text
/home/hugo/codes/multiflow/pocket_model_core.py
  soft_slot_decompose             -> src/floorflow/slots.py
  gated_cross_attention           -> src/floorflow/slots.py
  AtomPocketCrossAttention...     -> semantic channel + condition attention
  CrossModalFlowMixerHead gates   -> semantic channel relation gates
  sample_flow                     -> src/floorflow/flow.py::sample_floorplans
```

迁移时保留了槽分配、active-slot mask、双向 gated attention 和 Heun 积分，只是把分子 token 换成二维 floorplan token。

## 3. 数据表示

CubiCasa5K 的 `model.svg` 按 `Floor-N` 拆分成独立样本，直接解析为 `uint8 [H,W]` 标签图。标签空间不再手写固定编号，而是由 `taxonomy.py` 扫描全量数据生成：

- 背景和 wall；
- 全部门/窗子类型；
- 原始房间类型；低于阈值的类型归入 `room_rare`；
- 原始 49 种 FixedFurniture；低于阈值的类型归入 `furniture_rare`；
- structure。

当前默认阈值为房间出现次数 `100`、家具出现次数 `100`，生成 65 个类别。完整映射、计数和 rare 类列表保存在 `outputs/cache/cubicasa5k_full_v1/dataset_report.json`。

外轮廓由非 `Outdoor` 房间多边形并集生成，填洞后只膨胀到外墙厚度，避免把户外平台、露台和 detached railing 当作室内区域。

## 4. 生成模型

`FloorplanFlowModel` 使用三级条件 U-Net：

1. 输入为 64 个前景通道；simplex 模式下从 Dirichlet 单纯形先验采样。
2. 条件为轮廓 occupancy、边界、inside/outside depth 和归一化二维坐标，并加入楼层 ID embedding。
3. 轮廓通过独立编码器逐尺度注入主 U-Net。
4. bottleneck 使用多头 gated cross-attention，结构对应 `multiflow` 中 pocket-to-ligand 的 cross-attention velocity head。
5. 空间特征和轮廓特征分别通过 `soft_slot_decompose` 形成 16 个动态槽，并在两个方向执行 gated slot attention。
6. 64 个输出语义通道各有独立 query token，先对空间特征做 cross-attention，再在通道 token 之间做 self-attention，最后把通道间联合上下文反馈到空间速度场。
7. 通道 self-attention 带有可学习的建筑语义关系先验：门/窗与墙，卫浴设施与卫生间，柜体与厨房/储藏，壁炉与起居空间等关系在训练开始时已经连通。
8. 输出为 64 通道速度场，并带 semantic endpoint、wall、opening-contact、containment、room-boundary、adjacency 等同步头。
9. 训练采样 `t ~ U(0,1)` 或 logit-normal，目标是 `x1 - x0`。
10. 推理支持高斯 velocity flow 与 Dirichlet simplex flow，使用 Heun 积分；condition dropout 支持 classifier-free guidance。
11. 输出头并行使用可配置的 3/5/7 等多尺度卷积分支；router 对每个位置、每个语义通道独立产生 softmax 尺度权重，房间可以自动选择大感受野，门窗可以自动选择小感受野。
12. condition-only semantic head 提供稳定的布局锚点，flow 路径负责约束内的细化。
13. 解码默认直接 argmax；`smooth_kernel` 仅保留为独立后处理消融，不承担学习头尺度选择。

当前 distribution-CFM 主干审计为 6.462M 参数（含 coupled heads 与 style
adapters）；frozen contour room segmenter 为 12.395M，
structure/furniture/support refiners 合计 1.507M，MiTA graph adapter 为
0.023M，动态家具 query head 为 0.277M，trust gate 为 0.001M，完整部署栈
合计 20.664M。训练数据为
CubiCasa5K 的 5,126 个 train 户型、491 个 val 和 503 个 test 户型。

动态输出头的权重满足：

```text
g[k, u, c] = softmax_k(router(features)[u, c, k])
v_out[u, c] = sum_k g[k, u, c] * Conv_k(features)[u, c]
```

因此尺度不是按通道写死的规则，而是由训练自动决定的空间自适应混合。

多通道交互分两步：

```text
spatial features -> class-specific spatial readout -> cross-channel attention
                 -> gated semantic feedback -> updated velocity features
```

这对应 `multiflow` 将 topology、geometry、pocket、contact 等信息分通道建模、再跨通道融合的做法。消融时可用 `--disable-semantic-channel-attention` 和 `--disable-slot-mixing` 分别关闭两个阶段。

## 4.1 层级通道与动态图

`CoupledDistributionHub` 同时学习 dense pixel distribution 和 semantic
element distribution。通道不是平权的：墙体先约束门窗候选，房间分区再约束
家具和 support；每个元素组使用独立 query set，并通过
`OfficialMiTARouteAttention` 读取父组 token。query 数量和 mask 容量由
predicted complexity 与 `allocate_group_token_budgets` 决定。

递归 decoder 生成 contour-only partition graph。图节点包含 parent、depth、
bbox、面积、局部语义 logits 和 recursion strength。可选的
`RecursiveSemanticGraphAdapter` 将节点 token 送入 MiTA route attention，
在邻接节点间传播语义证据，再以零初始化残差修正局部 room logits。

图分支不直接替换完整房间输出。最终解码把几何支持和语义分配解耦：

```text
geometry_gate(u) = base_support(u) or
                   (graph_support(u) and graph_confidence(u) >= tau)
semantic_token(component) = argmax mean(base_room_logits over component)
```

因此 graph 负责候选分区几何，base semantic head 负责类别；连通域内再投影语义，
避免“Instance geometry 改善但 Room macro IoU 崩掉”的耦合退化。默认
`tau=0.2` 来自 val64 的受控选择，test503 不参与选择。

主推理路径不再读取 train-bank 最近邻。`FootprintCrossChannelPrior` 将
footprint 与局部特征 token 编码成连续先验，并通过 MiTA 依次路由到
wall、opening、room、furniture 和 support。该模块采用零初始化 residual
scale，重启旧 checkpoint 时保持恒等映射；训练后才逐步接管先验。

语义元素 query 数由预测 family density 动态决定。家具使用 Mask2Former
风格的 room-conditioned instance query 与 Hungarian set loss；门窗使用
wall token 和 opening token 的 interval query，替代自由的栅格 blob。

```text
src/floorflow/recursion/partition.py
src/floorflow/recursion/graph.py
src/floorflow/recursion/graph_adapter.py
src/floorflow/inference/recursive_room_prior.py
```

CubiCasa5K 的固定家具标注中高频对象是 Toilet、Sink、Shower、Closet、Cabinet、ElectricalAppliance、SaunaBench 和 Fireplace。这个版本没有单独的 bed/sofa 通道，因为该数据集的 `FixedFurniture` 中没有稳定出现这两类；若未来混入其它数据，可将它们映射到 `furniture_other` 后再扩展专用通道。

## 5. 损失与硬约束

distribution-CFM 总损失：

```text
L = L_flow + L_endpoint + L_pixel + L_element + L_coupling
    + L_source + L_dynamic_route + w_struct * L_struct
```

其中：

- `L_flow`：预测速度与 `x1 - x0` 的加权 MSE；
- `L_endpoint`：由 `xt + (1-t)v` 得到的 endpoint 与 one-hot 目标的 BCE；
- `L_pixel`：case-macro CE、Dice、room/family Tversky 和可选 boundary；
- `L_element`：Hungarian set loss，监督 semantic query 的 mask、类型和 presence；
- `L_coupling`：要求 element masks 与 pixel distribution 一致；
- `L_struct`：可选的门窗-墙、房间-墙边和家具-房间结构项，推理不读取 GT；
- 外墙带权重默认为 0.1；
- 轮廓外速度恒为零。

这不是把边界识别塞进生成模型，而是让已知外墙保持正确渲染的同时，把主要梯度留给内部设计。

## 6. 评估

首要研究指标：

- `room_macro_iou`：有目标面积的房间类别平均 IoU；
- `room_area_coverage`：内部区域中被房间类别覆盖的比例；
- `interior_wall_iou`：去掉外墙带后的隔墙 IoU；
- `opening_f1`：门窗联合 F1；
- `furniture_iou`：固定家具 IoU；
- `outside_footprint_fraction`：轮廓外非背景比例，硬约束下应为 0；
- `exterior_band_is_known_wall`：外墙回填正确率，应为 1。

`interior_pixel_accuracy` 只作为辅助指标。背景占主导，不能单独用来证明设计质量。

## 7. 当前限制

1. 256×256 栅格仍会损失部分细墙和门扇几何，后续应比较 384 或向量/实例表示。
2. 房间类别是粗粒度映射，尚未预测房间邻接图。
3. 当前仅生成固定家具，不能生成软装、材质和真实三维家装对象。
4. 连续 one-hot flow 是快速 baseline，不等同于离散拓扑生成。
5. 生成结果还没有经过建筑规范约束，如最小门宽、疏散连通性、房间面积分布。

## 8. 下一阶段

按截止日期优先做以下顺序：

1. 完成 256 栅格 baseline 的全量训练、test 指标和可视化。
2. 增加房间实例头、相邻房间图头和门-墙接触约束，把“房间分区”从像素提升到拓扑对象。
3. 比较 256 / 384 与 Euler / Heun 的精度-延迟曲线。
4. 增加 connectivity、door-wall alignment、room-area distribution 等结构指标。
5. 若时间允许，再引入外部 VLM 做语义评审，但 VLM 只作为评估器，不承担唯一生成主体。
