# RGB-D 自适应静态特征补偿

仅在 RGB-D / IMU-RGB-D 左图提取器启用；YOLO、人体运动判断、9×9 膨胀及后端跟踪判据保持原样。两种参数加载路径（旧 YAML 和 `File.version: "1.0"`）均支持。示例 YAML 已加入参数，默认全部关闭；开启 `FAST`、`Quota`、`Feedback` 即启用完整机制，`Stats` 独立控制日志。

## 参数

以下键统一使用 `AdaptiveFeatures.` 前缀。开关使用整数 `0` / `1`。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| FAST | 0 | 静态候选不足的网格用 minThFAST 补检 |
| Quota | 0 | 根据动态面积和反馈调整每帧配额 |
| Feedback | 0 | 使用上一帧内点数和状态更新补偿强度 |
| Stats | 0 | 每帧输出统计及计时 |
| MinStaticPerCell | 5 | 每个金字塔网格的基础静态候选阈值，1–1000 |
| MaxFeatures | 2000 | Nmax，启用 Quota 时必须 ≥ ORBextractor.nFeatures，上限 100000 |
| DynamicAlpha | 1.0 | 动态遮挡比例权重，0–10 |
| FeedbackFastGain | 2.0 | 反馈对网格候选阈值的增益，0–10 |
| FeedbackQuotaGain | 0.5 | 反馈对目标特征数量的增益，0–10 |
| LowInliers | 60 | 内点数低于该值时加强补偿 |
| HighInliers | 100 | 内点数达到该值时逐步减弱，必须大于 LowInliers |
| RiseStep | 0.25 | 每个低质量帧增加的强度，(0,1] |
| DecayStep | 0.10 | 每个恢复帧减少的强度，(0,1] |

缺省参数使用上表默认值；非法类型、非有限数、超范围值会在启动时报告错误。步长实际接受范围为 0.001–1。Feedback 本身不强制开启 FAST 或 Quota；它仅影响已开启的机制。

## 提取流程

保留原来的 iniThFAST 检测及“原始 FAST 候选为空时使用 minThFAST”回退。经现有动态掩码检查后，若静态候选少于 `MinStaticPerCell * (1 + FeedbackFastGain * strength)`，且尚未执行低阈值检测，就使用 minThFAST 补检。同一网格中以金字塔坐标去重，保留原候选，新增点也必须通过相同掩码检查。minThFAST 不低于 iniThFAST 时不执行额外补检。动态区域无法补入静态角点时允许输出不足。

`rd` 是最终膨胀后动态掩码在原图范围内的并集面积比，重叠区域只计一次。静态人体不计入。Feedback 关闭或 strength=0 时：

```
Ntarget = round(min(Nmax, N0 * (1 + DynamicAlpha * rd)))
```

有反馈时：

```
Ntarget = round(min(Nmax, N0 * (1 + DynamicAlpha * rd + FeedbackQuotaGain * strength)))
```

使用金字塔逆尺度权重分配目标，以累计舍入保证各层配额之和等于 Ntarget；不改变 `mnFeaturesPerLevel`。仍然由原八叉树选点，启用 Quota 时若节点分裂造成超额，仅裁掉该层超出配额的低响应点。所有开关关闭时保留旧算法输出（正常尺寸输入）；小尺寸金字塔层增加了空层/除零保护。

候选坐标按原来的层尺度及边界偏移映射到原图。方向、描述子、最终坐标缩放和 Frame 的 RGB-D 深度关联流程不变，按实际特征数量分配数组。

## 反馈与统计

strength 限制为 [0,1]。正常跟踪且上一帧真正执行过内点计数时，低于 LowInliers 按 RiseStep 增加，达到 HighInliers 按 DecayStep 减少，中间区间保持。未执行计数的正常帧保持强度，不读取陈旧内点数。RECENTLY_LOST / LOST 使用最大强度并忽略旧内点数；初始化、新地图、Reset 和 ResetActiveMap 清零反馈历史。所有动作只影响下一帧提取，不改变跟踪成功、重定位及优化判据。

`[AdaptiveFeatures]` 日志字段：

- `frame`, `dynamic_ratio`, `strength`, `target`, `orb`：输入帧号、动态并集比例、应用强度、目标数和最终 ORB 数。
- `static_before`, `static_after`：所有层/网格静态候选之和，前者包含原有空网格回退，后者包含新增补检去重后的候选；均在八叉树分配前计数。
- `compensated_cells`：新增低阈值补检的网格数，不包括旧回退。
- `inliers`, `state`：该帧 Tracking 结果；未执行有效内点计数时内点记为 -1。
- `compensation_ms`：配额/面积计算与新增 FAST、过滤、合并的直接耗时。
- `extraction_ms`：完整 ORB 提取耗时（含金字塔及描述子）。

`compensation_ms` 不是相对关闭功能运行的总耗时差：更多点导致的八叉树、描述子、匹配和优化开销需通过同一序列的开关对照统计评估。Stats 不开启时不输出日志。

## 编译

```bash
cmake --build build --target rgbd_tum -j2
```

本次开发使用的临时测试源码、运行脚本及测试产物已清理，以下保留已完成的验证记录。

## 本次验证记录

- 构建：`cmake --build build --target rgbd_tum -j2` 成功。
- 提取器回归：以修改前提交 `c8218dc` 为基线，关闭新增功能时，关键点位置、层级、方向、响应和描述子完全一致；补检合成场景静态候选由 8042 增至 16524。
- Tracking/Frame 测试：反馈上升、滞回、恢复衰减、无效历史、初始化、丢失及重置通过；使用真实 RGB-D Frame 验证特征、描述子、深度和虚拟右目索引对应。
- 真实数据：本机 `rgbd_dataset_freiburg3_walking_xyz` 前 180 帧，四个开关均开启，所有帧最终状态为 OK，正常退出。使用临时无界面驱动，不覆盖原有轨迹和点云。该测试不是完整序列的精度基准，也未验证 IMU-RGB-D 实际序列。
- 无界面运行暴露了已有的 `mptViewer` 未初始化问题：关闭补偿也在相同 Shutdown 位置崩溃。`src/System.cc` 补充了 `nullptr` 初始化，修复后复验正常退出；未改变后端算法。

该次真实数据运行统计（内点范围排除初始化时的 -1）：

```text
dynamic_ratio: min=0.000, max=0.345, mean=0.176
orb: min=1000.000, max=1345.000, mean=1175.744
inliers: min=130.000, max=424.000, mean=247.352
compensation_ms: min=0.579, max=4.694, mean=1.941
extraction_ms: min=9.250, max=15.605, mean=11.358
```

计时单位为毫秒。直接补偿耗时不包含更多特征带来的后续匹配/优化开销；结果随硬件与线程调度变化。
