# E-Walker 项目现状、缺口与训练计划

> 更新日期：2026-07-28  
> 当前稳定基线提交：`6bdb785`（训练稳定化修改待提交）  
> 适用模型：`E-Walker-inspired 7-DoF` 研究重建  
> 重要边界：本项目不声称该模型是 E-Walker 官方模型、在轨型号或飞行硬件。

## 1. 当前结论

项目已经从 Franka Panda 默认模型切换到 E-Walker-inspired 七自由度空间机械臂，
完成了 URDF、MuJoCo、Pinocchio、解析胶囊碰撞、环境、训练入口和第一版正式场景
的贯通。当前可以开展基于球形碰撞代理的全臂避障强化学习实验。

目前尚未完成高保真空间装配任务环境。现有障碍物仍是球形代理，不应在论文中
描述成真实桁架、面板、货架、对接环或航天器 CAD。

## 2. 模型来源和建模边界

### 2.1 公开文献直接支持的内容

依据 Nair 等人的 E-Walker 公开论文：

- 对称七转动关节结构；
- 三轴肩、单轴肘、三轴腕；
- 双端操作概念；
- 两根管状主连杆；
- 约 1.3 m 的 `1:6` 地面原型总体尺度；
- 地面原型执行器指标约 42 Nm。

### 2.2 本项目自行建模的内容

公开论文没有提供官方 URDF、完整 CAD、精确关节坐标系、逐连杆质量分布、
完整惯量张量和关节限位。因此以下内容均为研究建模值：

- 精确段长分配和胶囊半径；
- 七个关节轴和关节限位；
- `0.5 rad/s` 速度限制；
- 各连杆质量和惯量；
- 固定基座、关闭重力的空间环境近似；
- 紧凑肩部和腕部组件的允许几何嵌套对。

模型总建模质量为 10.6 kg。该值与文献中约 12 kg 的原型级描述分开记录，
不能写成实测质量。

详细来源和参数见：

```text
ewalker_description/README.md
ewalker_description/model_metadata.json
ewalker_description/urdf/ewalker.urdf
models/ewalker_scene.xml
```

## 3. 已实现功能

### 3.1 机器人模型

- 已实现 E-Walker-inspired 7-DoF URDF。
- 已实现对应 MuJoCo MJCF。
- Pinocchio 与 MuJoCo 均解析为 7 个自由度。
- 零位 TCP 坐标在两套模型中均为 `[0, 0, 1.30] m`。
- URDF 关节限位、速度限制和力矩限制已由代码动态读取。
- 训练、测试和正式场景生成入口默认使用新模型。

### 3.2 碰撞模型

- 解析规划器和 MuJoCo 使用 9 个对应胶囊体。
- 已在 20 组随机关节构型下对照胶囊端点、半径和姿态。
- 已区分正常相邻连杆接触、紧凑关节壳体嵌套和真实自碰撞。
- MuJoCo 接触作为终止判据，解析胶囊距离作为提前安全信号。
- 已修复可视化层把所有机器人胶囊错误绘制成球体的问题。

### 3.3 控制和训练

- 结构化动作：3 维任务空间参考修正 + 4 维零空间系数。
- 直接 7 维关节动作 SAC。
- PD 上叠加 7 维残差 SAC。
- 动力学力矩正则化。
- 独立安全 critic 和 Lagrange 约束。
- 风险门控。
- CBF 后处理安全滤波器。
- 并行环境、批量经验回放、定期验证和 best checkpoint。
- 关节速度裁剪已恢复，不能再绕过 URDF 速度限制。
- 力矩正则项已读取 E-Walker 的 42 Nm 建模上限。

### 3.4 正式场景数据

正式目录：

```text
results/ewalker_scenes/
```

数据规模：

| 场景 | 训练 | 验证 | 测试 | 障碍物 |
|---|---:|---:|---:|---|
| 自由空间 | - | - | 100 | 无障碍八字轨迹 |
| 全臂避障 | 60 | 20 | 100 | 3 个球形代理 |
| 狭窄通道 | 60 | 20 | 100 | 10 个球形代理 |
| 泛化 | 60 | 20 | 100 | 单球训练/验证，三球测试 |

基础场景共 640 个，指纹全局唯一。`curriculum` 是全臂避障与狭窄通道的
合并视图：

```text
results/ewalker_scenes/curriculum/train.json       # 120
results/ewalker_scenes/curriculum/validation.json  # 40
```

已完成的字段审计：

- 540 个障碍场景全部带规划器认证的 `feasible_q_path`；
- 规划路径记录的最小安全余量为 `0.025009758 m`；
- `pd_success=true` 的场景数为 0；
- 540 个障碍场景的零残差 PD 均因碰撞失败；
- 训练、验证、测试指纹无交集。

注意：以上是生成记录和字段级独立审计。尚未对新数据集的每个 oracle 插值
构型再执行一次完整的独立 MuJoCo 全路径复核。

### 3.5 论文和测试

- 论文已加入 E-Walker-inspired 来源、惯量建模值和非飞行型号声明。
- 新增 `code/test_ewalker_model.py`。
- 当前回归结果为 21 项测试通过。
- 本机没有 `latexmk`，尚未重新编译论文 PDF。

## 4. 最近训练诊断与稳定化协议

服务器：

```text
root@connect.nmb1.seetacloud.com:30068
项目目录：/root/manipulator
同步前 Git HEAD：6bdb785
GPU：RTX 4080 SUPER
```

旧试跑配置：

```text
模型：E-Walker-inspired 7-DoF
场景协议：curriculum
seed：11
总步数：500000
并行环境：32
每轮梯度更新：8
batch size：256
agent_type：structured
lambda_dyn：默认 1
安全 critic：开启
风险门控：开启
```

该参数组合在算法意义上等价于 `Ours-Full`，但属于 protocol v3 链路试跑，
已在约 382k 步停止。旧目录和最佳模型保留：

```text
structured/curriculum/seed_11
```

```text
/root/autodl-tmp/manipulator/checkpoints/ewalker_phase1/structured/curriculum/seed_11
```

固定验证集上的关键结果：

| 环境步 | 成功率 | 碰撞率 |
|---:|---:|---:|
| 275k（best） | 65.0% | 35.0% |
| 300k | 20.0% | 80.0% |
| 350k | 7.5% | 92.5% |
| 375k | 10.0% | 90.0% |

这不是验证噪声，而是训练后期失稳。停止时 Lagrange 乘子已从最佳点附近的
约 52.9 增长到约 94，并逼近旧硬上限 100；困难场景自适应采样同时削弱了
已学会场景的覆盖；按穿透深度计算的碰撞奖励又会低估浅接触终止。

protocol v4 作如下修正：

- Lagrange 学习率由 `1e-3` 降至 `1e-4`，最大值由 100 限制为 10；
- 场景采样的均匀混合比例由 20% 提高到 50%；
- 单场景采样概率不得超过均匀概率的 3 倍；
- MuJoCo 判定真实碰撞时，额外加入固定 `-500` 终止惩罚；
- protocol v3 checkpoint 禁止直接续训，只可单独评估或显式迁移 actor。

下一次试跑使用全新正式目录
`ours_full/curriculum/seed_11`，不复用 v3 的 critic、优化器或经验池。先以
单 seed 验证 300k 步后不再出现灾难性回退，再扩展五 seed 和其他方法。

### 4.1 protocol v4 完整试跑结论

`ours_full/curriculum/seed_11` 已完成 500k 步。最佳固定验证结果出现在
450543 步，总成功率 65%，但分层结果为：

```text
whole_body：7/20 = 35%
confined_space：19/20 = 95%
```

v4 已解决 Lagrange 饱和问题（最终乘子约 0.304），但暴露出状态表示的根本
缺口：策略只有胶囊到障碍的标量距离，没有避障方向；训练也没有启用障碍布局
或未来路径观测。总成功率被简单的 confined-space 子集显著抬高。

### 4.2 protocol v5（当前代码，尚未训练）

- 每个机械臂胶囊观测最近障碍的距离和三维远离方向；
- 默认编码最多 10 个按表面距离排序的障碍，每项带相对位置、半径和 mask；
- 默认加入未来 10、25、50 步参考路径点；
- 状态维度由旧实验的 55 变为 E-Walker 当前模型下的 141；
- 验证分别记录 whole-body/confined-space、自碰撞和障碍物碰撞；
- 整体奖励默认除以 10，降低终止碰撞奖励造成的 TD 尖峰；
- 熵温度下限由 0.02 提高到 0.05；
- safety critic 使用 Huber 损失并记录 `predicted_cost`；
- 动力学正则从力矩利用率 80% 开始施加归一化软惩罚；
- 恢复动作与零空间动作惩罚；
- manifest 增加不计入正式矩阵的 `whole_body_diagnostic` seed 11 诊断任务。

protocol v5 改变了观测维度和奖励尺度，与 v4 checkpoint 不兼容，必须从头
训练。下一步应先运行 whole-body 诊断任务，达到稳定改进后再重跑 curriculum；
不得直接启动五 seed 正式矩阵。

## 5. 尚未实现或尚未验证

### 5.1 机器人高保真建模

- 没有官方 E-Walker CAD 或飞行硬件参数。
- 没有详细的双端 LEE 机构、抓取、锁紧和对接接触模型。
- 没有自由漂浮基座和航天器—机械臂耦合动力学。
- 没有基座姿态控制器、动量管理和反作用轮/推进器模型。
- 没有柔性连杆、结构振动、关节回差、摩擦和传感器噪声标定。
- 没有质量、惯量和执行器参数的不确定性随机化。

### 5.2 任务场景

- 现有障碍物是球形代理，不是真实任务物体。
- 尚未实现桁架杆件的长胶囊/圆柱表示。
- 尚未实现航天器面板、舱壁和设备箱的盒体表示。
- 尚未实现装配接口、对接环、抓取点和容差约束。
- 尚未实现任务阶段：接近、抓取、搬运、插入、锁紧。
- 尚未实现移动障碍物和动态空间环境。
- 尚未将非球形几何统一接入 SDF、规划器、观测和场景 JSON schema。

### 5.3 场景验证

- 新 E-Walker 场景尚未执行独立的全量 MuJoCo oracle 路径复核。
- 尚未统计规划器复核与运行时 MuJoCo 接触检测的误差分布。
- 尚未加入可操作度、关节限位余量和力矩余量的场景分层统计。
- 当前所有障碍场景都专门构造为 PD 碰撞失败，尚缺少简单、中等、困难的
  平衡课程；训练分布可能过度偏向近碰撞阻挡器。
- 尚未检查 blocker 构造是否造成单一可利用模式或数据偏差。

### 5.4 正式训练与评测

- 50 个学习任务尚未完成。
- 当前只有一个 `Ours-Full` 等价配置的 seed 11 链路试跑。
- SAC-Joint、SAC-Residual、Ours-No-Physics、Ours-Physics 尚未在新模型上训练。
- 五个正式随机种子尚未跑齐。
- 独立泛化模型尚未训练。
- PD、Gradient Projection、CBF-QP 尚未在新固定测试集上统一评测。
- 尚未生成最终 JSONL、置信区间、消融表格和 LaTeX 结果。
- 尚未执行跨 seed 稳定性和失败案例分类。

### 5.5 工程和文档债务

- `code/experiments/manifest.py`、审计器、生成器和实验 README 已切换到
  `results/ewalker_scenes/`；后续不得重新引用旧 `paper_scenes`。
- 训练场景生成器会在一个 split 全部完成后才写文件，长任务中断会丢失该
  split 的内存结果；需要增加增量落盘和恢复机制。
- GitHub HTTPS remote 缺少凭据。本地和训练服务器 Git 已同步，但 GitHub
  尚未推送最新提交。
- 旧 Panda `paper_scenes`、未完成 `paper_scenes_v2`、E-Walker 冒烟集和无引用
  的旧场景图片已经清理；当前正式场景唯一来源为 `ewalker_scenes`。
- `challenge_stage1.json` 因胶囊调试脚本仍在使用而保留，
  `challenge_stage3.json` 因用户命令文件仍在使用而保留；它们不是正式训练集。

## 6. 正式训练矩阵

固定随机种子：

```text
11, 23, 37, 53, 71
```

学习方法：

| 方法 | 关键参数 |
|---|---|
| SAC-Joint | `--agent_type joint --lambda_dyn 0 --no_safety_critic` |
| SAC-Residual | `--agent_type residual --lambda_dyn 0 --no_safety_critic` |
| Ours-No-Physics | `--agent_type structured --lambda_dyn 0 --no_safety_critic --disable_gate` |
| Ours-Physics | `--agent_type structured --lambda_dyn 1 --no_safety_critic --disable_gate` |
| Ours-Full | `--agent_type structured --lambda_dyn 1` |

每种方法分别训练：

1. `curriculum`：用于自由空间、全臂避障和狭窄通道测试；
2. `generalization`：仅使用单球训练/验证，用于三球零样本泛化测试。

总任务数：

```text
5 方法 × 2 协议 × 5 seed = 50 个学习任务
```

无需学习的基线：

```text
PD
Gradient Projection
CBF-QP
```

最终评测矩阵：

```text
8 方法 × 4 场景 × 5 seed = 160 个评测任务
每个任务 100 个固定测试场景
```

## 7. 推荐执行安排

### 阶段 A：先完成当前链路试跑

1. 等待 `ewalker_structured_s11` 跑到 500k。
2. 检查 25k 间隔验证是否完整。
3. 检查 `ckpt_best.pt`、final checkpoint 和 replay 是否齐全。
4. 复核成功率、碰撞率、跟踪误差和安全代价是否存在异常。
5. 用 100 个固定 whole-body 和 confined-space 场景做一次试评测。
6. 决定复用该权重迁移，还是按正式 `ours_full` 名称重跑。

### 阶段 B：修复正式任务清单

1. 将 `manifest.py` 和实验 README 全部切换到 `ewalker_scenes`。
2. 将 checkpoint 根目录统一为：

   ```text
   /root/autodl-tmp/manipulator/checkpoints/ewalker_phase1
   ```

3. 重新生成正式 manifest。
4. 审查 50 个命令的模型路径、场景路径、方法参数和 seed。
5. 先执行每种方法的 `curriculum/seed_11`，完成方法级冒烟测试。

### 阶段 C：主课程训练

建议顺序：

1. Ours-Full：5 seeds；
2. Ours-Physics：5 seeds；
3. Ours-No-Physics：5 seeds；
4. SAC-Residual：5 seeds；
5. SAC-Joint：5 seeds。

先完成结构化方法有利于尽早发现论文核心方法的问题。服务器只有一张
RTX 4080 SUPER，建议最多并行两个训练任务并错开验证时间。每个任务保持：

```text
n_envs=32
grad_steps=8
UTD=0.25
val_every_steps=25000
checkpoint_every_steps=50000
steps=500000
```

不得为了加速而只修改某个方法的训练预算。

### 阶段 D：独立泛化训练

对五种方法分别从头训练五个 seed。禁止从 curriculum checkpoint 续训，
也禁止在训练或选模期间读取三球测试集。

### 阶段 E：经典方法和固定测试集评测

1. 先运行 PD、Gradient Projection、CBF-QP；
2. 再评测五种学习方法；
3. 所有方法使用同一测试场景、终止标准和指标计算；
4. 保存原始 JSONL，不只保存汇总均值；
5. 对碰撞失败、超时失败、奇异位形失败分别统计。

### 阶段 F：汇总和论文

1. 对五个 seed 计算均值和 95% 置信区间；
2. 生成成功率、碰撞率、跟踪误差、最小间距、完成时间、平滑度、能耗、
   力矩和门控使用率；
3. 生成消融表、训练曲线和典型失败案例图；
4. 更新论文实验设置中的 E-Walker 参数和场景代理说明；
5. 在有 LaTeX 环境的机器上重新编译全文；
6. 不得用预测值、oracle 性能或当前单 seed 试跑填充最终表格。

## 8. 在高保真任务场景上的后续路线

当前球形代理实验完成后，建议新增第二阶段任务环境，而不是直接替换一期数据：

1. 定义可版本化的场景 schema，支持 sphere、capsule、cylinder、box；
2. 用长胶囊表示桁架，用 box 表示面板和舱壁；
3. 为抓取点和装配接口增加目标姿态及容差；
4. 使用规划器做低成本全路径筛选；
5. 对筛选边界场景做 MuJoCo 接触复核；
6. 加入固定基座与自由漂浮基座两个实验层级；
7. 将新环境作为高保真迁移/泛化实验，不与一期球形代理结果混为一谈。

## 9. 常用命令

查看当前训练：

```bash
ssh -p 30068 root@connect.nmb1.seetacloud.com
screen -ls
tail -f /root/autodl-tmp/manipulator/logs/ewalker_structured_curriculum_s11.log
```

本地运行 E-Walker 可视化：

```bash
cd /home/merlin/manipulator
PYTHONPATH=code code/.venv/bin/python code/test.py \
  --method kp --steps 500 --n_obstacles 3 --render
```

使用正式 curriculum 启动训练时必须显式指定：

```bash
--scene_json results/ewalker_scenes/curriculum/train.json \
--val_json results/ewalker_scenes/curriculum/validation.json
```

## 10. 完成判据

项目的一期实验只有同时满足以下条件才能视为完成：

- 50 个学习任务全部完成且 checkpoint 可读取；
- 160 个固定评测任务全部产生原始结果；
- 不同 split 和方法没有场景或 checkpoint 串用；
- 新 E-Walker 场景完成独立碰撞复核；
- 五 seed 统计和置信区间完整；
- 论文明确模型、惯量和球形代理边界；
- 所有表格均可由脚本从原始 JSONL 重建；
- GitHub、开发机和训练服务器代码提交一致。
