# 运行环境

## 工作目录
- 所有命令在项目根目录 `/root/manipulator/` 下执行，不要 cd 到子目录
- 训练脚本：`code/.venv/bin/python -u code/train.py [参数]`
- 测试脚本：`code/.venv/bin/python -u code/test.py [参数]`

## 输出路径
- checkpoint 保存在根目录：`checkpoints/{run_name}/`
- 日志文件保存在根目录：`log/{run_name}.log`
- 训练日志输出重定向：`> /root/manipulator/log/{run_name}.log 2>&1 &`

## Python环境
- 使用 `code/.venv/bin/python` 虚拟环境

# 碰撞检测与 d_obs 说明

## 两种碰撞检测机制

系统使用两套独立的碰撞检测，理解它们的区别至关重要：

### 1. MuJoCo 物理碰撞（训练中使用的"真实"碰撞）
- 文件：`code/utils/collision.py` — `CollisionDetector` 类
- 使用 MuJoCo 原生碰撞引擎检测接触
- `detect_collisions()` 返回 `n_contacts`（接触点数量）
- `compute_collision_penalty()` 返回基于穿透深度的惩罚
- **这是"真实"碰撞**，只有当 MuJoCo 几何体实际接触时才触发
- 用在 `reward_fn.compute()` 中计算 r_collision
- 用在 `env.step()` 中判定 `collision` flag（line 348-349）

### 2. 胶囊体 SDF 距离（d_obs，保守近似）
- 文件：`code/utils/sdf.py` — `ObstacleSDF` 类
- 用胶囊体（capsule）近似机械臂连杆，计算到障碍物的有符号距离
- `capsule_to_sphere_distance()` 返回 signed distance（负值=穿透胶囊体）
- **d_obs < 0 不等于 MuJoCo 碰撞！** 胶囊体比 MuJoCo 碰撞模型大，是保守估计
- 用在 sigma 门控（`env.step()` line 253-255）：`sigma = smoothstep(d_obs)`，决定 RL 策略何时接管
- 用在奖励函数 `r_obs`：当 `d_obs < d_safe` 时给密集惩罚

### 总结对比

| 指标 | d_obs / 胶囊体 SDF | MuJoCo 碰撞检测 |
|------|-------------------|-----------------|
| 精度 | 保守（偏大） | 精确 |
| d<0 含义 | 穿透胶囊体近似 | 实际物理接触 |
| 用途 | sigma门控、r_obs奖励、验证集碰撞判定 | 训练碰撞终止、r_collision |
| 响应距离 | 更早触发（胶囊体更大） | 更晚触发（实际接触才触发） |

## 验证逻辑

在 `code/utils/validation.py` 中：
- **success** = `final_distance < 0.05 and not ep_ever_collided_mj`（到达目标且无MuJoCo碰撞）
- **collision** = `ep_ever_collided_mj`（MuJoCo接触数量 > 0）

在 `code/env/manipulator_env.py` 中：
- **collision** = MuJoCo 接触数量 > 0（line 384-385）
- **success** = `path_complete and not self._ever_collided`（路径走完且从未MuJoCo碰撞）

训练和验证使用**统一的 success/collision 判据**（均为 MuJoCo 真实碰撞检测）。

## 关键含义

1. **d_obs < 0 不意味着任务失败**：胶囊体穿透 ≠ MuJoCo 碰撞，模型可以容忍少量胶囊穿透
2. **sigma 门控在 d_obs < d_safe=0.06 时开始激活**：在安全区外就开始给 RL 策略更多控制权
3. **d_obs 负责"预警"，MuJoCo 负责"判罚"**：这种分离是"弃车保帅"机制的核心

# 论文写作规范

## 整体结构要求
- 使用IEEEtran会议模板，双栏排版
- 章节顺序：引言 → 相关工作 → 方法 → 实验 → 结论
- 数学公式使用equation环境，关键变量需在首次出现时定义

## 方法章节结构
### 标准子节顺序
1. **问题定义**
   - 任务目标（列举优化目标）
   - 控制结构（分层控制架构）
   - 强化学习形式化（MDP五元组：状态、动作、奖励）

2. **机械臂动力学模型**
   - 运动学关系（正运动学、雅可比矩阵）
   - 动力学方程（拉格朗日方程）
   - 零空间投影（投影矩阵定义）

3. **具体方法模块**（每个子模块需包含）
   - 动机说明（解决什么问题）
   - 理论依据（为什么这样设计）
   - 数学形式化（损失函数/约束/算法）
   - 与baseline的对比（如：正则化 vs reward shaping）

## 实验章节结构
### 实验设置
1. **对比方法**（至少5个）
   - 传统方法：2个（如RRT*、CHOMP）
   - 纯RL方法：2个（如PPO-Baseline、SAC-Baseline）
   - 消融实验：3个变体（单模块1、单模块2、完整方法）

2. **测试场景**（3类）
   - 场景1：稀疏障碍（3-5个静态障碍物）
   - 场景2：密集障碍/窄通道（10+障碍物）
   - 场景3：动态障碍（2-3个运动障碍物）

3. **评价指标**（5个）
   - 任务成功率
   - 样本效率（训练收敛曲线）
   - 路径质量（路径长度、执行时间）
   - 物理可行性（关节力矩平滑度）
   - 安全性（最小障碍距离）

### 结果分析
- 成功率对比
- 训练效率分析
- 物理可行性验证
- 消融实验结果

## 写作风格
- 每个方法模块需有明确的理论动机，避免纯描述性文字
- 数学符号首次出现需定义维度（如$M(q) \in \mathbb{R}^{7 \times 7}$）
- 对比baseline时需说明理论优势（如保持最优性、训练稳定性）
- 使用itemize环境列举要点，提高可读性

## 引用规范
- 使用\cite{}命令引用文献
- 理论依据需引用经典文献（如reward shaping引用Ng 1999）