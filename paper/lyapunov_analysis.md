# 有界RL干扰下的李雅普诺夫稳定性分析

## 1. 控制律与误差动力学

### 1.1 控制结构回顾

$$
\dot{q} = J^\dagger(q)\Big(\dot{x}_d + K_p(x_d - x) + \text{diag}(\sigma)\Delta \dot{x}_{\text{RL}}\Big) + N(q)\dot{q}_0
$$

其中 $\Delta \dot{x}_{\text{RL}} \in \mathbb{R}^6$ 为策略网络输出的任务空间扰动，$\dot{q}_0 \in \mathbb{R}^7$ 为零空间自运动速度，$\text{diag}(\sigma)$ 为门控算子，$N(q) = I - J^\dagger J$ 为零空间投影矩阵。

### 1.2 末端误差动力学的导出

两端左乘 $J(q)$，利用 $JJ^\dagger = I$ 和 $JN = 0$：

$$
J\dot{q} = \dot{x}_d + K_p e + \sigma \cdot \Delta\dot{x}_{\text{RL}}
$$

其中 $e = x_d - x \in \mathbb{R}^6$ 为末端跟踪误差。

由于 $\dot{x} = J\dot{q}$，得到：

$$
\dot{e} = \dot{x}_d - \dot{x} = -K_p e - \sigma \cdot \Delta\dot{x}_{\text{RL}}
$$

即：

$$
\boxed{\dot{e} = -K_p e + d(t)}
$$

其中 $d(t) = -\sigma \cdot \Delta\dot{x}_{\text{RL}} \in \mathbb{R}^6$ 为RL策略引入的干扰项。

这是**受扰线性系统**的标准形式。

---

## 2. 输入-状态稳定性（ISS）分析

### 2.1 Lyapunov候选函数

取二次型：

$$
V(e) = \frac{1}{2} e^T e = \frac{1}{2}\|e\|^2
$$

满足 $V(0)=0$，$V(e)>0\ \forall e \neq 0$，径向无界。

### 2.2 沿轨迹求导

$$
\begin{aligned}
\dot{V}(e) &= e^T \dot{e} = e^T(-K_p e + d) \\
&= -e^T K_p e + e^T d
\end{aligned}
$$

设 $K_p = k_p I$，$k_p > 0$（对角增益矩阵），则：

$$
\dot{V} = -k_p\|e\|^2 + e^T d
$$

应用 Cauchy-Schwarz 不等式 $|e^T d| \leq \|e\|\|d\|$：

$$
\dot{V} \leq -k_p\|e\|^2 + \|e\|\|d\|
$$

### 2.3 干扰有界假设

设RL干扰有界：

$$
\|d(t)\| \leq \delta, \quad \forall t \geq 0
$$

则：

$$
\dot{V} \leq -k_p\|e\|^2 + \delta\|e\| = -\|e\|(k_p\|e\| - \delta)
$$

### 2.4 一致最终有界性（UUB）

当 $\|e\| > \delta/k_p$ 时，$\dot{V} < 0$。因此跟踪误差是**一致最终有界**（Uniformly Ultimately Bounded, UUB）的，稳态界为：

$$
\boxed{\limsup_{t\to\infty} \|e(t)\| \leq \frac{\delta}{k_p}}
$$

**定理1：** 在控制律 (1) 作用下，若RL扰动 $\|d(t)\| \leq \delta$，则末端跟踪误差 $e(t)$ 收敛到紧集 $\mathcal{B} = \{\, e \in \mathbb{R}^6 \mid \|e\| \leq \delta/k_p \,\}$。

---

## 3. 干扰界 $\delta$ 的物理解读

将 $\delta$ 用论文中的设计变量表示：

$$
d = -\sigma \cdot \Delta\dot{x}_{\text{RL}}, \quad \|d\| \leq \|\sigma\| \cdot \|\Delta\dot{x}_{\text{RL}}\|
$$

- $\|\Delta\dot{x}_{\text{RL}}\| \leq a_{\max}$：策略网络输出经tanh激活，天然有界
- $\|\sigma\| \in [0, 1]$：门控算子由障碍距离动态调控

因此：

$$
\delta = a_{\max}, \quad \limsup_{t\to\infty} \|e(t)\| \leq \frac{a_{\max}}{k_p}
$$

### 3.1 推论1：安全区域（$\sigma \to 0$）

当 $d_{\text{obs}} \gg d_{\text{critical}}$ 时，$\sigma \to 0$，$\delta \to 0$：

$$
\boxed{\limsup_{t\to\infty} \|e(t)\| \to 0}
$$

**含义：** 安全区域内恢复名义PD控制的指数稳定性，跟踪误差收敛到零。

### 3.2 推论2：危险区域（$\sigma \to 1$）

当 $d_{\text{obs}} \to 0$ 时，$\sigma \to 1$，$\delta \to a_{\max}$：

$$
\boxed{\limsup_{t\to\infty} \|e(t)\| \leq \frac{a_{\max}}{k_p}}
$$

**含义：** 策略主动牺牲有界跟踪精度以换取避障空间——这就是**主任务松弛机制**的理论效果。该界可通过调节 $k_p$ 和 $a_{\max}$ 显式控制。

---

## 4. 零空间稳定性分析

### 4.1 关键性质

零空间项不影响末端：

$$
J(q)N(q) = 0, \quad N(q)^2 = N(q)
$$

因此末端动力学 $J\dot{q}$ 中不出现 $\dot{q}_0$，**末端与零空间稳定性解耦**。

### 4.2 可操作度作为零空间Lyapunov候选

定义可操作度：

$$
w(q) = \sqrt{\det(J(q)J(q)^T)} \in \mathbb{R}_{\geq 0}
$$

取Lyapunov候选：

$$
V_n(q) = w_{\max} - w(q)
$$

$V_n \geq 0$，$V_n = 0$ 当且仅当 $w = w_{\max}$。

### 4.3 策略引导下的零空间稳定性

零空间梯度满足：

$$
\nabla_q w = w(q) \cdot \text{vec}\left( \frac{\partial J}{\partial q} J^\dagger \right)
$$

策略网络通过奖励函数 $r_{\text{manip}} = w(q_t)$ 学习零空间动作 $\dot{q}_0$，使得 $\dot{q}_0$ 在可操作度梯度方向上产生分量：

$$
\langle \dot{q}_0, \nabla_q w \rangle \geq 0
$$

即策略学到的行为使机械臂远离奇异位形。

---

## 5. 联合系统稳定性

由 $JN = 0$ 的严格解耦性质，联合Lyapunov函数为：

$$
V_{\text{total}}(e, q) = V_t(e) + V_n(q)
$$

满足：

- $\dot{V}_{\text{total}} < 0$ 当 $\|e\| > \delta/k_p$ 且 $w > w_{\min}$
- 末端和零空间互不干扰

**定理2：** 在控制律 (1) 和策略 $\pi(s)$ 训练收敛的条件下，联合系统 $(e, q)$ 在有界RL扰动下保持稳定，末端误差有界，零空间位形一致最终有界于远离奇异位形的紧集。

---

## 6. 注意事项与限制

| 限制 | 说明 | 处理方法 |
|------|------|---------|
| **非奇异假设** | ISS分析假设 $J(q)$ 行满秩 | 可操作度奖励 $r_{\text{manip}}$ 引导策略远离奇异位形 |
| **界 $\delta/k_p$ 的保守性** | 分析给出的是充分条件，实际误差通常小于理论界 | 将理论界作为安全余量，实际误差由实验验证 |
| **$\sigma$ 显式公式** | 需补充 $\sigma = f(d_{\text{obs}})$ 的具体定义 | 见审稿要点 |
| **联合系统的严格ISS** | 零空间动态依赖于策略是否收敛 | 可在训练后验证 $\dot{V}_n \leq 0$ |

---

## 7. 论文中的写作建议

推荐以"稳定性分析"子节（约1-2页）形式插入方法章节末尾（第\ref{sec:algorithm}节之后），包含：

1. **末端跟踪误差的ISS**（第1-3节）
2. **零空间稳定性**（第4节）  
3. **联合系统稳定性**（第5节）
4. 明确说明分析依赖的假设及合理性
