# E-Walker 一期实验阶段结果（2026-08-11）

本文记录当前固定测试集上的阶段性结果。除 Ours-Full 的三种子结果外，下表
均为 seed 11 的 100 场景确定性评测，不应直接当作论文最终五种子统计。

## 当前最佳确定性控制器

正式入口为 `adaptive_gradient_cbf`：稀疏障碍场景使用平滑零空间距离梯度和
障碍/自碰撞双重 CBF；密集走廊使用平滑距离梯度；正式方法统一使用2 cm
自碰撞阈值。评测参数固定在 `code/experiments/runner.py`。

| 场景 | 方法 | 成功 | 碰撞 | 跟踪 RMS (cm) | 平滑度 | 力矩变化率 |
|---|---|---:|---:|---:|---:|---:|
| Whole-body | Adaptive Gradient-CBF | 92% | 8% | 1.24 | 0.943 | 54.59 |
| Whole-body | CBF-QP | 88% | 10% | 1.83 | 0.986 | 79.86 |
| Confined | Adaptive Gradient-CBF | 100% | 0% | 0.013 | 0.260 | 5.59 |
| Confined | Gradient Projection | 100% | 0% | 0.012 | 2.418 | 134.13 |
| Generalization | Adaptive Gradient-CBF | 94% | 6% | 1.22 | 0.840 | 50.84 |
| Generalization | CBF-QP | 92% | 6% | 1.96 | 0.980 | 82.10 |
| Free-space | Adaptive Gradient-CBF | 96% | 4% | 0.64 | 1.549 | 10.30 |
| Free-space | CBF-QP | 96% | 4% | 0.64 | 1.549 | 10.30 |

当前方法在四类场景的平均成功率为 95.5%，对应各场景最强经典安全基线的
平均成功率为 94.0%（Whole-body 取 CBF-QP，Confined 取 Gradient
Projection，Generalization 和 Free-space 取 CBF-QP）。优势主要体现在安全
与综合控制品质，并非每个次级指标都占优：自由空间的跟踪、平滑度和力矩变化
率仍弱于 CBF-QP，Whole-body 的能耗也高于 CBF-QP。

曾冻结2.5 cm候选参数后，另用 seed 20260811 生成100个未参与调参的自由空间初始
构型做了盲测。Adaptive Gradient-CBF和CBF-QP均为100%成功、0碰撞；前者
能耗2.883，后者2.897，但前者跟踪RMS（0.840 cm对0.679 cm）、平滑度
（1.592对1.523）和力矩变化率（11.99对9.99）略差。因此原固定测试集上的
1个百分点安全优势不可视为已复现；自由空间应报告安全持平，而不是显著优于。

## 学习方法现状

采用双重 CBF 复评的 Ours-Full 三个独立种子在 Whole-body 上为：seed 11
85%/12%、seed 23 78%/17%、seed 37 85%/11%（成功/碰撞），平均成功率
82.7% ± 4.0%。它尚未超过确定性控制器，因此当前不能把强化学习训练称为
论文主结果。

## 结果边界与下一步

1. 2.5 cm 自碰撞裕量是在原 free-space 测试集上做的探索性消融；新增盲测已
   表明安全性与CBF-QP持平且次级指标更差，因此正式默认值已回退到公平的2 cm。
2. free-space 的原始提升只有一个场景且未在盲测复现；下一步应把所有
   非相邻胶囊对改成多约束 CBF，避免“最小距离对切换”造成梯度不连续。
3. 统一跑完 5 个训练种子和 9 方法 × 4 场景矩阵，报告 bootstrap 95% 置信区间
   与配对成功率检验，不重复计算确定性方法的相同轨迹作为独立样本。
4. 对 capsule 距离和 MuJoCo contact 做逐对校准，并分类剩余的自碰撞、障碍
   碰撞和超时失败。
5. 一期球形代理完成后，再增加 box/capsule/cylinder 的任务几何、目标姿态与
   抓取/装配阶段；先由规划器筛选，再只对边界案例做 MuJoCo 全路径复核。

服务器结果目录：

```text
/root/autodl-tmp/manipulator/results/v6_remaining_baselines/
/root/autodl-tmp/manipulator/results/v7_self_margin/
```

其中 `free_space_blind_20260811.json`、`blind_cbf.jsonl` 和
`blind_adaptive.jsonl` 是参数冻结后的附加盲测集及结果。

对应日志：

```text
/root/autodl-tmp/manipulator/logs/v6_*.log
/root/autodl-tmp/manipulator/logs/v7_free_margin*.log
```
