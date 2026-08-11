# 一期论文实验与课程训练协议

本文档是一期实验的正式操作说明。场景定义、训练顺序、checkpoint选择和
最终评测均以这里列出的接口为准。

## 1. 正式数据

唯一正式场景目录为：

```text
results/ewalker_scenes/
├── manifest.json
├── free_space/test.json
├── whole_body/{train,validation,test}.json
├── confined_space/{train,validation,test}.json
├── curriculum/{train,validation}.json
└── generalization/{train,validation,test}.json
```

旧的 `results/phase1_splits/` 和 `results/solvable_scenes/` 已删除，禁止在
训练命令、评测命令或论文结果中引用。

场景规模如下：

| 实验 | 训练 | 验证 | 测试 | 几何协议 |
|---|---:|---:|---:|---|
| 自由空间 | - | - | 100 | 无障碍8字轨迹 |
| 整臂避障 | 60 | 20 | 100 | 直线扫描，3个球形障碍 |
| 狭小空间 | 60 | 20 | 100 | 10球走廊，净宽0.44--0.50 m |
| 泛化 | 60 | 20 | 100 | 单球训练/验证，三球测试 |

共640个基础场景，场景指纹全局不重复。`curriculum` 是整臂避障和狭小
空间数据的合并视图，不计为新增场景。

## 2. 可解性标准

场景不能仅凭起点和终点IK可达而被接受。生成器使用任务路径IK图搜索
冗余构型分支，并检查：

- 关节限位；
- 最低可操作度；
- 胶囊体到球形障碍物的安全间距；
- 胶囊自碰撞；
- MuJoCo非相邻连杆自碰撞；
- 相邻IK层之间的插值碰撞；
- 插值末端位置相对规定任务轨迹的偏差。

Oracle路径只用于离线筛选和审计，不进入策略观测，也不作为控制器参考。
正式数据曾使用独立MuJoCo环境复查150,400个路径构型，障碍接触和自碰撞
计数均为0。

重新生成场景：

```bash
cd /root/manipulator/code
screen -dmS paper_scene_generation bash -lc \
  '.venv/bin/python -u -m experiments.generate_paper_scenes \
   --output-dir results/ewalker_scenes --seed 20260726 \
   > /tmp/paper_scene_generation.log 2>&1'
```

查看进度：

```bash
screen -ls
tail -f /tmp/paper_scene_generation.log
```

## 3. 训练协议

共有两个相互独立的学习协议，不能把泛化训练接在主课程checkpoint之后。

### 3.1 主课程模型

主课程训练文件为：

```text
results/ewalker_scenes/curriculum/train.json       # 120个
results/ewalker_scenes/curriculum/validation.json  # 40个
```

120个训练场景由60个三球整臂场景和60个十球走廊场景组成。当前实现是
**混合自适应课程**，不是“先整臂、再走廊”的阶段式课程：

1. 训练开始时，120个场景等概率随机采样。
2. 每个场景至少完成一次后，记录场景成功率EMA。
3. 成功率越低，场景的后续采样权重越高。
4. 采样分布始终保留20%均匀分量，防止简单场景被遗忘。
5. 每25,000个全局环境步在40个固定验证场景上评测。
6. checkpoint按成功率、碰撞率、跟踪误差的字典序选优。

该模型的checkpoint用于实验1自由空间、实验2整臂避障和实验3狭小空间。

### 3.2 泛化模型

泛化模型独立从头训练：

```text
results/ewalker_scenes/generalization/train.json       # 60个单球场景
results/ewalker_scenes/generalization/validation.json  # 20个单球场景
results/ewalker_scenes/generalization/test.json        # 100个三球场景
```

训练和模型选择期间不得读取三球测试集。最终把单球checkpoint直接用于三球
测试，报告零样本迁移结果。该checkpoint只用于实验6。

## 4. 方法、种子与任务数量

学习方法为：

```text
SAC-Joint
SAC-Residual
Ours-No-Physics
Ours-Physics
Ours-Full
```

每种方法分别训练主课程模型和泛化模型，种子固定为：

```text
11, 23, 37, 53, 71
```

因此共有 `5种方法 x 2种协议 x 5个种子 = 50` 个学习任务。

评测还包括PD、Gradient Projection、CBF-QP和Adaptive Gradient-CBF，
共9种方法、4类场景和5个种子，即180个评测任务。每个任务使用100个固定
测试场景。Adaptive Gradient-CBF是确定性方法，重复种子仅用于保持评测清单
结构一致；相同场景上的轨迹应完全一致。

Adaptive Gradient-CBF使用统一入口按场景几何选择安全结构：整臂避障等稀疏
场景采用平滑零空间梯度、障碍CBF及24对非相邻胶囊的多约束自碰撞CBF，
梯度幅值为0.3、平滑系数为
0.8；10障碍走廊采用平滑零空间梯度，平滑系数为0.9，避免密集CBF约束之间
的反复投影。正式方法统一使用2 cm自碰撞阈值；自由空间没有障碍梯度，因此
该方法退化为多自碰撞约束CBF-QP。该分支只读取场景协议，不读取场景ID。

## 5. 生成任务清单

配置唯一来源为 `code/experiment_config.py`，可执行任务由
`code/experiments/manifest.py` 生成：

```bash
cd /root/manipulator/code
.venv/bin/python -m experiments.manifest \
  --checkpoint-root /root/autodl-tmp/manipulator/checkpoints/phase1 \
  --result-root ../results/phase1 \
  --output ../results/phase1/manifest.json
```

清单中的 `training_jobs` 已按以下循环顺序展开：

```text
方法
  -> curriculum协议
     -> seed 11, 23, 37, 53, 71
  -> generalization协议
     -> seed 11, 23, 37, 53, 71
```

这个列表顺序只是任务调度顺序，不代表主课程内部的场景顺序。不同方法和
不同种子彼此独立，可以在显存允许时并行运行。

正式训练默认使用32个并行环境和每轮8次梯度更新，保持
`UTD = 8 / 32 = 0.25`。Replay Buffer按整个环境批次写入，减少主进程
逐条写入的Python开销。

## 6. 云服务器运行规范

项目目录：

```text
/root/manipulator
```

checkpoint统一保存到临时大容量磁盘：

```text
/root/autodl-tmp/manipulator/checkpoints/phase1
```

每个训练任务使用独立的命名`screen`会话。例如：

```bash
cd /root/manipulator/code
screen -dmS ours_full_curriculum_s11 bash -lc \
  '<manifest.json中对应的training_jobs命令> \
   > /root/autodl-tmp/ours_full_curriculum_s11.log 2>&1'
```

常用监控命令：

```bash
screen -ls
screen -r ours_full_curriculum_s11
tail -f /root/autodl-tmp/ours_full_curriculum_s11.log
nvidia-smi
```

训练恢复必须同时具有checkpoint和对应replay文件。新协议不能默认续训旧
协议checkpoint；仅做迁移时应加载actor，而不是伪装成完整resume。

## 7. 最终评测与论文表格

所有学习任务完成后，执行清单中的 `evaluation_jobs`。评测器通过
`--scene-json`读取固定测试集，输出统一JSONL指标，包括成功、碰撞、完成
时间、跟踪RMS/峰值、最小间距、平滑度、力矩变化率、能耗、零空间利用度、
参考修正和门控触发率。

汇总结果并生成论文LaTeX表格：

```bash
cd /root/manipulator/code
.venv/bin/python -m experiments.report ../results/phase1/*/*/*.jsonl \
  --output-dir ../paper/generated
```

论文只引用评测程序生成的结果。缺失实验不得填入预测值，也不得把oracle
路径性能作为策略性能报告。

## 8. 推荐执行顺序

在只有一张GPU时，推荐按以下工程顺序执行：

1. 对每种方法先运行`curriculum/seed_11`做完整链路检查。
2. 确认验证、best checkpoint、final checkpoint和replay均正常保存。
3. 完成该方法其余4个主课程种子。
4. 完成该方法5个泛化种子。
5. 依次处理其余学习方法。
6. 运行无需训练的PD、Gradient Projection、CBF-QP和
   Adaptive Gradient-CBF评测。
7. 运行所有学习方法的固定测试集评测。
8. 汇总统计、生成置信区间、曲线和LaTeX表格。

当前RTX 4080 SUPER服务器使用16核CPU配额，实测适合在同一GPU上并行2个
独立种子，以提高整套实验吞吐。各任务应错开启动，尽量避免串行验证同时
发生。每项任务仍使用相同的32环境、8梯度步和场景协议；若GPU显存、
CPU负载或单任务步速明显恶化，应降低并发数，而不能单独修改某个方法的
训练参数。
