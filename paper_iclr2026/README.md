# ICLR 2026 论文项目

## 文件结构
- `main.tex`: 主论文文件
- `references.bib`: 参考文献
- `iclr2026_conference.sty`: ICLR 2026 会议样式文件
- `Makefile`: 编译脚本

## 编译方法

### 使用 Makefile
```bash
make          # 编译生成 PDF
make clean    # 清理临时文件
make view     # 打开 PDF
```

### 手动编译
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## 论文结构

按照 CLAUDE.md 规范，论文包含以下章节：

1. **Introduction**: 研究背景和贡献
2. **Related Work**: 相关工作综述
3. **Method**: 方法详述
   - 问题定义（任务目标、控制结构、RL形式化）
   - 机械臂动力学模型
   - 零空间优化模块
4. **Experiments**: 实验设置和结果分析
5. **Conclusion**: 总结

## 下一步工作

- 补充实验数据和图表
- 完善相关工作引用
- 添加算法伪代码
- 补充消融实验细节
