# 面向指数增强策略的量化算法优化设计与实现

本科毕业设计。以沪深 300 指数增强为应用场景，对其核心环节——带约束的组合优化算法——做优化设计与实现，并在真实 A 股数据上做严格回测。

指数增强的目标是在紧贴基准指数的同时获取稳定的超额收益。其关键计算步骤是“组合构建”：在已有个股 alpha 预测的前提下，求解满足跟踪误差、行业偏离、个股权重与换手率等约束的最优持仓权重。本项目聚焦这一步，重点解决它在 A 股落地时的两个问题：协方差估计病态，以及低信噪比下的预测不确定性。

- 开源仓库：<https://github.com/aokimi0/IndexEnhancementStrategy>

## 功能概览

项目包含两部分工作：

1. **约束型多因子组合优化算法**（应对协方差病态）
   - LightGBM 多因子模型输出截面 alpha；
   - Ledoit-Wolf 缩减估计 + 最小特征值修正，抑制协方差病态导致的“误差最大化”；
   - 构建带跟踪误差/行业/权重/换手约束的二次规划，OSQP → CLARABEL → SCS 三级求解器回退保证数值稳健。
   - 相关代码：`src/portfolio/optimizer.py`、`src/models/lightgbm_model.py`

2. **不确定性感知的组合优化扩展**（应对预测不确定性）
   - 用 Split/Mondrian Conformal Prediction 为每只股票的预测构造置信度；
   - 提供两种纳入方式：把不确定性并入协方差风险项，以及按 Conformal 覆盖率在“信任模型”与“退向稳健分散”之间门控的稳健混合。
   - 相关代码：`src/models/conformal*.py`、`src/portfolio/uncertainty_aware_optimizer.py`、`scripts/improved_conformal.py`

## 实验结果

回测统一采用月度点位（point-in-time）真实成分 + 20 个交易日 purge/embargo 的去前瞻口径，计入 0.1% 双边手续费与 0.1% 滑点。

长周期 2015–2024（约束优化算法相对静态基线的增益）：

| 配置 | 年化超额 | IR | 最大回撤 | 期末净值 |
|---|---:|---:|---:|---:|
| 静态多因子基线 | −8.68% | −0.99 | −77.1% | 0.45 |
| LightGBM + 约束优化器 | +3.11% | +0.20 | −39.4% | 1.45 |
| + 外部宏观 | +0.53% | +0.03 | −46.1% | 1.14 |

约束消融显示完整约束（IR 0.201）优于移除任一约束的版本，其中移除换手率约束最差（IR −0.090、年化换手由 2.19 升至 7.74）。

短周期 2024–2025（不确定性感知扩展，对标同口径不加权约束优化器 IR −0.15）：

| 方案 | IR | 夏普 | 年化超额 | 最大回撤 |
|---|---:|---:|---:|---:|
| 不加权基线 | −0.15 | 0.42 | −1.6% | −15.1% |
| 信度门控稳健混合 | +0.50 | 0.67 | +3.9% | −12.8% |
| 不确定性风险项 | +0.34 | 0.60 | +3.2% | −15.7% |

注：A 股短样本下名义 90% 的 Conformal 区间实测覆盖仅约 63%，且出现“高置信反低收益”的倒挂，因此不宜直接按置信度逐股加权。5 个随机种子下信度门控稳健混合全部由负转正（均值 −0.17 → +0.39）。

## 环境与安装

全流程不依赖 GPU。主要依赖：`numpy`/`pandas`/`scipy`、`lightgbm`、`scikit-learn`、`cvxpy` 及 `osqp`/`clarabel`/`scs`、`numba`、`matplotlib`。

```bash
conda create -n index-enhancement python=3.11 -y
conda activate index-enhancement
pip install -r requirements.txt
```

如需使用付费数据源或在线服务，请在仓库根目录的 `.env`（已 gitignore）中配置相应密钥。

## 使用方法

```bash
# 论文健康检查（校验 \cite/\ref/图片路径等，不修改文件）
python -m src.pipelines.check_paper_health

# 长周期 2015–2024 去前瞻回测（复现 IR −0.99 → +0.20）
python scripts/pit_run_leakfree.py

# 不确定性感知扩展（短样本，对标不加权基线 IR −0.15）
python scripts/improved_conformal.py

# 约束条件消融
python -m src.pipelines.run_feature_ablation

# 因子面板构建（数据源限流时耗时较长）
python -m src.pipelines.build_factor_panel --start-date 20150101 --end-date 20241231

# 重新生成论文图表
python -m src.pipelines.generate_experiment_charts
```

论文编译：`cd paper && make compile`（需 XeLaTeX + biber，或用 `make docker-compile`）。

## 目录结构

```text
IndexEnhancementStrategy/
├─ paper/            # 毕业论文 LaTeX 源、Markdown 预览与图表
├─ src/
│  ├─ config.py      # 全局配置
│  ├─ data/          # 数据源客户端与本地 CSV 加载
│  ├─ factors/       # 因子计算与预处理
│  ├─ models/        # LightGBM alpha 模型、Conformal Prediction
│  ├─ portfolio/     # 约束二次规划优化器、不确定性感知优化器
│  ├─ backtest/      # 回测引擎与绩效指标
│  ├─ pipelines/     # 实验流水线
│  └─ utils/         # 工具函数
├─ scripts/          # 去前瞻实验脚本
├─ data/             # 原始数据（缓存与结果目录已 gitignore）
├─ logs/             # 实验日志（gitignore）
└─ requirements.txt
```

## 设计要点

- 求解稳健：估计层 Ledoit-Wolf 缩减 + 求解层最小特征值修正保证半正定；三级求解器回退，失败时回退至上期或基准权重，保证长周期连续求解不中断。
- 预测与优化解耦：优化器只接收标准化 alpha 与可选置信度，上游模型可任意替换。
- 严格去偏：点位真实成分（含退市/调出股）规避幸存者偏差，purge/embargo 杜绝前瞻泄漏。

## 备注

- 论文内容以 `paper/preview.md` 为审阅基准，正式版为 `paper/manual.tex`。
- 实验日志保存在 `logs/`，过程数据与结果在 `data/processed/`（均 gitignore）。
