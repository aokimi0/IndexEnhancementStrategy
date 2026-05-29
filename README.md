# 面向指数增强策略的量化算法优化设计与实现

> 本科毕业设计。以沪深 300 指数增强为应用场景，围绕其**核心量化算法——带约束的组合优化**做深度（而非广度）的优化设计与实现。
>
> 指数增强的本质，是在跟踪误差、行业偏离、个股权重与换手率等多重约束界定的可行域内，把含噪的截面 alpha 预测高效、稳健地转化为可实施的主动权重，即最大化主动管理基本定律 `IR = TC·IC·√N` 中的传输系数 `TC`。
>
> 开源仓库：<https://github.com/aokimi0/IndexEnhancementStrategy>

## 一、两项工作（对应 A 股落地的两重挑战）

| 工作 | 应对挑战 | 做法 | 关键文件 |
|---|---|---|---|
| **约束型多因子组合优化算法** | 挑战一：协方差病态与"误差最大化" | LightGBM 截面 alpha → Ledoit-Wolf 缩减 + 最小特征值修正 → 带跟踪误差/行业/权重/换手约束的二次规划 → OSQP→CLARABEL→SCS 三级求解器回退 | `src/portfolio/optimizer.py`、`src/models/lightgbm_model.py` |
| **不确定性感知的组合优化扩展** | 挑战二：低信噪比下的预测不确定性 | Split/Mondrian Conformal 置信度 → ①不确定性进协方差风险项 ②以 Conformal 覆盖率门控"信任 ML / 退向稳健分散"比例的信度门控稳健混合 | `src/models/conformal*.py`、`src/portfolio/uncertainty_aware_optimizer.py`、`scripts/improved_conformal.py` |

## 二、关键实证

所有长短周期均采用**月度点位（point-in-time）真实成分** + **20 个交易日 purge/embargo** 严格去前瞻口径，计入 0.1% 双边手续费与 0.1% 滑点。

### 长周期 2015–2024（约束优化算法的增益）

| 配置 | 年化超额 | IR | 最大回撤 | 期末净值 |
|---|---:|---:|---:|---:|
| 静态多因子基线 | −8.68% | −0.99 | −77.1% | 0.45 |
| **LightGBM + 约束优化器** | **+3.11%** | **+0.20** | **−39.4%** | **1.45** |
| + 外部宏观 | +0.53% | +0.03 | −46.1% | 1.14 |

约束消融：完整约束 IR **0.201** 优于移除任一约束的版本（无换手率约束最差，IR 转负 −0.090、换手由 2.19 飙至 7.74），印证传输系数理论下"以可控 TC 损失换可实施性"的权衡。

### 短周期 2024–2025（不确定性感知扩展，对标同口径不加权约束优化器 IR −0.15）

| 方案 | IR | 夏普 | 年化超额 | 最大回撤 |
|---|---:|---:|---:|---:|
| 不加权基线 | −0.15 | 0.42 | −1.6% | −15.1% |
| **信度门控稳健混合** | **+0.50** | 0.67 | +3.9% | −12.8% |
| 不确定性风险项 | +0.34 | 0.60 | +3.2% | −15.7% |
| （稳健因子腿，机理参考） | +1.11 | 0.86 | +9.6% | −13.2% |

- **失效诊断**：A 股短样本下名义 90% 的 Conformal 区间实测覆盖仅约 **63%**，且置信度与收益"高置信反低收益"倒挂——故不宜直接按置信度逐股加权。
- **稳定性**：5 个随机种子下，不加权基线全部为负（均值 −0.17），信度门控稳健混合全部转正（均值 +0.39），逐种子稳定超越。

## 三、仓库结构

```text
IndexEnhancementStrategy/
├─ paper/                       # 毕业论文 LaTeX 源
│  ├─ main.tex / abstract.tex / manual.tex / references.tex ...
│  ├─ preview.md                # 论文 Markdown 预览（内容审阅基准）
│  └─ figures/                  # 论文实际引用的图（arch5 架构图 + 4 张实验图）
├─ src/
│  ├─ config.py                 # 全局配置
│  ├─ data/                     # akshare/baostock/tushare 客户端 + 本地 CSV 加载
│  ├─ factors/                  # 因子计算（engine）+ 预处理（MAD 缩尾 / Z-Score / 行业中性）
│  ├─ models/                   # LightGBM alpha 模型 + Split/Mondrian Conformal
│  ├─ portfolio/                # 约束二次规划优化器 + 不确定性感知优化器
│  ├─ backtest/                 # 回测引擎 + 绩效指标 + Numba 核
│  ├─ pipelines/                # 实验流水线（面板构建 / 回测 / 消融 / Conformal / 健康检查）
│  └─ utils/                    # 控制台工具
├─ scripts/                     # 点位(PIT)+去前瞻(embargo) leak-free 实验脚本 + 改进 Conformal
├─ data/
│  ├─ csi300_raw/               # 沪深 300 原始数据（成分 / 估值 / 财务 / 行情）
│  ├─ cache/                    # 数据源缓存（gitignore）
│  └─ processed/                # 实验面板与结果 CSV（gitignore）
├─ logs/                        # 实验日志（gitignore）
└─ requirements.txt
```

## 四、环境与安装

全流程 **CPU 友好、无 GPU 依赖**。核心栈：`numpy`/`pandas`/`scipy`、`lightgbm`、`scikit-learn`（Ledoit-Wolf 与 Conformal 校准）、`cvxpy` + `osqp`/`clarabel`/`scs`（二次规划）、`numba`（回测核）、`matplotlib`。

```bash
conda create -n index-enhancement python=3.11 -y
conda activate index-enhancement
pip install -r requirements.txt

# 行情/财务/宏观数据源（可选，按需）；如使用 LLM/付费数据源，
# 在仓库根 .env（gitignore 内）中配置对应密钥。
```

## 五、复现入口

```bash
# 1) 论文健康检查（\cite/\ref/图片路径/未用条目静态校验，不改文件）
python -m src.pipelines.check_paper_health

# 2) 长周期 2015–2024 严格口径（点位成分 + 20 日 embargo）LightGBM 两配置净值/指标
#    复现表 4-2 的 IR −0.99 → +0.20
python scripts/pit_run_leakfree.py

# 3) 不确定性感知扩展改进方案（leak-free 短样本，对标不加权基线 IR −0.15）
#    复现表 4-5：信度门控 +0.50 / 不确定性风险项 +0.34
python scripts/improved_conformal.py

# 4) 约束条件消融（逐一移除跟踪误差/行业/换手约束）
python -m src.pipelines.run_feature_ablation

# 5) 因子面板构建（akshare 限流时耗时较长；日期与输出按需指定）
python -m src.pipelines.build_factor_panel --start-date 20150101 --end-date 20241231

# 6) 重新生成论文图表
python -m src.pipelines.generate_experiment_charts
```

> 论文编译：`cd paper && make compile`（需 XeLaTeX + biber；亦可 `make docker-compile` 使用 TeXLive 2022 镜像）。

## 六、设计要点

- **求解稳健**：估计层 Ledoit-Wolf 缩减压低条件数，求解层对称化 + 最小特征值修正保证半正定；OSQP→CLARABEL→SCS 三级回退，失败则回退上期/基准权重，保证长周期数千次连续求解不中断。
- **预测与优化解耦**：优化器只接收标准化 alpha 与（可选）置信度，上游模型可任意替换，便于消融与复用。
- **严格去偏**：点位真实成分（含退市/调出股）规避幸存者偏差，purge/embargo 杜绝前瞻泄漏；不确定性门控的覆盖率只用 `t − embargo` 前已实现标签计算，严格 leak-free。

## 七、约定

- 优先做日频 / 月频研究，不引入高频复杂度。
- 实验日志统一保存在 `logs/`，过程数据与结果在 `data/processed/`（均 gitignore）。
- 论文内容以 `paper/preview.md` 为审阅基准，正式版为 `paper/manual.tex`。
