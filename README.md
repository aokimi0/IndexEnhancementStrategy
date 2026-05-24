# IndexEnhancementStrategy

> 本科毕业设计《面向指数增强策略的量化算法优化设计与实现》
> 以沪深 300 为标的，将<code>LLM 舆情</code>、<code>Stacking 集成</code>、<code>Conformal Prediction</code> 与<code>Numba JIT 加速</code> 系统嵌入到经典 Smart Beta 框架

## 一、四项核心创新

| 编号 | 创新 | 关键指标 | 关键文件 |
|---|---|---|---|
| **C1** | 基于 Claude Opus 4.7 的中文财经舆情情感因子 | 单条新闻 5.2×10⁻⁴ USD，610 条仅 0.31 USD | `src/data/llm_sentiment.py` + `src/factors/sentiment.py` |
| **C2** | 异构轻量多模型 Stacking 集成（LGBM + XGB + GRU + Ridge + MLP 元学习器） | 真实数据 IR 5.92 击败单 LGBM IR 5.84 | `src/models/{stacking,xgboost_model,gru_model,ridge_model}.py` |
| **C3** | Split / Mondrian Conformal Prediction + 三种置信加权 QP | Mondrian 覆盖率 84.7% 高置信桶 IR +0.16 | `src/models/conformal*.py` + `src/portfolio/uncertainty_aware_optimizer.py` |
| **C4** | Numba JIT + joblib 并行高性能回测引擎 | 9.84× 加速，数值偏差 < 10⁻¹⁵ | `src/backtest/numba_kernels.py` + `src/backtest/parallel.py` |

## 二、关键实证（长周期 2015-2024 沪深 300）

| 策略 | 年化超额 | Sharpe | IR | 最大回撤 | 换手 |
|---|---:|---:|---:|---:|---:|
| 多因子等权基线（无优化器） | 4.21% | 0.168 | 0.237 | -58.53% | 0.13 |
| LightGBM + 优化器 | 28.07% | 1.289 | 1.685 | -31.71% | 2.08 |
| LightGBM + 宏观 + Ledoit-Wolf | **41.31%** | **1.881** | **2.449** | **-25.37%** | 2.24 |

短周期 2024-2025 五策略对比（详见 `data/processed/final_strategy_comparison.csv`）显示：**Stacking + 优化器 IR 5.92 > LightGBM + 优化器 IR 5.84**，验证 C2 异构集成的实证收益。

## 三、与 2026 前沿对照

本研究 §2.9 系统对照 2025-2026 同主题前沿工作，并明确定位本工作的"覆盖广度与工程可复现性优先"：

- **LLM Agent**：StockBench / QuantAgent / FinMem / AlphaAgent / Karim 2026
- **MoE**：MIGA（CSI300 24% 年化超额 SOTA）/ LLMoE / FTS-Text-MoE / AlphaMix
- **扩散模型**：Diffusion Factor Model / Diffolio
- **自适应 CP**：Gibbs ACI / SPCI / TCP / ECI（ICLR 2025）/ CPTC（NeurIPS 2025）/ ResCP

## 四、仓库结构

```text
IndexEnhancementStrategy/
├─ paper/                 # 毕业论文 LaTeX (556 行 manual.tex + 31 行 abstract.tex + 51 篇文献 bib)
│  └─ figures/            # 18 张 PNG 图表 (chart_c1~c4 + chart_summary + 早期长周期图)
├─ src/
│  ├─ data/               # akshare/baostock 客户端 + LLM 情感打分
│  ├─ factors/            # 因子计算 + sentiment 聚合
│  ├─ models/             # LightGBM/XGBoost/GRU/Ridge/Stacking/Conformal
│  ├─ portfolio/          # cvxpy 二次规划优化器 + 置信加权扩展
│  ├─ backtest/           # 回测引擎 + Numba JIT 核 + joblib 并行
│  └─ pipelines/          # 20+ 实验流水线脚本
├─ data/
│  ├─ cache/              # akshare/baostock/LLM 缓存
│  └─ processed/          # 35+ 实验数据 CSV
├─ docs/
│  └─ experiment_log.md   # 1096 行 §3.1-§3.20 完整实验记录
└─ logs/                  # 训练与失败日志
```

## 五、环境与配置

```bash
conda create -n index-enhancement python=3.11 -y
conda activate index-enhancement
pip install -r requirements.txt

# .env (gitignore 内) 配置 LLM API key
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-7
ANTHROPIC_FAST_MODEL=claude-haiku-4-5
ANTHROPIC_MAX_USD=20
```

## 六、复现入口

```bash
# 1. 构建因子面板（约 10 分钟，akshare 限流可能延长）
python -m src.pipelines.build_factor_panel     --start-date 20240101 --end-date 20250601     --output processed/hs300_panel_2024_2025.csv

# 2. 跑五策略最终对比（约 2.5 分钟）
python -m src.pipelines.final_strategy_comparison

# 3. 跑 LLM 舆情打分（haiku 控本，约 5 分钟、~0.3 USD）
python -m src.pipelines.build_sentiment_panel     --start 2025-01-01 --end 2025-05-30     --output processed/sentiment_panel.csv --max-codes 60 --max-usd 8

# 4. Numba 回测性能基准
python -m src.pipelines.benchmark_engine

# 5. 论文健康检查
python -m src.pipelines.check_paper_health

# 6. 重新生成论文图表
python -m src.pipelines.generate_innovation_charts
```

## 七、约定

- 仓库根目录的 `.cursorrules` 是项目主指南
- 训练日志统一保存在 `logs/` 目录
- 所有命名采用小写下划线或小驼峰
- 优先做日频/月频研究，不引入高频复杂度

## 八、关键稳健性证据

本研究在 11 个维度做了敏感性 / 消融测试，详见论文 §5.13。下表汇总核心证据：

| 测试维度 | 关键发现 | 论文位置 |
|---|---|---|
| 五策略最终对比 | **Stacking IR 5.92 > LightGBM IR 5.84** (带约束) | §5.9 |
| 约束消融 | 完整约束是收益-可实施性最优折衷 | §5.3 |
| 训练窗口 (12/24/36/60 月) | 12 月最优，A 股强非平稳性 | §5.10 |
| 严格 OOS (2025) | 近期 IR 2.86 / 长历史 IR -0.48 | §5.10 |
| Conformal α 敏感性 | 偏差恒定 -8.4 ~ -9.2 pp (与 α 无关) | §5.7 |
| Mondrian 行业分组 | 合成数据矫正反向 / 真实数据未矫正 | §5.7 |
| ACI 简化模拟 | γ=0.005 估算覆盖率 89.7% | §5.7 |
| Top-N (10/20/30/50) | TC ≈ 0.82 (等权 7.13 → 带约束 5.84) | §5.9 |
| Numba 规模 surface | Kernel 级加速 130-310x | §5.8 |
| 数据完整性 | 多数因子 ≥94% | §5.1 |

## 九、项目工程价值

- ~50 个 Python 文件覆盖 7 个子模块（data / factors / models / portfolio / backtest / pipelines / utils）
- 21 个端到端 pipeline 脚本
- 35+ 实验数据 CSV
- 18 张论文图表
- 51 篇文献（含 2025-2026 顶会顶刊 16 篇）
- 自研论文健康检查工具 `check_paper_health.py`
- 全程 CPU 友好，无 GPU 依赖
- LLM 调用通过 SHA-256 缓存 + 硬性预算守门，单次实验成本可压缩到 1 USD 以内

