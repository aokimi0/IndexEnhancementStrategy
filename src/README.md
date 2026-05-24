# 代码模块清单

本目录按职责分层组织，共 7 个子模块、~50 个 Python 文件。

## 一、目录结构

```
src/
├── config.py                              # 项目路径与环境变量配置
├── data/                                  # 数据接入与对齐
│   ├── akshare_client.py                  # akshare + baostock 主客户端（含 BaoStock 补全估值/财务）
│   ├── tushare_client.py                  # tushare 备用客户端
│   ├── loaders.py                         # DataService 拉取拼接缓存
│   ├── news_client.py                     # [C1] 东方财富新闻接口封装
│   └── llm_sentiment.py                   # [C1] Claude API 封装 + BudgetTracker + SHA-256 缓存
├── factors/                               # 因子计算与预处理
│   ├── engine.py                          # 16 个数值因子计算 + 行业中性化
│   ├── preprocess.py                      # MAD 缩尾 + Z-Score
│   └── sentiment.py                       # [C1] 日频聚合 + 时间衰减
├── models/                                # 机器学习模型
│   ├── base.py                            # [C2] AlphaModelBase 抽象基类
│   ├── lightgbm_model.py                  # LightGBM 滚动/冻结训练
│   ├── xgboost_model.py                   # [C2] XGBoost L1
│   ├── gru_model.py                       # [C2] PyTorch CPU GRU L1
│   ├── ridge_model.py                     # [C2] Ridge L1
│   ├── stacking.py                        # [C2] OOF + MLP 元学习器
│   ├── conformal.py                       # [C3] Split / Mondrian Conformal Prediction
│   └── conformal_lightgbm.py              # [C3] 与 LightGBM 解耦的包装层
├── portfolio/                             # 组合优化
│   ├── optimizer.py                       # cvxpy 二次规划 + Ledoit-Wolf
│   └── uncertainty_aware_optimizer.py     # [C3] 三种置信加权 QP
├── backtest/                              # 回测引擎
│   ├── engine.py                          # BaselineBacktestEngine（含 numba 集成）
│   ├── metrics.py                         # 17 项绩效指标
│   ├── numba_kernels.py                   # [C4] @njit NAV 累乘核 + NumPy fallback
│   └── parallel.py                        # [C4] joblib 并行 + 模型缓存
├── pipelines/                             # 21 个端到端实验脚本
│   ├── build_factor_panel.py              # 构建沪深300因子面板
│   ├── augment_external_features.py       # 补北向/M2/利差等外部特征
│   ├── collect_higher_frequency_data.py   # 采集分钟级高频数据
│   ├── build_sentiment_panel.py           # [C1] 端到端 LLM 舆情管线
│   ├── run_baseline_backtest.py           # 多因子基线回测
│   ├── run_lightgbm_experiment.py         # LightGBM 实验（滚动/冻结）
│   ├── run_feature_ablation.py            # 特征分组消融
│   ├── compare_feature_ablation.py        # 消融汇总
│   ├── compare_strategies.py              # 策略对比
│   ├── run_stacking_experiment.py         # [C2] 5 variant Stacking 实验
│   ├── run_conformal_experiment.py        # [C3] 4 scheme Conformal 实验
│   ├── validate_coverage.py               # [C3] 覆盖率与置信分桶分析
│   ├── sentiment_ir_uplift.py             # [C1] sentiment 因子 IR 增益对照
│   ├── benchmark_engine.py                # [C4] Numba 性能基准
│   ├── explore_window_length.py           # 训练窗口长度消融
│   ├── analyze_market_regimes.py          # 牛熊周期分段
│   ├── final_strategy_comparison.py       # 五策略端到端对比（论文 §5.9）
│   ├── generate_experiment_charts.py      # 早期论文图表
│   ├── generate_innovation_charts.py      # C1-C4 创新点图表
│   └── check_paper_health.py              # 论文 cite/label/figure 一致性检查
└── utils/
    └── console.py                          # Windows/UTF-8 终端编码
```

## 二、关键运行入口

```bash
# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh && conda activate index-enhancement

# 1. 数据准备（约 40 分钟，akshare 限流时更久）
python -m src.pipelines.build_factor_panel \
    --start-date 20240101 --end-date 20250601 \
    --output processed/hs300_panel_2024_2025.csv

# 2. C1 LLM 舆情打分（haiku 控本，约 5 分钟、~0.3 USD）
python -m src.pipelines.build_sentiment_panel \
    --start 2025-01-01 --end 2025-05-30 \
    --output processed/sentiment_panel.csv --max-codes 60 --max-usd 8

# 3. C2 Stacking 集成
python -m src.pipelines.run_stacking_experiment \
    --input processed/hs300_panel_2024_2025_v2.csv \
    --variants lgbm xgb ridge stacking --use-external-features

# 4. C3 Conformal Prediction
python -m src.pipelines.run_conformal_experiment \
    --input processed/hs300_panel_2024_2025_v2.csv \
    --schemes baseline alpha_scale candidate_filter objective_penalty \
    --use-external-features

# 5. C4 Numba 性能基准
python -m src.pipelines.benchmark_engine

# 6. 五策略最终对比（约 2.5 分钟）
python -m src.pipelines.final_strategy_comparison

# 7. 论文图表与健康检查
python -m src.pipelines.generate_innovation_charts
python -m src.pipelines.check_paper_health
```

## 三、约定

- 所有失败 ts_code 写入 `logs/akshare_failures.log`、`logs/news_failures.log`
- 成功请求数据缓存到 `data/cache/{daily,daily_basic_v2,financial_indicators_v2,news,llm_sentiment}/`
- 缓存策略：单股票文件 + SHA-256(text) 哈希
- LLM 调用通过 `.env` 中 `ANTHROPIC_MAX_USD` 硬性预算守门
- 所有模型 CPU 友好、参数量 ≤ 百万级
