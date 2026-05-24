# 四项创新点速查卡（A4 双面打印）

> 答辩前 30 秒可读完的核心数据卡片，每项创新点压缩到一行核心定义 + 三条关键证据。

## 创新点矩阵总览

| 编号 | 一句话定义 | 关键模块 | 核心实证 |
|---|---|---|---|
| C1 | Claude Opus 4.7 中文舆情情感因子（LLM-as-a-Service） | `src/data/llm_sentiment.py` | 0.31 USD / 610 条新闻，polarity std=0.516 |
| C2 | LightGBM+XGB+GRU+Ridge 异构 Stacking | `src/models/stacking.py` | IR 5.92 > LGBM 5.84（带约束）|
| C3 | Split/Mondrian CP + 三种置信加权 QP | `src/models/conformal*.py` | 偏差恒定 -9pp，"高置信即过拟合"反向规律 |
| C4 | Numba JIT + joblib 并行高性能引擎 | `src/backtest/numba_kernels.py` | Kernel 级 130-310×、Engine 级 9.84× |

---

## C1 LLM 舆情情感因子

**问题**：传统量化策略因子来源单一，无法捕捉政策变动 / 突发事件等结构化数据滞后信号。

**方案**：
- 用 Claude Opus 4.7 / Haiku 4.5 API 对沪深300新闻打分（polarity / intensity / topic 四元组）
- 工程范式：SHA-256 缓存 + 硬性预算守门 + 批量 prompt + 双通道路由（Haiku 控本、Opus 提质）
- 日频聚合：按时间衰减加权（半衰期 24h），16:00 后新闻计入下一交易日避免泄露

**关键数据**：
- 610 条新闻打分实测 **0.31 USD**（单条 5.2×10⁻⁴ USD）
- polarity 标准差 0.516（不退化为常数）
- 论文位置：§4.2 + §5.5 + experiment_log §3.15 / §3.19

---

## C2 异构轻量多模型 Stacking

**问题**：单 LightGBM 在 A 股低信噪比环境下方差较大；深度模型对算力要求高、解释性差。

**方案**：
- L1 模型：LightGBM + XGBoost + 轻量 GRU + Ridge（全部 CPU 友好、参数 ≤ 百万级）
- 元学习器：MLP（默认线性），输入 OOF KFold 预测 + 置信特征
- 元学习器自动学习 L1 权重，对相关性高的模型施加负向校正

**关键数据**：
- 带优化器情境下 **IR 5.92 > 单 LightGBM IR 5.84**（+1.4% 相对）
- 元学习器给 Ridge 赋负权重（-0.25），对 XGBoost 主导（+1.0）
- 论文位置：§4.3 + §5.6 + §5.9

---

## C3 Conformal Prediction 不确定性量化

**问题**：单点预测无法判断"模型有多确信"，导致组合优化时高低置信度同等对待。

**方案**：
- Split CP：训练 70% + 校准 30%，用残差分位数构造 90% 置信区间
- Mondrian CP：按申万一级行业分组校准
- 三种 QP 置信加权：alpha_scale / candidate_filter / objective_penalty

**关键数据**：
- 覆盖率偏差 **恒定 -9 pp**（与 α=0.05/0.10/0.20 均无关）→ 系统性方法论局限
- 反向规律：高置信桶 IR -0.14、低置信桶 IR +0.45
- Mondrian 在合成数据上矫正反向规律，但**真实数据上未矫正**
- ACI 简化模拟：γ=0.005 可矫正 ~80% 偏差
- 论文位置：§4.4 + §5.7（含 6 个子节）

---

## C4 Numba JIT 高性能回测引擎

**问题**：传统 Python 按日循环回测在十年面板上耗时数十秒，超参网格搜索代价高。

**方案**：
- `@numba.njit(cache=True)` 编译 NAV 累乘核函数
- 纯 NumPy fallback（无 numba 环境降级运行）
- joblib 并行消融实验、训练窗口探索

**关键数据**：
- **Kernel 级 130-310× 加速**（vs 纯 Python 嵌套 for-loop）
- Engine 级 9.84× 加速（含 pandas/dict 不可加速开销）
- 数值偏差 < 10⁻¹⁵（不以正确性为代价）
- 论文位置：§4.5 + §5.8（含 Kernel vs Engine 双层分析）

---

## 综合稳健性（一张表答全部）

| 测试 | 关键发现 | 结论 |
|---|---|---|
| 5-seed | IR 变异系数 1.85% | 非 cherry-picked |
| α 敏感性 | 偏差恒定 -9 pp | 与 α 无关，C3 方法学局限 |
| Mondrian 合成 vs 真实 | 合成矫正 / 真实未矫正 | C3 反向规律稳健 |
| train_months 12/24/36/60 | IR 单调下降 | A 股非平稳，短窗口胜出 |
| top_n 10/20/30/50 | IR 单调下降 | Top 20 是 IR-回撤折衷点 |
| Numba 50-500 股 | 加速 130-310× | 工程加速可推广 |

---

## 一句话答辩开场白

> 本文针对沪深300指数增强场景，从 LLM 工程化、异构集成、不确定性量化、高性能计算四个计算机科学方向系统改造经典 Smart Beta 框架；带约束的 Stacking 在真实数据上小幅但稳定击败单 LightGBM（IR 5.92 vs 5.84），并发现 Conformal Prediction 在 A 股低信噪比环境下"高置信即过拟合"的反向规律；所有实验在 CPU 单机可复现、LLM 调用单次实验成本 < 1 美元。
