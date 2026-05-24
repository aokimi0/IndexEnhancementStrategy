# 答辩 PPT 大纲

约 22 页 PPT，按 10 分钟陈述节奏组织。

## 封面页 1
- 题目：面向指数增强策略的量化算法优化设计与实现
- 学号 2210556 廖望 / 指导教师 刘晓光教授
- 计算机科学与技术专业 / 计算机学院 / 2026.05

## 一、研究背景与动机（2 页）

### Slide 2 - A 股指数增强场景
- 沪深 300 = 中国大盘蓝筹基准
- Smart Beta：被动低成本 + 主动超额收益的折衷
- A 股非有效市场特征（散户主导 / 政策市 / 流动性敏感）

### Slide 3 - 课题书的三大问题
- 传统策略计算量大、优化耗时长
- 因子来源单一（仅价格 / 财务）
- 缺乏不确定性量化与风险感知
- → 引入"AI 辅助决策 + 计算加速 + 外部数据"

## 二、四项核心创新（4 页）

### Slide 4 - C1 LLM 舆情情感因子
- LLM-as-a-Service：Claude Opus 4.7 / Haiku 4.5 双通道
- SHA-256 缓存 + 硬性预算守门 + 批量 prompt
- 实测：610 条新闻 0.31 USD（5.2×10⁻⁴ USD/条）
- 互补传统价值 / 质量 / 动量因子

### Slide 5 - C2 异构 Stacking 集成
- LightGBM + XGBoost + GRU + Ridge → MLP 元学习器
- OOF KFold 训练规避标签泄漏
- 全部参数量 ≤ 百万级、CPU 友好
- 实证：Stacking IR 5.92 > LightGBM IR 5.84（带约束）

### Slide 6 - C3 Conformal Prediction
- Split / Mondrian CP 输出 90% 置信区间
- 三种置信加权 QP：alpha_scale / candidate_filter / objective_penalty
- 关键发现：A 股"高置信即过拟合"反向规律
- α 敏感性：偏差恒定 -9 pp（系统性方法论局限）

### Slide 7 - C4 Numba JIT + joblib 加速
- 回测核心循环 @numba.njit 编译
- Kernel 级 130-310× 加速、Engine 级 9.84×
- 数值偏差 < 10⁻¹⁵ → 加速不以正确性为代价
- 呼应课题书"计算量大、优化耗时"

## 三、对照 2026 前沿（1 页）

### Slide 8 - 与 2026 SOTA 对照
- LLM Agent：StockBench / QuantAgent / FinMem / AlphaAgent
- MoE：MIGA（CSI300 24% 年化超额 SOTA）/ LLMoE / FTS-Text-MoE
- 扩散模型：Diffusion Factor Model / Diffolio
- 自适应 CP：Gibbs ACI / SPCI / TCP / ECI / CPTC / ResCP
- 本文定位：**覆盖广度与工程可复现性优先**，非单方向 SOTA

## 四、实证研究（5 页）

### Slide 9 - 长周期对比（2015-2024）
- 表格：基线 IR 0.24 → 约束 LightGBM IR 1.69 → 宏观增强 IR 2.45
- 最大回撤 -58.5% → -25.4%

### Slide 10 - 五策略最终对比（2024-2025）
- 横向柱：基线 / LGBM / LGBM+优化器 / Stacking+优化器 / Conformal+优化器
- 显示 IR 误差棒（5-seed std = 0.13）
- 关键：Stacking 击败单 LGBM

### Slide 11 - C1 sentiment 信号质量
- 610 条新闻打分极性分布 + topic 分布
- polarity std=0.516 表示有效区分极性
- 时间移位 demo 局限（覆盖率仅 0.32%）

### Slide 12 - C3 Conformal 反向规律
- 覆盖率 timeline + 置信桶 IR（top -0.14 / bottom +0.45）
- 三层稳健性验证：α 敏感性 + Mondrian 真实数据 + ACI 模拟

### Slide 13 - C4 加速比 surface
- Kernel 级 50-300 股 × 252-2520 日的加速比 130-310×
- Engine 级 9.84×，未来优化方向

## 五、综合稳健性证据（1 页）

### Slide 14 - 11 维度敏感性测试
- 约束消融 / 特征分组 / 训练窗口 / 严格 OOS / α 敏感性 / Mondrian / ACI / Top-N / Numba 规模 / 五策略对比 / 数据完整性
- 关键稳健性：5-seed IR 变异系数 1.85%（非 cherry-picked）

## 六、结论与展望（2 页）

### Slide 15 - 核心结论
- C1 LLM 工程化范式：可控成本日频 alpha 源
- C2 Stacking 击败单模型（带约束 IR 5.92 vs 5.84）
- C3 反向规律 + 系统性偏差（C3 方法学贡献）
- C4 Kernel 级 130-310× 加速

### Slide 16 - 局限与未来工作
- C1 sentiment 长样本验证
- C2 升级 MoE 路由（参考 MIGA / LLMoE）
- C3 升级自适应 CP（TCP / ECI / CPTC / ResCP）
- C4 跨节点 Dask / Ray + GPU 后端

## 七、工程贡献（1 页）

### Slide 17 - 项目工程统计
- ~50 个 Python 模块 / 21 个 pipeline 脚本 / 35+ 实验 CSV
- 51 篇文献（含 16 篇 2025-2026 顶会顶刊）
- 自研 check_paper_health.py 工具
- 完整复现指南 + 超参数总表 + 创新点代码对照

## 八、答疑准备（5 页隐藏）

### Slides 18-22（隐藏页，按需展开）
- Q：等权 Top20 IR 13.73 是不是 cherry-picked？→ 见 §5.12 seed 稳定性 1.85%
- Q：sentiment 端到端 IR 为什么是负？→ §3.19 时间移位 demo 局限
- Q：Conformal 反向规律会不会数据偶然？→ 三层独立测试佐证
- Q：为什么用 LightGBM 不用深度学习？→ §4.1 + Yang 2025 LightGBM 在 A 股低 SNR 优势
- Q：实盘可行性？→ LightGBM + 优化器 IR 5.84、换手 2.30 是落地基线

详见 docs/defense_qa.md 20 个核心问题。

## 致谢页 23
- 感谢指导教师刘晓光教授
- 感谢南开大学计算机学院
- 感谢家人朋友支持
