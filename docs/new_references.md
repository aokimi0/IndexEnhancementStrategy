# 新增参考文献（2026 年 5 月增补）

本次为 `paper/nkthesis.bib` 补充 **13 篇** 2023–2025 高质量文献，围绕四项创新点 C1–C4 的最新进展，覆盖 LLM 金融、Conformal Prediction、Stacking 集成与 A 股深度学习。

注：原 bib 文件已有 `Aydin2025TCP` / `Yang2025ECI` 两条与待新增主题重复，本次按"只 append、不修改已有条目"的要求**未重复添加**对应内容；但用户后续若想用准确作者信息（Aich 等 / Wu 等），需要手动修改这两个旧条目。

## 创新点对应分布

| 创新点 | 文献数量 | BibTeX key |
|--------|---------|------------|
| C1（LLM × 中文财经舆情） | 5 | `LopezLira2024ChatGPT`、`Tang2025AlphaAgent`、`Lan2025FinChinaSA`、`Yu2024FinMem`、`Huang2024OpenFinLLMs` |
| C2（Stacking 集成 / 资产定价 / A 股深度学习） | 4 | `Mekelburg2024Pooling`、`Li2024MASTER`、`Chen2024DeepLearningAP`、`Ji2025MMFTrans`（`Tang2025AlphaAgent` 也覆盖 C2） |
| C3（Conformal Prediction + 不确定性量化） | 4 | `Gibbs2021ACI`、`Xu2023SPCI`、`Kato2024CPPS`、`Hallberg2024MultiStepACI` |
| C4（高性能回测） | 0 | 既有 `Lam2015Numba` 已覆盖，C4 偏工程，无对应顶会论文 |

C1 与 C3 是本次新增最密集的方向（各 4–5 篇），因为 LLM × Finance 和 CP × Time Series 在 2023–2025 涌现大量顶会顶刊新方法，是本文创新点最需要文献支撑的两块。

## 必引文献 Top 5（强烈建议直接进 paper 正文）

1. **`LopezLira2024ChatGPT`** — Lopez-Lira & Tang (2024)
   - LLM × 金融预测领域被引最广的奠基论文之一，证明 GPT-4 可从新闻标题直接预测下一日股价反应，且效果在 LLM 容量阈值之上才显现；为本文 C1 提供"为什么用 Claude Opus 4.7 而非 BERT/GPT-2"的直接理论依据。

2. **`Tang2025AlphaAgent`** — Tang et al. (KDD 2025)
   - 在 **CSI 500 中国 A 股** 与 S&P 500 上验证 LLM 驱动因子挖掘有效性，是与本文 C1 / C2 创新点最贴近的同期工作，必须对照。

3. **`Gibbs2021ACI`** — Gibbs & Candès (NeurIPS 2021)
   - Adaptive Conformal Inference 算法奠基论文。C3 整章方法学必引。

4. **`Kato2024CPPS`** — Kato (2024)
   - 直接把 Conformal Prediction 应用到组合选择 (CPPS)，与本文 C3「置信加权 QP 优化」高度对应，是 C3 最近似的同期工作。

5. **`Chen2024DeepLearningAP`** — Chen, Pelger & Zhu (Management Science 2024)
   - Management Science 顶刊，深度学习实证资产定价的标志性工作；为本文 C2 异构 Stacking 集成提供权威「ML × 资产定价」引用支撑。

## 全部 13 篇详细清单

### C1 LLM × 量化金融（5 篇）

**1. LopezLira2024ChatGPT**
- 标题：Can ChatGPT forecast stock price movements? Return predictability and large language models
- 作者：Alejandro Lopez-Lira, Yuehua Tang
- 年份：2024（arXiv v5）
- 出处：arXiv:2304.07619
- 核心论点：GPT-4 对新闻标题打分（好/坏/中性）可直接预测出下一日股票收益；模型容量越大效果越好，基本款 GPT-1/GPT-2/BERT 几乎无效；负面新闻与小盘股上效果尤为显著。
- 对应创新点：C1

**2. Tang2025AlphaAgent**
- 标题：AlphaAgent: LLM-driven alpha mining with regularized exploration to counteract alpha decay
- 作者：Ziyi Tang, Zechuan Chen, Jiarui Yang, Jiayao Mai, Yongsen Zheng, Keze Wang, Jinrui Chen, Liang Lin
- 年份：2025
- 出处：KDD 2025（arXiv:2502.16789）
- 核心论点：用 LLM Agent 自动生成可解释、抗衰减 alpha 因子，通过 AST 相似度约束防止同质化；在 CSI 500 与 S&P 500 上长期保持 alpha 显著。
- 对应创新点：C1（LLM 因子构造），并附带 C2 对照基线

**3. Lan2025FinChinaSA**
- 标题：Chinese fine-grained financial sentiment analysis with large language models
- 作者：Yinyu Lan, Yanru Wu, Wang Xu, Weiqiang Feng, Youhao Zhang
- 年份：2025
- 出处：Neural Computing and Applications, 37: 24883–24892
- 核心论点：构造 11036 条标注中文财经新闻数据集 FinChina SA；评测多个开源 LLM 与 ChatGPT 在细粒度（公司级）情感分类上的能力，发现 LLaMA 系列经指令微调后显著超越 zero-shot ChatGPT。
- 对应创新点：C1（中文舆情数据与基线对照）

**4. Yu2024FinMem**
- 标题：FinMem: A performance-enhanced LLM trading agent with layered memory and character design
- 作者：Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, Yang Li, Denghui Zhang, Rong Liu, Jordan W. Suchow, Khaldoun Khashanah
- 年份：2024
- 出处：arXiv:2311.13743（ICAIF 2024）
- 核心论点：LLM 交易 Agent 框架，引入分层记忆 + 性格设定模拟人类交易员；在真实股票数据上累积收益显著超越 RL/算法 baseline。
- 对应创新点：C1（LLM Agent 决策架构对照）

**5. Huang2024OpenFinLLMs**
- 标题：Open-FinLLMs: Open multimodal large language models for financial applications
- 作者：Jimin Huang, Mengxi Xiao, Dong Li, Qianqian Xie, Alejandro Lopez-Lira, Benyou Wang, Xiao-Yang Liu, Sophia Ananiadou 等 44 人
- 年份：2024
- 出处：arXiv:2408.11878
- 核心论点：首个开源多模态金融 LLM 套件（FinLLaMA / FinLLaMA-Instruct / FinLLaVA），在 14 个金融任务上超越 GPT-4，提供本文 C1 在本地推理时可替代 Claude Opus 的开源对照。
- 对应创新点：C1（开源 LLM 备选基线）

### C2 Stacking / Ensemble / A 股深度学习（4 篇）

**6. Mekelburg2024Pooling**
- 标题：Pooling and winsorizing machine learning forecasts to predict stock returns with high-dimensional data
- 作者：Erik Mekelburg, Jack Strauss
- 年份：2024
- 出处：Journal of Empirical Finance, 79: 101538
- 核心论点：单一 ML 模型（LASSO、RF、XGBoost、NN、LightGBM）在中、美、加、德、英多个市场普遍 OOS 失败；只有「pooling + winsorizing」集成才能稳定预测股票收益。直接论证本文 C2 集成框架的必要性。
- 对应创新点：C2

**7. Li2024MASTER**
- 标题：MASTER: Market-guided stock transformer for stock price forecasting
- 作者：Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, Sen Huang
- 年份：2024
- 出处：AAAI 2024, 38(1): 162–170
- 核心论点：用市场指数信息引导跨股票横截面注意力的 Transformer 架构，在 CSI 300 / CSI 500 / CSI 800 实证，比基线提升 13% 排序指标、47% 组合指标。与本文 C2 GRU baseline 同台对照。
- 对应创新点：C2（A 股 Transformer 强基线）

**8. Chen2024DeepLearningAP**
- 标题：Deep learning in asset pricing
- 作者：Luyang Chen, Markus Pelger, Jason Zhu
- 年份：2024
- 出处：Management Science, 70(2): 714–750
- 核心论点：用 GAN + LSTM + FFN 估计条件 SDF；在美股全样本上同时优化 Sharpe、解释力与定价误差，论证深度学习因子相对传统线性模型的显著优势。Management Science 顶刊为 C2 提供权威背书。
- 对应创新点：C2

**9. Ji2025MMFTrans**
- 标题：Chinese stock prediction based on a multi-modal transformer framework: Macro-micro information fusion
- 作者：Shihao Ji, Zihui Song, Fucheng Zhong, Jisen Jia, Zhaobo Wu, Zheyi Cao, Tianhao Xu
- 年份：2025
- 出处：arXiv:2501.16621
- 核心论点：四通道并行编码器（技术指标 / 财经文本 / 宏观 / 事件图）+ 动态门控融合，在 CSI 300 实证将 RMSE 降低 23.7%、Sharpe 提升 31.9%。直接对应本文「LLM 舆情 + 多因子融合」的多模态思路。
- 对应创新点：C2（多模态融合对照）

### C3 Conformal Prediction × 时间序列 / 金融（4 篇）

**10. Gibbs2021ACI**
- 标题：Adaptive conformal inference under distribution shift
- 作者：Isaac Gibbs, Emmanuel Candès
- 年份：2021
- 出处：NeurIPS 2021, 34: 1660–1672
- 核心论点：奠基性 ACI 算法，在线 update 显著性水平 α_t，对非可交换/分布漂移数据也保证长期覆盖率收敛到 1−α。C3 整章必引。
- 对应创新点：C3

**11. Xu2023SPCI**
- 标题：Sequential predictive conformal inference for time series
- 作者：Chen Xu, Yao Xie
- 年份：2023
- 出处：ICML 2023, PMLR 202: 38707–38727
- 核心论点：利用残差时间相关性，用分位数回归在线自适应地估计未来非一致性得分分位数；理论上保证条件覆盖率渐近有效，区间宽度比 ACI 显著更窄。
- 对应创新点：C3

**12. Kato2024CPPS**
- 标题：Conformal predictive portfolio selection
- 作者：Masahiro Kato（Mizuho-DL Financial Technology）
- 年份：2024
- 出处：arXiv:2410.16333
- 核心论点：提出 CPPS 框架，将 CP 直接嵌入组合选择决策；既支持均值-方差也支持分位数（尾部风险）目标。直接对应本文 C3「置信加权 QP」的设计思路。
- 对应创新点：C3（与本文创新点最贴合的对照工作）

**13. Hallberg2024MultiStepACI**
- 标题：Adaptive conformal inference for multi-step ahead time-series forecasting online
- 作者：Johan Hallberg Szabadváry
- 年份：2024
- 出处：PMLR 230: 250–263（CoPA 2024）
- 核心论点：把 ACI 扩展到多步预测，每步使用独立目标错误率与学习率；通过 conformalised ridge regression + MIMO 策略示范如何在 multi-horizon 预测中保持 finite-sample 覆盖。
- 对应创新点：C3（如本文做多日 holding period 预测可参考）

## 已存在但作者信息有误的条目（建议手动修订）

| 已有 key | 实际论文 | 正确作者 |
|---------|---------|---------|
| `Aydin2025TCP` (arXiv:2507.05470) | 同 TCP 论文 | Agnideep Aich, Ashit Baran Aich, Dipak C. Jain |
| `Yang2025ECI` (arXiv:2502.00818) | 同 ECI 论文 | Junxi Wu, Dongjian Hu, Yajie Bao, Shu-Tao Xia, Changliang Zou |

按用户要求未修改这两条，但实际作者已通过 web 验证；如要严格规范，请人工 patch 这两条作者字段。

## 筛选总结

- **顶刊顶会论文**：8 篇（Management Science 1, J. Empirical Finance 1, AAAI 1, NeurIPS 1, ICML 1, KDD 1, Neural Comp. & Appl. 1, PMLR CoPA 1）
- **高质量 arXiv preprint**：5 篇（全部 2024–2025）
- **A 股市场实证**：3 篇直接在 CSI 300/500/800 上验证（`Tang2025AlphaAgent`、`Li2024MASTER`、`Ji2025MMFTrans`）
- **中文情感分析专门数据集**：1 篇（`Lan2025FinChinaSA` 11k 条标注语料）

新增文献覆盖了本文 4 项创新点中最重要的 C1/C2/C3 三大方向，C4 因属工程加速，原有 `Lam2015Numba` 引用已足够。所有条目均严格按 BibTeX 规范追加到 `paper/nkthesis.bib` 文件末尾，未修改任何已有条目。
