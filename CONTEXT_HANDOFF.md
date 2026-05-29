# 论文重构任务 · 继续上下文（Handoff）

> 本文件用于在新会话中无缝接手"毕业论文重构"任务。读完即可继续，无需回溯历史对话。
> 最后更新：2026-05-29。

---

## 0. 一句话目标

把本科毕设论文重构为聚焦标题 **《面向指数增强策略的量化算法优化设计与实现》** 的版本：**追求深度而非广度**、学术化正式文风、结构对标研究生学位论文（参考论文 `ref.pdf` / `ref.md`，写作指导 `ref-ins.md`）。

---

## 1. 关键约束与规范（必须遵守）

- **写作指导**：`ref-ins.md`。核心规则：全文 **5 章**；**每章最多 3 节、建议 2 节**；文献以正式论文/专著为主；核心章必须有**系统架构图**与**伪代码/流程图**；实验章"切忌只放图不说话"，每图须配深入分析与对比论证。
- **参考论文**：`ref.pdf`（南开硕士《GPU 近似最近邻检索系统架构设计》，可用 Read 工具直接读取，文本干净；`ref.md` 是其 OCR 版，较乱，优先读 pdf）。其写法特征：
  - 冷静技术化，**无营销修辞**；
  - 绪论=3 节（研究背景及挑战 / 本文主要工作 / 本文组织结构）；挑战分"层面"逐层做机理分析；主要工作用 • 列举且**与挑战一一对应**，每条"问题→设计→效果"；组织结构逐章叙述。
  - 实验章每张图都包在**完整论证段落**里："为验证 X，在 Y 数据/设置上做 Z 实验，结果如图所示 → 图例/坐标说明 → 现象描述（含定量对比）→ 机理/结论"。
- **当前工作流（重要）**：**先改 `paper/preview.md`（Markdown 工作稿），逐章与 ref 对齐、由用户校对；定稿后再同步到 `paper/manual.tex`**。现在 md 是内容的"真源"。
- **本地不编译 LaTeX**（无工具链 + Docker 拉镜像网络失败）；用户在**云端编译**。本地仅用 `check_paper_health.py` 做一致性校验，并可重跑实验/重绘图。
- **数据真实**：所有数字/结论来自真实实验，不得编造。允许重跑 pipeline 补数据。

---

## 2. 范围决策（已和用户敲定）

- **砍掉**：① LLM 舆情因子（原 C1）；② Stacking 异构集成（原 C2）；③ Numba/joblib 工程加速（原 C4）。
- **保留并做深**（标题主线）：**约束型组合优化算法**（正向骨架，IR 0.237→2.449）+ **Conformal 不确定性感知组合优化扩展**（算法层创新，含"高置信即过拟合"反向规律与 ACI 矫正路径）。
- 在"Stacking vs Conformal 组合优化"二选一中，**选了 Conformal/组合优化**（更贴"量化算法优化"标题、算法内容更厚、有强正向骨架）。
- LightGBM 多因子仅作为**上游 alpha 输入**（背景，非创新）。

---

## 3. 新论文结构（5 章 / 每章≤3 节）与各章状态

| 章 | 节 | 状态（在 `preview.md`） |
|---|---|---|
| 一 绪论 | 1.1 研究背景与问题挑战 / 1.2 主要工作与贡献 / 1.3 论文组织结构 | ✅ **已对齐 ref 润色** |
| 二 背景知识与相关工作 | 2.1 指数增强与组合优化理论基础（4 子节）/ 2.2 不确定性量化与 Conformal Prediction（3 子节）/ 2.3 本章小结 | ✅ **已扩充**（~13k 字，原来太薄） |
| 三 指数增强组合优化算法的设计与实现（核心） | 3.1 系统整体框架（TikZ 架构图）/ 3.2 约束型多因子组合优化算法（伪代码+复杂度）/ 3.3 不确定性感知扩展（伪代码+流程图） | ⚠️ **尚未逐句对齐 ref 第三章**（内容在，待润色） |
| 四 实验与分析 | 4.1 实验环境与设置（4.1.1/4.1.2）/ 4.2 实验结果与分析（4.2.1~4.2.4）/ 4.3 本章小结 | ✅ **已重构**（图 10→4 张，每图配完整论证段落） |
| 五 总结与展望 | 5.1 工作总结 / 5.2 局限与未来展望 | ⚠️ 基础版，待润色 |

**篇幅目标**：二/三/四章篇幅相当（对标 ref 约 18/18/13 页）；绪论/总结作为首尾章偏短合理。

### 第四章保留的 4 张图（md 中实际引用）
1. `chart_01_long_horizon_nav`（十年净值，4.2.1）
2. `chart_c3_metrics_compare`（置信加权方案对比，4.2.3）
3. `chart_c3_confidence_buckets`（置信分桶反向规律，4.2.3）
4. `chart_04_strict_oos_2025_nav_2024train`（严格OOS净值，4.2.4）
- 已删除引用：`chart_02/03/05/06/07/08`、`chart_c3_coverage_timeline`、`chart_c3_alpha_sensitivity`、`chart_summary`、`chart_c5`、所有 `chart_c1/c2/c4`。
- 覆盖率/α敏感性/特征重要性/regime/回撤 等已**改写为文字或表格**。

---

## 4. 文件清单

- `paper/preview.md` —— **当前内容真源**（Markdown 工作稿，含图、mermaid 架构图/流程图、伪代码、表格）。逐章润色就改这里。
- `paper/manual.tex` —— LaTeX 正文。**尚未与 preview.md 最新内容同步**！其中：
  - 第一章仍是旧的 2 节结构；第二章仍是旧的薄版（~5k）；第四章仍是旧的"图堆砌"版（~10 图）。
  - 第三章、abstract、main.tex 基本是新的。
  - **待办：md 定稿后，把 Ch1（3 节）、Ch2（扩充版）、Ch4（重构版+仅 4 图）以及 Ch3/Ch5 的改动同步进 manual.tex**。
- `paper/abstract.tex` —— ✅ 已重写（聚焦约束组合优化+不确定性感知 CP，中英文）。
- `paper/main.tex` —— ✅ 已加 TikZ 库：`\usetikzlibrary{patterns,positioning,fit,backgrounds,arrows.meta,calc}`。题目/作者信息已是本论文。
- `paper/nkthesis.bib` —— 51 条文献，所需 key 均已存在（Grinold1989/Clarke2002/Grinold2000/Ding2017/Michaud1989/Ledoit2004/Fama2015/Gu2020/Chen2024DeepLearningAP/Yang2025/Ke2017LightGBM/Vovk2005Conformal/Romano2019CQR/Angelopoulos2021Gentle/Kato2024CPPS/Gibbs2021ACI/Xu2023SPCI/Hallberg2024MultiStepACI/Aydin2025TCP/Yang2025ECI/Sun2025CPTC/Neglia2026ResCP 等）。被砍主题的 ~30 条会变成"未引用"（`\printbibliography[category=cited]` 只打印被引用项，无害）。
- `src/pipelines/restyle_paper_charts.py` —— ✅ 学术风重绘脚本，含 chart_01/02/03/04/05/07/08 + 4 张 c3 的绘图函数（中文标注、无图内标题、统一配色、png+pdf、`_render_if_ready` 守卫缺数据时跳过）。
- `src/portfolio/optimizer.py` / `uncertainty_aware_optimizer.py` —— 第三章算法 1/3 的源码依据。
- `src/models/conformal.py` —— 第三章算法 2（Split/Mondrian CP）源码依据。
- `src/pipelines/check_paper_health.py` —— cite/ref/figure 一致性检查（上次同步时 issues=0）。

---

## 5. 环境与命令

- **Python（务必用这个）**：`/Users/aokimi/miniconda3/envs/index-enhancement/bin/python`（matplotlib 3.10.9 / lightgbm / cvxpy / scikit-learn 齐全）。以模块方式跑：`cd 仓库根 && <python> -m src.pipelines.<name> ...`。
- 一致性检查：`<python> -m src.pipelines.check_paper_health`
- 重绘图表：`<python> -m src.pipelines.restyle_paper_charts`
- 云端编译（用户侧）：`cd paper && make compile`（xelatex+biber）。

---

## 6. 后台任务状态（截至最后更新）

- **6.5h 抓取（PID 90665）已结束**：2015–2024 因子面板已建好 → `data/processed/hs300_factor_panel_constrained_fast_extended_2015_2024.csv`（334MB）。
- **正在跑（接力的离线流水线，日志 `logs/regen_chart01.log`，完成标记 `ALL_DONE_REGEN`）**：augment_external_features ✅ → run_baseline_backtest → run_lightgbm_experiment(因子) → run_lightgbm_experiment(外部) → analyze_market_regimes → restyle_paper_charts。完成后 **`chart_01` 会自动重绘**为中文学术风。
  - 接手时先 `tail logs/regen_chart01.log` 看是否出现 `ALL_DONE_REGEN`；若中断，按下方命令补跑。
- `chart_04` 与 4 张 `chart_c3_*` 已是学术风成品。

### 若需手动补跑 chart_01（面板已存在，离线、约 20–40 分钟）
```bash
P=/Users/aokimi/miniconda3/envs/index-enhancement/bin/python
cd /Users/aokimi/code/毕设/IndexEnhancementStrategy
$P -m src.pipelines.augment_external_features --input processed/hs300_factor_panel_constrained_fast_extended_2015_2024.csv --output processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv
$P -m src.pipelines.run_baseline_backtest --input processed/hs300_factor_panel_constrained_fast_extended_2015_2024.csv --top-n 20 --use-optimizer --nav-output processed/baseline_nav_constrained_extended_2015_2024_v2.csv --positions-output processed/baseline_positions_constrained_extended_2015_2024_v2.csv --metrics-output processed/baseline_metrics_constrained_extended_2015_2024_v2.csv
$P -m src.pipelines.run_lightgbm_experiment --input processed/hs300_factor_panel_constrained_fast_extended_2015_2024.csv --top-n 20 --use-optimizer --nav-output processed/lightgbm_nav_constrained_extended_2015_2024_v3.csv --importance-output processed/lightgbm_feature_importance_constrained_extended_2015_2024_v2.csv --prediction-output processed/_p1.csv --positions-output processed/_pos1.csv --metrics-output processed/lightgbm_metrics_constrained_extended_2015_2024_v3.csv
$P -m src.pipelines.run_lightgbm_experiment --input processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv --top-n 20 --use-optimizer --use-external-features --nav-output processed/lightgbm_nav_external_2015_2024.csv --importance-output processed/lightgbm_feature_importance_external_2015_2024.csv --prediction-output processed/_p2.csv --positions-output processed/_pos2.csv --metrics-output processed/lightgbm_metrics_external_2015_2024.csv
$P -m src.pipelines.analyze_market_regimes --output processed/regime_analysis.csv || true
$P -m src.pipelines.restyle_paper_charts
```

---

## 7. 关键实验数字（论文中已用，便于核对/复用）

- **长周期增益（2015–2024，均约束型）**：静态多因子基线 IR 0.237 / 超额 4.21% / MDD −58.53%；LightGBM 驱动 IR 1.685 / 28.07% / −31.71%；+宏观与中性化 IR **2.449** / 41.31% / **−25.37%**。
- **约束消融**：完整约束 IR 1.685；无跟踪误差 1.472（反降）；无行业 1.760；无换手 2.334 但换手 2.08→8.14。
- **置信加权方案（2024–2025）**：基线(不加权) IR **2.893**；alpha缩放 2.065；候选过滤 2.582；目标惩罚 2.821（均未超基线）。
- **覆盖率 α 敏感性**：名义 95/90/80% → 实测 86.6/80.9/70.8%，偏差恒定 **−8.4~−9.2 pp**（与 α 无关）。月度覆盖 2024-08 骤降至 0.43。
- **置信分桶反向规律**：高置信 IR −0.140 / 中 −0.028 / 低 **+0.454**；命中率 42.5/41.9/48.4%（全低于 50%）。Mondrian 真实数据复测：覆盖 80.9→81.3%，高置信桶 IR 恶化至 −0.205。
- **ACI 模拟**：η=0.005→89.7%，0.01→89.5%，0.05→87.4%，0.10→84.8%（对照静态 80.9%），矫正约 80% 偏差。
- **三策略对比（2024–2025）**：等权基线 IR 0.732/换手5.13；LightGBM+优化器 IR **5.839**/换手2.30/MDD−9.22%；+Conformal目标惩罚 IR 5.634。
- **传输系数**：Top10/20/30/50 等权 IR 8.26/7.13/6.37/4.98；TC≈5.839/7.13≈**0.82**。
- **严格 OOS**：2024→2025 基线1.460/factor2.729/+ext **2.857**；2015–2024→2025 基线1.460/factor−0.482/+ext−0.283（长历史反而负，证 A 股强非平稳）。
- **随机种子稳定性**：5 种子 IR 均值 7.01 / 标准差 0.13 / 变异系数 **1.85%**。

---

## 8. 下一步待办（按优先级）

1. **继续逐章润色 `preview.md` 对齐 ref**：第三章（对照 ref 第三章"研究动机→总体架构→模块设计与优化"）、第五章；并复核第二/四章。每章给用户校对、认可后定稿。
2. 确认 `chart_01` 已重绘（查 `logs/regen_chart01.log` 的 `ALL_DONE_REGEN`，并用 Read 看图核对中文/无英文标题/三线净值）。
3. **md 全部定稿后 → 同步到 `manual.tex`**：Ch1 改 3 节、Ch2 换扩充版、Ch4 换重构版（仅 4 图、α敏感性改表、删除已弃用图的 `\includegraphics`）、并入 Ch3/Ch5 改动；同步 `\cite`/`\ref`/`\label`。
4. 跑 `check_paper_health` 确保 issues=0（尤其检查删图后无悬空 `\ref`、无缺失 `\includegraphics`）。
5. 交用户云端 `make compile`。

---

## 9. 注意事项

- 不要重新引入被砍的三块内容（LLM 舆情 / Stacking / Numba）。
- TikZ 宽流程图已用 `\resizebox{\textwidth}{!}{...}` 兜底防溢出。
- 同步 tex 时，表格用 `booktabs`、算法用 `algorithm`+`algpseudocode`（main.tex 已设 `\floatname{algorithm}{算法}`、输入/输出中文）。
- 文风：去除"乌托邦陷阱""意外揭示""答辩常问""教学价值"等口语/营销表达。
