# -*- coding: utf-8 -*-
"""一次性润色 paper/manual.tex 与 paper/abstract.tex：
- 去除 C1/C2/C3/C4 项目代号
- enumerate 散文化
- 章节、caption、行文里的过程性表述统一
"""

from __future__ import annotations

from pathlib import Path

PAPER = Path("paper")
MANUAL = PAPER / "manual.tex"
ABSTRACT = PAPER / "abstract.tex"

def apply(text: str, replacements: list[tuple[str, str]], unique: bool = True) -> str:
    """按顺序逐对替换 old→new。

    Args:
        text: 输入文本
        replacements: (old, new) 对
        unique: 若 True 要求每条 old 唯一命中（默认）；False 允许全局替换
    """
    for i, (old, new) in enumerate(replacements):
        if old not in text:
            raise SystemExit(f"[FATAL] 第 {i} 项 old 未命中:\n{old[:200]}")
        if unique and text.count(old) > 1:
            raise SystemExit(f"[FATAL] 第 {i} 项 old 出现 {text.count(old)} 次:\n{old[:200]}")
        text = text.replace(old, new)
    return text


def polish_manual(text: str) -> str:
    """对 manual.tex 应用全部润色规则。"""
    rules: list[tuple[str, str]] = []

    rules += [
        (r"\section{C1：基于 Claude Opus 4.7 的舆情情感因子}",
         r"\section{基于大语言模型的舆情情感因子}"),
        (r"\section{C2：异构轻量多模型 Stacking 集成框架}",
         r"\section{异构多模型 Stacking 集成框架}"),
        (r"\section{C3：Conformal Prediction 不确定性量化与置信加权组合优化}",
         r"\section{Conformal Prediction 与置信加权组合优化}"),
        (r"\section{C4：Numba JIT + joblib 并行的高性能回测引擎}",
         r"\section{基于 Numba JIT 与 joblib 并行的回测引擎}"),
        (r"\section{C1：LLM 舆情因子可行性验证}",
         r"\section{舆情情感因子可行性验证}"),
        (r"\section{C2：Stacking 集成对比（2024-2025）}",
         r"\section{Stacking 集成对比（2024-2025）}"),
        (r"\section{C3：Conformal Prediction 与置信加权（2024-2025）}",
         r"\section{Conformal Prediction 与置信加权(2024-2025)}"),
        (r"\section{C4：回测引擎加速比基准}",
         r"\section{回测引擎加速比基准}"),
    ]

    rules += [
        (r"\caption{C1 LLM 舆情打分管线成本与信号质量（610 条新闻 / 61 股）}",
         r"\caption{LLM 舆情打分管线成本与信号质量（610 条新闻 / 61 股）}"),
        (r"\caption{C1 LLM 舆情打分极性分布与日均情感时间序列}",
         r"\caption{LLM 舆情打分极性分布与日均情感时间序列}"),
        (r"\caption{C1 LLM 舆情打分按主题分类的极性分布}",
         r"\caption{LLM 舆情打分按主题分类的极性分布}"),
        (r"\caption{C2 异构多模型与 Stacking 集成对照（2024-2025 短样本，等权 Top20）}",
         r"\caption{异构多模型与 Stacking 集成对照（2024-2025 短样本，等权 Top20）}"),
        (r"\caption{C2 异构多模型与 Stacking 集成 IR / Sharpe / 年化超额对比}",
         r"\caption{异构多模型与 Stacking 集成 IR / Sharpe / 年化超额对比}"),
        (r"\caption{C2 Stacking 元学习器各 L1 模型权重的滚动轨迹}",
         r"\caption{Stacking 元学习器各 L1 模型权重的滚动轨迹}"),
        (r"\caption{C2 Stacking 消融实验（合成数据）：去掉 GRU 后 IR 由 4.59 暴跌至 0.39}",
         r"\caption{Stacking 消融实验（合成数据）：去掉 GRU 后 IR 由 4.59 暴跌至 0.39}"),
        (r"\caption{C3 Conformal Prediction 三种置信加权方案对照（2024-2025 约束型 LightGBM）}",
         r"\caption{Conformal Prediction 三种置信加权方案对照（2024-2025 约束型 LightGBM）}"),
        (r"\caption{C3 Conformal Prediction 四种置信加权方案绩效对照}",
         r"\caption{Conformal Prediction 四种置信加权方案绩效对照}"),
        (r"\caption{C3 按月份 Conformal 覆盖率与理论 90\% 参考线}",
         r"\caption{按月份 Conformal 覆盖率与理论 90\% 参考线}"),
        (r'\caption{C3 按置信度分桶的 IR 与命中率（揭示"高置信即过拟合"反向规律）}',
         r'\caption{按置信度分桶的 IR 与命中率（揭示"高置信即过拟合"反向规律）}'),
        (r"\caption{C3 Conformal Prediction 覆盖率对 $\alpha$ 的敏感性（偏差恒定 -9 pp）}",
         r"\caption{Conformal Prediction 覆盖率对 $\alpha$ 的敏感性（偏差恒定 -9 pp）}"),
        (r"\caption{C4 Numba JIT vs 纯 Python 回测引擎性能对照（M-series Mac，single thread）}",
         r"\caption{Numba JIT vs 纯 Python 回测引擎性能对照（M-series Mac，single thread）}"),
        (r"\caption{C4 Kernel 级 Numba JIT vs 纯 Python 嵌套 for-loop 的加速比 surface}",
         r"\caption{Kernel 级 Numba JIT vs 纯 Python 嵌套 for-loop 的加速比 surface}"),
        (r"\caption{C4 Numba JIT vs Python 回测引擎在不同规模上的加速比}",
         r"\caption{Numba JIT vs Python 回测引擎在不同规模上的加速比}"),
    ]

    rules += [
        (r"\caption{四项创新点对应的代码模块清单}",
         r"\caption{核心算法与工程模块清单}"),
        (r"\multirow{4}{*}{C1 LLM 舆情因子}",
         r"\multirow{4}{*}{舆情情感因子}"),
        (r"\multirow{6}{*}{C2 Stacking 集成}",
         r"\multirow{6}{*}{Stacking 集成}"),
        (r"\multirow{5}{*}{C3 Conformal Prediction}",
         r"\multirow{5}{*}{Conformal Prediction}"),
        (r"\multirow{4}{*}{C4 Numba 加速}",
         r"\multirow{4}{*}{Numba 加速}"),
    ]

    old_contrib = (
        r"针对沪深 300 指数增强场景，本文以\emph{算法创新}与\emph{工程加速}为核心，将大语言模型、集成学习、统计学习与高性能计算等计算机科学方向的前沿方法系统性地嵌入经典 Smart Beta 策略框架，具体贡献包括："
        "\n\n"
        r"\begin{enumerate}"
        "\n"
        r"    \item \textbf{C1：基于 Claude Opus 4.7 的中文舆情情感因子}。设计了一条 LLM-as-a-Service 管线（SHA-256 缓存 + 硬性预算守门 + 双通道路由），将 LLM 从昂贵研究工具转为可控成本的日频 alpha 来源；实测单条新闻打分 $5.2\times 10^{-4}$ USD，与传统价值/质量/动量因子语义互补\cite{Tetlock2007Sentiment,Yang2023FinGPT,Wu2023BloombergGPT}（详见 §\ref{sec:sentiment_results}）。"
        "\n\n"
        r"    \item \textbf{C2：异构轻量多模型 Stacking 集成}。在 LightGBM\cite{Ke2017LightGBM} 基础上同步引入 XGBoost\cite{Chen2016XGBoost}、轻量 GRU\cite{Cho2014GRU} 与 Ridge 等异构 L1 模型，通过 OOF KFold + MLP 元学习器自适应融合\cite{Wolpert1992}；全部参数量 $\le$ 百万级、CPU 友好；带约束情境下 IR 5.92 小幅但稳定击败单 LightGBM IR 5.84（详见 §\ref{sec:final_comparison}）。"
        "\n\n"
        r"    \item \textbf{C3：Conformal Prediction 不确定性量化与置信加权 QP}。将 Split / Mondrian CP\cite{Vovk2005Conformal,Romano2019CQR,Angelopoulos2021Gentle} 与本土化二次规划组合优化器耦合，提出三种置信加权方案；实测覆盖率与理论 90\% 偏差恒定 -9 pp（与 $\alpha$ 取值无关），并揭示 A 股低信噪比下 ``高置信即过拟合'' 的反向规律（详见 §\ref{sec:conformal_results} 与 §\ref{subsec:alpha_sensitivity}）。"
        "\n\n"
        r"    \item \textbf{C4：Numba JIT + joblib 并行的高性能回测引擎}。将按日 Python 循环重写为 Numba\cite{Lam2015Numba} JIT 编译的 NAV 累乘核 + joblib 并行的消融框架；Kernel 级加速 130-310$\times$、Engine 级加速 9.84$\times$、数值偏差 $< 10^{-15}$（详见 §\ref{subsec:c4_kernel_vs_engine}），直接呼应课题书所提 ``传统策略计算量大、优化过程耗时长'' 的核心痛点。"
        "\n"
        r"\end{enumerate}"
    )
    new_contrib = (
        r"针对沪深 300 指数增强场景，本文以算法创新与工程加速为双重抓手，将大语言模型、集成学习、统计学习与高性能计算等计算机科学方向的前沿方法系统性地嵌入经典 Smart Beta 策略框架。"
        "\n\n"
        r"在语义信号层，本文设计了一条围绕 Claude Opus 4.7 的 LLM-as-a-Service 管线：以 SHA-256 缓存避免重复打分、以预算上限守门防止成本失控，并通过 Haiku/Opus 双通道按任务难度路由模型选择，将原本昂贵的研究型大模型转化为可控成本的日频 alpha 信号来源。实测单条新闻打分仅需 $5.2\times 10^{-4}$ USD，输出的舆情因子在语义维度上与传统价值、质量、动量因子互补\cite{Tetlock2007Sentiment,Yang2023FinGPT,Wu2023BloombergGPT}，详见第 \ref{sec:sentiment_results} 节。"
        "\n\n"
        r"在模型集成层，本文在 LightGBM\cite{Ke2017LightGBM} 主回归器之外同步引入 XGBoost\cite{Chen2016XGBoost}、轻量 GRU\cite{Cho2014GRU} 与 Ridge 三个异构 L1 模型，通过 KFold 折外预测构造元特征矩阵，再由 MLP 元学习器自适应融合\cite{Wolpert1992}。整个集成框架全部参数控制在百万级以内，训练与推理均可在 CPU 上完成；在带约束的真实回测中信息比率达到 5.92，稳定地小幅领先单 LightGBM 的 5.84（第 \ref{sec:final_comparison} 节）。"
        "\n\n"
        r"在不确定性量化层，本文将 Split 与 Mondrian Conformal Prediction\cite{Vovk2005Conformal,Romano2019CQR,Angelopoulos2021Gentle} 与本土化二次规划组合优化器耦合，提出三种置信加权方案，把每只股票的预测可信度直接传导到 QP 的目标项或可行域中。实测覆盖率与理论 90\% 之间的偏差在不同显著性水平下稳定保持在 -9 个百分点左右，并意外地揭示出 A 股低信噪比环境中 ``高置信即过拟合'' 的反向规律（第 \ref{sec:conformal_results} 与第 \ref{subsec:alpha_sensitivity} 节）。"
        "\n\n"
        r"在工程加速层，本文将按日 Python 循环重写为 Numba\cite{Lam2015Numba} JIT 编译的 NAV 累乘核，并以 joblib 并行驱动消融实验。Kernel 级加速比稳定在 130--310$\times$，端到端 Engine 级加速比 9.84$\times$，数值偏差始终低于 $10^{-15}$（第 \ref{subsec:c4_kernel_vs_engine} 节）。这一加速直接回应了课题书所指出的``传统策略计算量大、优化过程耗时长''核心痛点。"
    )
    rules.append((old_contrib, new_contrib))

    old_struct = (
        r"本文共分六章。第 \ref{sec:contributions} 节（即本章）介绍研究背景、A 股市场非有效性特征与本文四项贡献；第二章梳理主动管理基本定律、机器学习资产定价、Conformal Prediction 与 2025-2026 前沿工作的文献综述；第三章给出 Smart Beta 策略重构与核心因子设计；第四章详述模型选型与组合优化框架，并按 C1（LLM 舆情）/ C2（Stacking 集成）/ C3（Conformal Prediction）/ C4（Numba 加速）四个子节展开新增模块的算法设计，章末给出创新点与代码模块的对应关系；第五章给出长周期（2015-2024）与短周期（2024-2025）的完整实证研究，含五策略最终对比、严格 OOS 验证、覆盖率敏感性、ACI 简化模拟与 Kernel 级加速比 surface 等共 11 个实验子节；第六章总结主要结论、学科贡献与未来工作路线图，章末给出实验复现指南附录。"
    )
    new_struct = (
        r"本文共分六章。第 \ref{sec:contributions} 节介绍研究背景、A 股市场非有效性特征与本文四项主要贡献。第二章系统梳理主动管理基本定律、机器学习资产定价、Conformal Prediction 与 2025-2026 年前沿工作的文献。第三章给出 Smart Beta 策略重构与核心因子的设计。第四章详述模型选型与组合优化框架，依次展开舆情情感因子、Stacking 集成、Conformal Prediction 置信加权以及 Numba 加速回测引擎四个新增模块的算法设计，并在章末给出与代码模块的对应关系。第五章在长周期（2015-2024）与短周期（2024-2025）两条样本上完成完整的实证研究，含五策略最终对比、严格 OOS 验证、覆盖率敏感性、ACI 简化模拟与 Kernel 级加速比 surface 等共 11 个实验子节。第六章总结主要结论、学科贡献与未来工作路线图，并给出实验复现指南。"
    )
    rules.append((old_struct, new_struct))

    rules += [
        (r"本文 C1 舆情因子管线直接采纳该范式。",
         r"本文的舆情因子管线即在此范式下落地。"),
        (r"，与本文 C3 高度相关。",
         r"，与本文的不确定性量化设计高度相关。"),
        (r"本文 C1 退而求其次，将 LLM 作为离线的语义抽取层，把可控成本作为首要工程约束。",
         r"本文则退而求其次，将 LLM 作为离线的语义抽取层，把可控成本作为首要的工程约束。"),
        (r"与本文 C2 的轻量级 Stacking 框架形成参数规模与可解释性的鲜明对比。",
         r"与本文采用的轻量级 Stacking 框架在参数规模与可解释性上形成鲜明对比。"),
        (r"本文 C2 的 MLP 元学习器可视为该路线的轻量级实现，其端到端结构便于 CPU 训练与可解释性。",
         r"本文的 MLP 元学习器可视为这一路线的轻量级实现，其端到端结构便于 CPU 训练与解释。"),
        (r"是本文 C3 的直接竞争对手。",
         r"也是本文 Conformal Prediction 设计的直接对照。"),
        (r"综上，本文 C1-C4 的定位是\emph{覆盖广度与工程可复现性优先}：在四个相互正交的子方向上以最小依赖与可控成本实现完整管线，而非在任一子方向追求 SOTA 算法。第 §\ref{sec:future_work} 节将给出与上述前沿融合的具体路线图。",
         r"综上，本文四项贡献的整体定位是\emph{覆盖广度与工程可复现性优先}：在四个相互正交的子方向上以最小依赖与可控成本实现完整可复现管线，而非在任一子方向追求 SOTA 算法。第 \ref{sec:future_work} 节将给出与上述前沿融合的具体路线图。"),
        (r"该层提供本文所有后续创新点（C1-C3）所依赖的 baseline 预测信号。",
         r"该层为后续的舆情因子、Stacking 集成与 Conformal Prediction 模块提供统一的 baseline 预测信号。"),
        (r"长周期实验对应 §\ref{sec:contributions} 中 C2/C3/C4 的稳健性验证，",
         r"长周期实验主要用于验证 Stacking 集成、Conformal Prediction 与 Numba 加速三项贡献的稳健性，"),
        (r"为对接 C1（LLM 舆情，2025 年新闻数据成本最低）与 C2/C3 的端到端流程验证，",
         r"为兼顾 LLM 舆情打分的成本可控性（2025 年新闻数据成本最低）以及 Stacking、Conformal 模块的端到端流程验证，"),
        (r"本节在 61 只沪深 300 成分股、2026-03-30 至 2026-05-24 区间共 610 条新闻上验证 C1 管线。打分全程走 \texttt{claude-haiku-4-5} 快速通道，预算守门 10 美元。",
         r"本节在 61 只沪深 300 成分股、2026-03-30 至 2026-05-24 区间共 610 条新闻上验证舆情因子打分管线。全部调用经由 \texttt{claude-haiku-4-5} 快速通道，并在 10 美元预算上限内运行。"),
        (r"这印证了本文 C3 的核心实证发现的稳健性，",
         r"这印证了本章 Conformal Prediction 反向规律这一核心实证发现的稳健性，"),
        (r"留作 §\ref{sec:future_work} 中的 C3 升级方向。",
         r"留作 §\ref{sec:future_work} 节中 Conformal Prediction 模块的升级方向。"),
        (r"这也是 C4 后续工程优化的明确方向。",
         r"这也是后续工程优化的明确方向。"),
        (r"& Stacking IR 5.92 > LightGBM IR 5.84 & C2 异构集成在带约束情境下小幅但稳定击败单模型（§\ref{sec:final_comparison}） \\",
         r"& Stacking IR 5.92 > LightGBM IR 5.84 & 异构集成在带约束情境下小幅但稳定击败单模型（§\ref{sec:final_comparison}） \\"),
        (r"\textbf{综合结论}：本文四项创新点 C1-C4 均经过\emph{多维度敏感性 / 消融测试}的稳健性验证。最重要的稳健性证据集中体现于：(i) Stacking 在带优化器的真实回测中持续小幅击败单 LightGBM（C2 实证支撑），(ii) Conformal Prediction 反向规律在 $\alpha$ 敏感性与 Mondrian 真实数据复测下均稳健（C3 方法学发现稳健性），(iii) Numba JIT 在 5 个量级规模上加速比稳定在 130-310$\times$（C4 工程加速可推广性）。这些证据共同支撑了 §\ref{sec:future_work} 路线图的实施依据。",
         r"\textbf{综合结论}：本文四项贡献均经过多维度敏感性与消融测试的稳健性验证。其中三类证据最为关键：第一，Stacking 集成在带优化器的真实回测中持续小幅击败单 LightGBM，为集成框架的实证价值提供了直接支撑；第二，Conformal Prediction 反向规律在 $\alpha$ 敏感性扫描与 Mondrian 真实数据复测下保持稳健，证明这一方法学发现并非个别参数下的偶然结果；第三，Numba JIT 在跨越 5 个量级的规模上加速比始终稳定在 130--310$\times$，说明工程加速具有良好的可推广性。这三类证据共同支撑了 §\ref{sec:future_work} 节路线图的实施依据。"),
        (r"C3 章节给出的 Split / Mondrian CP",
         r"Conformal Prediction 章节给出的 Split / Mondrian CP"),
    ]

    old_conclusion = (
        r"本文以沪深 300 指数增强为应用场景，从计算机算法与工程加速两个维度提出并实证了四项创新点："
        "\n"
        r"\begin{enumerate}"
        "\n"
        r"    \item \textbf{C1（LLM 舆情因子）}：通过 Claude Opus 4.7 / Haiku 4.5 双通道、SHA-256 缓存与硬性预算守门，将 LLM 调用从昂贵的离线服务转化为可控成本的日频 alpha 来源；端到端单次实验成本可压缩到 10 美元以内。"
        "\n"
        r"    \item \textbf{C2（异构 Stacking 集成）}：LightGBM + XGBoost + GRU + Ridge 的异构 L1 模型族通过 OOF KFold 与 MLP 元学习器自适应融合；在 2024-2025 短样本上，元学习器自动给同质化最强的 Ridge 模型赋予负向校正权重，体现了 Stacking 在低信噪比金融数据中的自动模型选择能力。"
        "\n"
        r"    \item \textbf{C3（Conformal Prediction 置信加权）}：Split / Mondrian CP 与本土化 QP 优化器耦合，三种置信加权方案在短样本上的实测覆盖率与理论 90\% 的偏差控制在 10 个百分点内。本文同时揭示了 A 股低信噪比环境下\emph{高置信度即过拟合}的反向规律，为后续不确定性感知型组合提供了实证依据。"
        "\n"
        r"    \item \textbf{C4（Numba JIT 加速）}：将回测核心循环改写为 Numba JIT，实测十年沪深 300 面板上加速 9.84$\times$ 且数值偏差小于 $10^{-15}$；joblib 并行进一步将消融实验加速 1.48$\times$。该工程加速直接呼应了课题书``传统策略计算量大、优化过程耗时长''的核心痛点。"
        "\n"
        r"\end{enumerate}"
    )
    new_conclusion = (
        r"本文以沪深 300 指数增强为应用场景，从计算机算法与工程加速两个维度提出并实证了四项贡献。"
        "\n\n"
        r"在舆情信号方面，借助 Claude Opus 4.7 与 Haiku 4.5 双通道、SHA-256 缓存与预算上限守门，将原本昂贵的 LLM 调用转化为可控成本的日频 alpha 来源，整条管线的端到端单次实验成本可被压缩到 10 美元以内。"
        "\n\n"
        r"在模型集成方面，LightGBM、XGBoost、轻量 GRU 与 Ridge 四个异构 L1 模型通过 KFold 折外预测与 MLP 元学习器自适应融合；2024-2025 短样本上，元学习器自动给同质化最强的 Ridge 赋予负向校正权重，展现了 Stacking 在低信噪比金融数据中自动甄别冗余模型的能力。"
        "\n\n"
        r"在不确定性量化方面，将 Split 与 Mondrian Conformal Prediction 与本土化 QP 优化器耦合，三种置信加权方案的实测覆盖率与理论 90\% 的偏差在短样本上稳定地控制在 10 个百分点之内；同时揭示了 A 股低信噪比环境下\emph{高置信即过拟合}这一反向规律，为后续不确定性感知型组合提供了直接的实证依据。"
        "\n\n"
        r"在工程加速方面，回测核心循环被改写为 Numba JIT 编译版本，十年沪深 300 面板上加速 9.84$\times$ 且数值偏差小于 $10^{-15}$；joblib 并行进一步把消融实验加速 1.48$\times$。这一加速直接回应了课题书所提的``传统策略计算量大、优化过程耗时长''核心痛点。"
    )
    rules.append((old_conclusion, new_conclusion))

    rules += [
        (r"    \item \textbf{数据扩展（C1）}：sentiment 因子真实数据覆盖率仅 0.32\%，下一步引入巨潮资讯公告与 Wind 研报扩大语料，并参照 StockBench\cite{Wang2026StockBench} 的 contamination-free 设计避免回测污染。",
         r"    \item \textbf{舆情语料扩展}：sentiment 因子的真实数据覆盖率目前仅 0.32\%，下一步将引入巨潮资讯公告与 Wind 研报扩大语料，并参照 StockBench\cite{Wang2026StockBench} 的 contamination-free 设计避免回测污染。"),
        (r"    \item \textbf{C2 升级为 MoE 路由}：将 MLP 元学习器升级为 MIGA\cite{Zhou2024MIGA} 风格分组 MoE 或 LLMoE\cite{Li2025LLMoE} 的 LLM-as-Router，并在国金金工 TSGRU+LGBM\cite{Guojin2026TSGRU} 框架内做对照。",
         r"    \item \textbf{集成框架升级为 MoE 路由}：将 MLP 元学习器替换为 MIGA\cite{Zhou2024MIGA} 风格的分组 MoE 或 LLMoE\cite{Li2025LLMoE} 的 LLM-as-Router，并与国金金工 TSGRU+LGBM\cite{Guojin2026TSGRU} 框架做横向对照。"),
        (r"    \item \textbf{C3 升级为自适应 CP}：尝试 TCP\cite{Aydin2025TCP}、ECI\cite{Yang2025ECI}（ICLR 2025）、CPTC\cite{Sun2025CPTC}（NeurIPS 2025）与 ResCP\cite{Neglia2026ResCP} 等具有 change-point 感知与在线衰减学习率的变体，矫正 §\ref{subsec:alpha_sensitivity} 揭示的 -9 pp 系统性偏差。",
         r"    \item \textbf{自适应 Conformal Prediction}：进一步尝试 TCP\cite{Aydin2025TCP}、ECI\cite{Yang2025ECI}（ICLR 2025）、CPTC\cite{Sun2025CPTC}（NeurIPS 2025）与 ResCP\cite{Neglia2026ResCP} 等具备 change-point 感知与在线衰减学习率的变体，以矫正 §\ref{subsec:alpha_sensitivity} 节揭示的 -9 pp 系统性偏差。"),
        (r"    \item \textbf{C4 跨节点扩展}：基于 Dask / Ray 做跨节点参数搜索；调用 LightGBM 的 GPU 后端验证学术与工程 trade-off。",
         r"    \item \textbf{加速引擎跨节点扩展}：基于 Dask / Ray 做跨节点参数搜索，并调用 LightGBM 的 GPU 后端进一步评估学术与工程之间的 trade-off。"),
    ]

    rules += [
        (r"    \item \textbf{C1 LLM 舆情}：\texttt{python -m src.pipelines.build\_sentiment\_panel --start 2025-01-01 --end 2025-05-30 --output processed/sentiment\_panel.csv --max-codes 60 --max-usd 8}",
         r"    \item \textbf{舆情情感因子}：\texttt{python -m src.pipelines.build\_sentiment\_panel --start 2025-01-01 --end 2025-05-30 --output processed/sentiment\_panel.csv --max-codes 60 --max-usd 8}"),
        (r"    \item \textbf{C2 Stacking}：\texttt{python -m src.pipelines.run\_stacking\_experiment --input processed/hs300\_panel\_2024\_2025\_v2.csv --variants lgbm xgb ridge stacking --use-external-features}",
         r"    \item \textbf{Stacking 集成}：\texttt{python -m src.pipelines.run\_stacking\_experiment --input processed/hs300\_panel\_2024\_2025\_v2.csv --variants lgbm xgb ridge stacking --use-external-features}"),
        (r"    \item \textbf{C3 Conformal}：\texttt{python -m src.pipelines.run\_conformal\_experiment --input processed/hs300\_panel\_2024\_2025\_v2.csv --schemes baseline alpha\_scale candidate\_filter objective\_penalty --use-external-features}",
         r"    \item \textbf{Conformal Prediction}：\texttt{python -m src.pipelines.run\_conformal\_experiment --input processed/hs300\_panel\_2024\_2025\_v2.csv --schemes baseline alpha\_scale candidate\_filter objective\_penalty --use-external-features}"),
        (r"    \item \textbf{C4 Numba}：\texttt{python -m src.pipelines.benchmark\_engine}",
         r"    \item \textbf{Numba 加速}：\texttt{python -m src.pipelines.benchmark\_engine}"),
        (r"完整 C1+C2+C3+C4 实验耗时约：",
         r"四项创新点的完整端到端实验耗时约："),
    ]

    text = apply(text, rules)

    text = apply(text, [(r"折衷", r"折中")], unique=False)
    return text


def polish_abstract(text: str) -> str:
    """对 abstract.tex 做散文化润色。"""
    old_zh = r'''本文以沪深 300 指数增强为应用场景，针对传统量化策略\emph{计算耗时长、因子来源单一、不确定性量化缺失}三类核心问题，提出并实证了一套以\emph{算法创新}与\emph{工程加速}为核心的指数增强框架。具体四项创新：(1) 设计了基于 Claude Opus 4.7 的中文财经舆情情感因子管线，通过结构化 prompt、SHA-256 缓存与硬性预算守门，将大语言模型从昂贵的离线服务转化为可控成本的日频 alpha 来源，单条新闻打分实测成本 $5.2\times 10^{-4}$ 美元；(2) 实现了 LightGBM、XGBoost、轻量 GRU 与 Ridge 异构 L1 模型 + MLP 元学习器的 Stacking 集成框架，所有模型参数量在百万级以内、全程 CPU 友好；(3) 将 Split / Mondrian Conformal Prediction 与本土化二次规划组合优化器耦合，提出三种置信加权方案，实测覆盖率与理论 90\% 的偏差约 9 个百分点，并揭示了 A 股低信噪比环境下"高置信度即过拟合"的反向规律；(4) 将回测核心循环改写为 Numba JIT 编译，并以 joblib 并行化消融实验，实测十年沪深 300 面板上回测加速 9.84$\times$、数值偏差小于 $10^{-15}$。本文系统对照了 2025-2026 年同主题前沿工作（StockBench、QuantAgent、MIGA、Diffusion Factor Model、Temporal CP 等），并明确定位本研究为"覆盖广度与工程可复现性优先"的开放实现。完整实验在沪深 300 成分股上验证：约束型 LightGBM 基线信息比率（IR）由静态多因子的 0.237 提升至 2.449，最大回撤由 -58.53\% 收敛至 -25.37\%。本研究的代码、数据面板与实验结果均已开源至 GitHub 仓库：\url{https://github.com/aokimi0/IndexEnhancementStrategy}。'''
    new_zh = r'''本文以沪深 300 指数增强为应用场景，针对传统量化策略\emph{计算耗时长、因子来源单一、不确定性量化缺失}三类核心问题，提出并实证了一套以算法创新与工程加速为双轮驱动的指数增强框架。在舆情信号层面，设计了一条围绕 Claude Opus 4.7 的中文财经新闻情感因子管线，通过结构化 prompt、SHA-256 缓存与预算上限守门，将原本昂贵的大语言模型调用转化为可控成本的日频 alpha 来源，单条新闻打分的实测成本仅约 $5.2\times 10^{-4}$ 美元。在模型集成层面，实现了 LightGBM、XGBoost、轻量 GRU 与 Ridge 四个异构 L1 模型加 MLP 元学习器的 Stacking 集成框架，所有模型参数控制在百万级以内、训练与推理全程 CPU 友好。在不确定性量化层面，将 Split 与 Mondrian Conformal Prediction 与本土化二次规划组合优化器耦合，提出三种置信加权方案；实测覆盖率与理论 90\% 之间的偏差约 9 个百分点，并意外揭示了 A 股低信噪比环境下"高置信即过拟合"的反向规律。在工程加速层面，将回测核心循环改写为 Numba JIT 编译版本，并以 joblib 并行驱动消融实验，十年沪深 300 面板的端到端回测加速 9.84$\times$，数值偏差小于 $10^{-15}$。本文同时系统对照了 2025-2026 年同主题前沿工作（StockBench、QuantAgent、MIGA、Diffusion Factor Model、Temporal CP 等），明确将本研究定位为"覆盖广度与工程可复现性优先"的开放实现。完整实证在沪深 300 成分股上验证：约束型 LightGBM 基线信息比率（IR）由静态多因子的 0.237 提升至 2.449，最大回撤由 -58.53\% 收敛至 -25.37\%。本研究的代码、数据面板与实验结果均已开源至 GitHub 仓库：\url{https://github.com/aokimi0/IndexEnhancementStrategy}。'''

    old_en = r'''This paper proposes and empirically validates an index enhancement framework for the CSI 300 index, targeting three core problems of traditional quantitative strategies: long computation time, monotonous factor sources, and lack of uncertainty quantification. The framework centers on \emph{algorithmic innovation} and \emph{engineering acceleration} with four key contributions: (1) A Chinese financial sentiment factor pipeline based on Claude Opus 4.7 is designed, which transforms expensive offline LLM services into a controllable-cost daily alpha source through structured prompts, SHA-256 caching, and hard budget gating, with a measured cost of $5.2\times 10^{-4}$ USD per news item; (2) A Stacking ensemble framework integrating heterogeneous L1 models (LightGBM, XGBoost, lightweight GRU, Ridge) with an MLP meta-learner is implemented, where all models stay within million-level parameters and remain CPU-friendly throughout; (3) Split / Mondrian Conformal Prediction is coupled with a localized quadratic-programming portfolio optimizer, with three confidence-weighting schemes proposed; the measured empirical coverage deviates from the theoretical 90\% by about 9 percentage points, and reveals an inverse pattern of ``higher confidence implies over-fitting'' in the low signal-to-noise A-share market; (4) The backtest engine's daily loop is rewritten with Numba JIT and parallelized via joblib for ablation experiments, achieving a 9.84$\times$ speedup on the ten-year CSI 300 panel with numerical deviation below $10^{-15}$. This paper systematically positions itself against 2025-2026 frontier work (StockBench, QuantAgent, MIGA, Diffusion Factor Models, Temporal CP, etc.) and explicitly emphasizes ``coverage breadth and engineering reproducibility'' over single-direction SOTA. Comprehensive experiments on CSI 300 constituents show that the constrained LightGBM baseline lifts the information ratio (IR) from 0.237 (static multi-factor) to 2.449, with maximum drawdown shrinking from -58.53\% to -25.37\%. The code, data panels, and experimental results are open-sourced at the GitHub repository: \url{https://github.com/aokimi0/IndexEnhancementStrategy}.'''
    new_en = r'''This paper proposes and empirically validates an index enhancement framework for the CSI 300 index, targeting three core problems of traditional quantitative strategies: long computation time, monotonous factor sources, and missing uncertainty quantification. The framework is driven jointly by algorithmic innovation and engineering acceleration. At the semantic-signal layer, we design a Chinese financial news sentiment pipeline built on Claude Opus 4.7, in which structured prompts, SHA-256 caching, and a hard budget guard turn an otherwise expensive large language model into a cost-controllable source of daily alpha at roughly $5.2\times 10^{-4}$ USD per news item. At the model-ensemble layer, we implement a Stacking framework that fuses heterogeneous L1 learners (LightGBM, XGBoost, a lightweight GRU, and Ridge) through an MLP meta-learner, keeping every component within million-level parameters and CPU-friendly throughout. At the uncertainty-quantification layer, we couple Split / Mondrian Conformal Prediction with a localized quadratic-programming portfolio optimizer and propose three confidence-weighting schemes; the empirical coverage deviates from the nominal 90\% by roughly 9 percentage points, and an inverse ``higher confidence implies over-fitting'' pattern is uncovered in the low signal-to-noise A-share market. At the engineering layer, the daily loop of the backtest engine is rewritten with Numba JIT and parallelized via joblib for ablation, delivering a 9.84$\times$ end-to-end speedup on the ten-year CSI 300 panel with numerical deviation below $10^{-15}$. This paper further situates itself against 2025-2026 frontier works (StockBench, QuantAgent, MIGA, Diffusion Factor Models, Temporal CP, and so on), explicitly positioning the study as a ``coverage-first and reproducibility-first'' open implementation rather than a single-direction SOTA pursuit. Comprehensive experiments on CSI 300 constituents show that the constrained LightGBM baseline lifts the information ratio (IR) from 0.237 (static multi-factor) to 2.449, with the maximum drawdown shrinking from -58.53\% to -25.37\%. The code, data panels, and experimental results are all open-sourced at the GitHub repository: \url{https://github.com/aokimi0/IndexEnhancementStrategy}.'''

    return apply(text, [(old_zh, new_zh), (old_en, new_en)])


def main() -> None:
    """串行执行 manual 与 abstract 的润色。"""
    manual = MANUAL.read_text(encoding="utf-8")
    abstract = ABSTRACT.read_text(encoding="utf-8")

    manual2 = polish_manual(manual)
    abstract2 = polish_abstract(abstract)

    MANUAL.write_text(manual2, encoding="utf-8")
    ABSTRACT.write_text(abstract2, encoding="utf-8")

    print(f"[OK] manual.tex: {len(manual)} -> {len(manual2)} chars")
    print(f"[OK] abstract.tex: {len(abstract)} -> {len(abstract2)} chars")


if __name__ == "__main__":
    main()
