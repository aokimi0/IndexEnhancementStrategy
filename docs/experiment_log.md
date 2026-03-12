# 实验记录

## 1. 记录目的

本文档用于汇总当前阶段已经完成的实验、关键参数、结果指标与阶段性结论。
当前记录的重点是验证研究链路是否跑通，并形成后续论文中“基线策略”与“AI 增强策略”的实验起点。

## 2. 当前实验环境

### 2.1 数据源

- 主数据源：`akshare + baostock`
- 说明：当前采用 `akshare` 获取指数成分与行情数据，采用 `baostock` 补充日频估值与季度财务指标，避免 `Tushare Pro` 权限受限带来的基本面缺口

### 2.2 运行环境

- 环境管理：`conda`
- Python 版本：`3.12`
- 主要依赖：
  - `pandas`
  - `numpy`
  - `akshare`
  - `baostock`
  - `lightgbm`
  - `scikit-learn`

### 2.3 当前数据特点

- 价格类字段已可稳定获取
- 动量、波动率等技术类因子已可构造
- 日频估值字段已补齐核心口径，如 `turnover_rate`、`pe_ttm`、`pb`、`ps_ttm`
- 季度财务字段已补齐核心质量与成长口径，如 `roe`、`netprofitmargin`、`yoynetprofit`、`assetturnover`、`cfotoor`
- 部分行业的毛利率字段仍存在缺失，主要集中于银行等金融类公司
- 个股日频数据已经加入缓存、失败跳过与失败日志机制

## 3. 已完成实验

## 3.1 小样本因子面板构建实验

### 目的

验证“数据读取 -> 因子构建 -> 超额收益标签生成”这条最小研究链路是否可运行。

### 设置

- 股票池：沪深300成分股中选取 10 只样本股
- 时间区间：`2024-01-01` 到 `2024-03-31`
- 数据源：`akshare`
- 输出文件：`data/processed/hs300_factor_panel_sample.csv`

### 结果

- 输出面板规模：`580 x 24`
- 已成功生成字段：
  - 行情字段：`open`、`high`、`low`、`close`、`vol`
  - 技术因子：`daily_return`、`ret_20`、`ret_60`、`volatility_20`
  - 标签字段：`future_return_20d`、`benchmark_future_return_20d`、`label_excess_return_20d`

### 结论

- 研究链路已经从文档阶段进入可运行阶段
- 面板结构已经足够支撑回测与后续机器学习实验

## 3.2 小样本多因子基线回测实验

### 目的

构建第一版可复现的多因子基线策略，用作后续机器学习增强实验的对照组。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_sample.csv`
- 调仓频率：月度
- 选股方式：截面因子等权打分
- 持仓数量：前 5 只股票
- 权重方式：等权

### 输出文件

- `data/processed/baseline_nav_sample.csv`
- `data/processed/baseline_positions_sample.csv`
- `data/processed/baseline_metrics_sample.csv`

### 指标结果

- `annual_return`: `0.4616`
- `benchmark_annual_return`: `0.6067`
- `annual_excess_return`: `-0.1451`
- `annual_volatility`: `0.1637`
- `tracking_error`: `0.1294`
- `information_ratio`: `-1.1209`
- `max_drawdown`: `-0.0463`

### 结论

- 小样本基线没有跑赢基准
- 但该实验成功验证了回测模块、持仓生成和指标输出的正确性

## 3.3 扩展版因子面板构建实验

### 目的

将研究范围从小样本扩展到接近沪深300全成分股，并覆盖更长时间区间，形成后续正式实验输入。

### 设置

- 股票池：沪深300成分股
- 时间区间：`2023-01-03` 到 `2024-12-31`
- 数据源：`akshare`
- 输出文件：`data/processed/hs300_factor_panel_extended_2023_2024.csv`

### 结果

- 面板规模：`143384 x 24`
- 股票覆盖数：`298`
- 覆盖区间：`2023-01-03` 到 `2024-12-31`
- 当前未覆盖股票：`302132`、`600930.SH`

### 结论

- 扩展版面板已经达到“可用于正式策略实验”的规模
- 当前面板可直接用于基线与机器学习增强实验

## 3.4 扩展版多因子基线回测实验

### 目的

在接近全成分股和近两年时间窗口上运行基线策略，形成更有意义的正式对照组。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_extended_2023_2024.csv`
- 调仓频率：月度
- 持仓数量：前 20 只股票
- 权重方式：等权

### 输出文件

- `data/processed/baseline_nav_extended_2023_2024.csv`
- `data/processed/baseline_positions_extended_2023_2024.csv`
- `data/processed/baseline_metrics_extended_2023_2024.csv`

### 指标结果

- `annual_return`: `0.1994`
- `benchmark_annual_return`: `0.1071`
- `annual_excess_return`: `0.0923`
- `annual_volatility`: `0.2118`
- `sharpe_ratio`: `0.9416`
- `tracking_error`: `0.1567`
- `information_ratio`: `0.5889`
- `max_drawdown`: `-0.1818`

### 结论

- 基线策略在扩展版样本上取得了正的年化超额收益
- 已可作为论文中的正式基线策略

## 3.5 LightGBM 增强实验

### 目的

使用机器学习模型预测个股未来超额收益，并与基线策略进行可比对照。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_extended_2023_2024.csv`
- 模型：`LightGBM Regressor`
- 特征：
  - `ret_20`
  - `ret_60`
  - `volatility_20`
- 标签：`label_excess_return_20d`
- 滚动训练窗口：12 个月
- 最小训练样本数：1500
- 回测持仓数量：前 20 只股票

### 输出文件

- `data/processed/lightgbm_predictions_extended_2023_2024.csv`
- `data/processed/lightgbm_feature_importance_extended_2023_2024.csv`
- `data/processed/lightgbm_nav_extended_2023_2024.csv`
- `data/processed/lightgbm_positions_extended_2023_2024.csv`
- `data/processed/lightgbm_metrics_extended_2023_2024.csv`
- `data/processed/strategy_comparison_extended_2023_2024.csv`

### 指标结果

- `annual_return`: `0.1990`
- `benchmark_annual_return`: `0.1071`
- `annual_excess_return`: `0.0918`
- `annual_volatility`: `0.2235`
- `sharpe_ratio`: `0.8900`
- `tracking_error`: `0.1455`
- `information_ratio`: `0.6313`
- `max_drawdown`: `-0.1591`

### 特征重要性均值

- `ret_60`: `5064.25`
- `volatility_20`: `2777.50`
- `ret_20`: `1158.25`

### 结论

- `LightGBM` 相比基线策略，年化收益和年化超额收益基本接近
- `LightGBM` 的信息比率更高
- `LightGBM` 的最大回撤更小
- 当前模型最依赖的是中期动量与波动率因子

## 3.6 十年区间扩展版基线回测实验

### 目的

验证在更接近论文正式实验要求的长周期区间内，扩展后的 `akshare + baostock` 数据链路是否稳定，并观察基线策略的长期表现。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_extended_2015_2024.csv`
- 时间区间：`2015-01-01` 到 `2024-12-31`
- 股票池：沪深300成分股研究池
- 数据源：`akshare + baostock`
- 调仓频率：月度
- 持仓数量：前 20 只股票
- 权重方式：等权

### 输出文件

- `data/processed/hs300_factor_panel_extended_2015_2024.csv`
- `data/processed/baseline_nav_extended_2015_2024.csv`
- `data/processed/baseline_positions_extended_2015_2024.csv`
- `data/processed/baseline_metrics_extended_2015_2024.csv`

### 结果

- 面板规模：`618056 x 35`
- `annual_return`: `0.1871`
- `benchmark_annual_return`: `0.1184`
- `annual_excess_return`: `0.0687`
- `annual_volatility`: `0.2890`
- `sharpe_ratio`: `0.6473`
- `tracking_error`: `0.1824`
- `information_ratio`: `0.3767`
- `max_drawdown`: `-0.6424`

### 结论

- 扩展后的免费数据链路已经能够支撑十年级别的研究面板与基线回测
- 基线策略在长周期样本上仍保持正的年化超额收益，但波动和最大回撤明显偏高
- 下一阶段应优先引入组合约束、风险控制和更完整的财务因子，以改善长期回撤表现

## 3.7 十年区间约束型基线回测实验

### 目的

验证在长周期样本中引入跟踪误差、行业偏离、个股权重上限和换手率约束后，基线策略是否更符合“指数增强”定位。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_constrained_fast_extended_2015_2024.csv`
- 时间区间：`2015-01-01` 到 `2024-12-31`
- 股票池：沪深300研究股票池
- 数据源：`akshare + baostock`
- 调仓频率：月度
- 组合构建：约束型优化组合
- 约束参数：
  - 年化跟踪误差上限：`8%`
  - 行业偏离上限：`2%`
  - 单只个股权重上限：`5%`
  - 月度单边换手上限：`20%`

### 输出文件

- `data/processed/baseline_nav_constrained_extended_2015_2024_v2.csv`
- `data/processed/baseline_positions_constrained_extended_2015_2024_v2.csv`
- `data/processed/baseline_metrics_constrained_extended_2015_2024_v2.csv`

### 指标结果

- `annual_return`: `0.0502`
- `benchmark_annual_return`: `0.0081`
- `annual_excess_return`: `0.0421`
- `annual_volatility`: `0.2984`
- `sharpe_ratio`: `0.1682`
- `tracking_error`: `0.1777`
- `information_ratio`: `0.2370`
- `max_drawdown`: `-0.5853`
- `benchmark_max_drawdown`: `-0.4670`
- `excess_max_drawdown`: `-0.3368`
- `monthly_win_rate`: `0.5583`
- `upside_capture`: `3.9668`
- `downside_capture`: `1.0000`
- `annual_turnover`: `0.1249`
- `avg_ex_ante_tracking_error`: `0.1894`
- `max_ex_ante_tracking_error`: `0.4663`
- `avg_max_industry_deviation`: `0.4061`

### 结论

- 约束型基线显著降低了绝对收益水平，但保留了正的年化超额收益
- 相比无约束版本，最大回撤有所收敛，但改善幅度有限
- 从指数增强视角看，绝对回撤仍需结合 `benchmark_max_drawdown` 与 `excess_max_drawdown` 一起解释
- 当前约束型基线的跟踪误差与行业偏离控制仍偏松，说明第一版优化器还需要进一步校准

## 3.8 十年区间约束型 LightGBM 增强实验

### 目的

验证在相同约束组合框架下，`LightGBM` 对多因子静态打分是否存在显著增益。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_constrained_fast_extended_2015_2024.csv`
- 模型：`LightGBM Regressor`
- 特征：
  - `ep_ttm`
  - `bp`
  - `roe`
  - `grossprofitmargin`
  - `netprofitmargin`
  - `yoynetprofit`
  - `assetturnover`
  - `cfotoor`
  - `ret_20`
  - `ret_60`
  - `volatility_20`
  - `turnover_20`
- 标签：`label_excess_return_20d`
- 滚动训练窗口：`12` 个月
- 最小训练样本数：`1500`
- 回测持仓数量：前 `20` 只
- 组合构建：与约束型基线相同的优化器与风险约束

### 输出文件

- `data/processed/lightgbm_predictions_constrained_extended_2015_2024_v2.csv`
- `data/processed/lightgbm_feature_importance_constrained_extended_2015_2024_v2.csv`
- `data/processed/lightgbm_nav_constrained_extended_2015_2024_v3.csv`
- `data/processed/lightgbm_positions_constrained_extended_2015_2024_v3.csv`
- `data/processed/lightgbm_metrics_constrained_extended_2015_2024_v3.csv`
- `data/processed/strategy_comparison_constrained_extended_2015_2024_v3.csv`

### 指标结果

- `annual_return`: `0.2887`
- `benchmark_annual_return`: `0.0081`
- `annual_excess_return`: `0.2807`
- `annual_volatility`: `0.2240`
- `sharpe_ratio`: `1.2891`
- `tracking_error`: `0.1666`
- `information_ratio`: `1.6852`
- `max_drawdown`: `-0.3171`
- `benchmark_max_drawdown`: `-0.4670`
- `excess_max_drawdown`: `-0.3813`
- `monthly_win_rate`: `0.7000`
- `upside_capture`: `0.8291`
- `downside_capture`: `0.9999`
- `annual_turnover`: `2.0828`
- `avg_ex_ante_tracking_error`: `0.0748`
- `max_ex_ante_tracking_error`: `0.1148`
- `avg_max_industry_deviation`: `0.0204`

### 特征重要性均值

- `volatility_20`: `2529.88`
- `ret_60`: `2446.87`
- `ret_20`: `2039.81`
- `bp`: `725.67`
- `ep_ttm`: `668.61`
- `turnover_20`: `589.17`

### 结论

- 在相同约束框架下，`LightGBM` 相比约束型基线取得了更高的年化收益、年化超额收益和信息比率
- `LightGBM` 的最大回撤显著低于约束型基线，也低于基准自身的最大回撤
- 当前模型最主要依赖的仍然是技术面与估值面特征，质量类财务因子贡献较弱
- 从结果看，机器学习增强比单纯加入组合约束更能改善长期风险收益比

## 3.9 十年区间基准口径校验实验

### 目的

校验十年区间实验中沪深300基准收益的计算口径，确认早期结果与约束版本结果之间的差异来源于策略本身还是基准构造方式。

### 校验方法

- 直接使用 `akshare` 的沪深300指数收盘价序列计算真实基准净值与年化收益
- 对比早期十年基线回测中的 `benchmark_nav`
- 对无约束十年基线面板补入真实 `benchmark_daily_return` 后重新计算指标

### 结果

- 基于沪深300真实指数收盘价计算得到的十年基准年化收益约为 `0.0081`
- 约束型基线与约束型 `LightGBM` 回测中使用的基准收益口径与真实指数口径一致
- 早期无约束十年基线结果中的 `benchmark_annual_return = 0.1184` 为错误口径
- 错误原因在于早期版本使用了“截面平均个股收益”近似基准日收益，而非直接使用沪深300指数日收益
- 修正基准口径后，无约束十年基线结果更新为：
  - `annual_return`: `0.1871`
  - `benchmark_annual_return`: `0.0081`
  - `annual_excess_return`: `0.1790`
  - `tracking_error`: `0.1972`
  - `information_ratio`: `0.9078`
  - `max_drawdown`: `-0.6424`
  - `benchmark_max_drawdown`: `-0.4670`
  - `excess_max_drawdown`: `-0.4636`

### 结论

- 当前约束版本实验使用的基准收益序列是可信的
- 早期十年无约束基线结果中的基准收益指标需要以本次校验后的口径为准
- 从论文写作角度，应统一使用真实沪深300指数日收益构造基准净值，避免混用“个股截面平均收益”近似口径

## 3.10 十年区间约束消融实验

### 目的

验证在约束型 `LightGBM` 增强策略中，不同约束条件对收益、回撤和风险暴露的实际影响，明确当前策略改进究竟来自机器学习信号本身，还是来自特定约束配置。

### 设置

以十年区间约束型 `LightGBM` 策略为完整版本，对组合优化中的约束项逐一做消融：

- `full_constraints`：完整约束版本
- `no_tracking_error`：去掉跟踪误差约束
- `no_industry_constraint`：去掉行业偏离约束
- `no_turnover_constraint`：去掉换手率约束

除被移除的约束项外，其余参数保持一致。

### 输出文件

- `data/processed/constraint_ablation_lightgbm_2015_2024.csv`
- `data/processed/lightgbm_metrics_ablation_no_te_2015_2024.csv`
- `data/processed/lightgbm_metrics_ablation_no_industry_2015_2024.csv`
- `data/processed/lightgbm_metrics_ablation_no_turnover_2015_2024.csv`

### 结果

完整约束版本：

- `annual_return`: `0.2887`
- `annual_excess_return`: `0.2807`
- `sharpe_ratio`: `1.2891`
- `information_ratio`: `1.6852`
- `max_drawdown`: `-0.3171`
- `annual_turnover`: `2.0828`

去掉跟踪误差约束：

- `annual_return`: `0.2678`
- `annual_excess_return`: `0.2597`
- `sharpe_ratio`: `1.1413`
- `information_ratio`: `1.4718`
- `max_drawdown`: `-0.3362`
- `annual_turnover`: `2.0832`

去掉行业偏离约束：

- `annual_return`: `0.3071`
- `annual_excess_return`: `0.2990`
- `sharpe_ratio`: `1.3937`
- `information_ratio`: `1.7597`
- `max_drawdown`: `-0.3247`
- `annual_turnover`: `2.0822`

去掉换手率约束：

- `annual_return`: `0.3991`
- `annual_excess_return`: `0.3911`
- `sharpe_ratio`: `1.7908`
- `information_ratio`: `2.3337`
- `max_drawdown`: `-0.2983`
- `annual_turnover`: `8.1435`

### 结论

- 去掉跟踪误差约束后，收益、夏普比率和信息比率反而下降，说明当前完整版本中的跟踪误差约束并未显著抑制策略收益，反而有助于稳定风险暴露
- 去掉行业偏离约束后，收益与信息比率有所上升，但平均行业偏离同步增大，说明行业约束确实在限制主动风险扩张
- 去掉换手率约束后，收益和夏普比率提升最明显，同时年化换手率从 `2.0828` 激增到 `8.1435`，表明换手约束是当前影响收益最显著的约束项
- 综合来看，换手率约束是当前完整策略中代价最大的限制条件，而行业约束和跟踪误差约束更多体现为“以较小收益代价换取更稳定的组合形态”

## 3.11 十年区间外部数据增益实验

### 目的

验证在当前表现最优的约束型 `LightGBM` 策略基础上，引入结构化外部数据是否能够进一步提升收益表现与风险调整后收益。

### 设置

- 基础模型：约束型 `LightGBM`
- 基础特征：价值、质量、成长、技术因子
- 外部增强特征：
  - `northbound_net_inflow`：北向资金净流入
  - `m2_yoy`：M2 同比增速
- 组合构建：与前述约束型 `LightGBM` 相同的优化器与约束配置

### 输出文件

- `data/processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv`
- `data/processed/lightgbm_metrics_external_2015_2024.csv`
- `data/processed/lightgbm_feature_importance_external_2015_2024.csv`
- `data/processed/external_feature_comparison_2015_2024.csv`

### 指标结果

不含外部数据的约束型 `LightGBM`：

- `annual_return`: `0.2887`
- `annual_excess_return`: `0.2807`
- `sharpe_ratio`: `1.2891`
- `information_ratio`: `1.6852`
- `max_drawdown`: `-0.3171`
- `monthly_win_rate`: `0.7000`

加入外部数据后的约束型 `LightGBM`：

- `annual_return`: `0.4223`
- `annual_excess_return`: `0.4143`
- `sharpe_ratio`: `1.9584`
- `information_ratio`: `2.5181`
- `max_drawdown`: `-0.2785`
- `monthly_win_rate`: `0.7500`

### 特征重要性均值

- `volatility_20`: `1715.80`
- `ret_60`: `1676.59`
- `northbound_net_inflow`: `1572.09`
- `ret_20`: `1452.22`
- `m2_yoy`: `1285.82`
- `bp`: `477.45`
- `ep_ttm`: `428.81`
- `turnover_20`: `391.20`

### 结论

- 引入外部数据后，策略的年化收益、年化超额收益、夏普比率和信息比率均明显提升
- 最大回撤由 `-0.3171` 进一步下降到 `-0.2785`，说明外部数据不仅提升了收益，也改善了回撤表现
- 北向资金净流入和 `M2` 同比在特征重要性中均进入前列，表明外部数据并非冗余变量，而是对模型决策提供了有效增量信息
- 从十年长周期结果看，结构化外部数据对指数增强策略的稳健性和收益能力均具有正向增益

## 3.12 2025 年严格训练集外测试实验

### 目的

验证在冻结模型参数并计入交易成本后，当前策略在训练区间之外的时间外推能力，进一步排查此前过高 `Sharpe` 是否来自滚动重训和零成本假设。

### 设置

- 统一测试区间：`2025-01-01` 到 `2025-12-31`
- 回测口径：冻结模型参数，不在测试期内滚动更新
- 交易成本：纳入单边手续费 `0.1%` 与单边滑点 `0.1%`
- 两组训练窗口：
  - `2024 -> 2025`
  - `2015-2024 -> 2025`
- 对比对象：
  - `baseline`
  - `factor_only`
  - `factor_with_external`

### 输出文件

- `data/processed/strict_oos_comparison_2024train.csv`
- `data/processed/strict_oos_comparison_2015train.csv`
- `data/processed/baseline_metrics_strict_oos_2025.csv`
- `data/processed/lightgbm_metrics_strict_oos_factoronly_2024train.csv`
- `data/processed/lightgbm_metrics_strict_oos_external_2024train.csv`
- `data/processed/lightgbm_metrics_strict_oos_factoronly_2015train.csv`
- `data/processed/lightgbm_metrics_strict_oos_external_2015train.csv`

### 指标结果

`2024 -> 2025` 严格 OOS：

- `annual_return`: `0.3858`
- `baseline annual_excess_return`: `0.1142`
- `baseline sharpe_ratio`: `1.7591`
- `baseline information_ratio`: `1.4604`
- `factor_only annual_excess_return`: `0.2856`
- `factor_only sharpe_ratio`: `2.5371`
- `factor_only information_ratio`: `2.7288`
- `factor_with_external annual_excess_return`: `0.2838`
- `factor_with_external sharpe_ratio`: `2.6553`
- `factor_with_external information_ratio`: `2.8570`

`2015-2024 -> 2025` 严格 OOS：

- `baseline annual_excess_return`: `0.1142`
- `baseline sharpe_ratio`: `1.7591`
- `baseline information_ratio`: `1.4604`
- `factor_only annual_excess_return`: `-0.0371`
- `factor_only sharpe_ratio`: `1.2197`
- `factor_only information_ratio`: `-0.4824`
- `factor_with_external annual_excess_return`: `-0.0215`
- `factor_with_external sharpe_ratio`: `1.3078`
- `factor_with_external information_ratio`: `-0.2830`

### 结论

- 在冻结模型并加入交易成本后，原先滚动 OOS 中 `Sharpe > 4` 的极端结果明显回落，说明此前结果确实受到滚动重训与零成本假设的明显抬升
- 当训练窗口限定为 `2024 -> 2025` 时，`LightGBM` 仍然优于基线，但夏普比率已下降到 `2.5` 左右，说明模型仍有一定泛化能力，但乐观程度被显著削弱
- 当训练窗口扩展为 `2015-2024 -> 2025` 时，`LightGBM` 相对基线不再具备明显优势，甚至出现负的年化超额收益，说明当前模型对近期市场状态的依赖较强，长历史训练并未自动带来更好的外推能力
- 引入外部数据后，在 `2024 -> 2025` 设定下仍略有提升，但在 `2015-2024 -> 2025` 设定下仅能部分缓解性能下滑，表明外部变量有增量作用，但不足以完全对冲市场非平稳性
- 因此，更合理的结论是：当前策略在“近期训练 -> 次年测试”的场景下具备一定预测能力，但若采用更长历史冻结训练，其泛化能力显著减弱，这提示后续应重点关注市场风格切换、训练窗口长度以及成本约束对模型有效性的影响

## 4. 当前阶段总结

当前项目已经具备完整的实验闭环：

1. 数据采集
2. 因子构建
3. 超额收益标签生成
4. 多因子基线回测
5. LightGBM 滚动预测
6. 机器学习增强回测
7. 基线与增强策略对比

在当前阶段，项目已经不再只是“价格因子 + 等权基线”的工程验证，而是形成了：

- 免费数据源下的十年级研究面板
- 多因子基线、约束型基线和约束型 `LightGBM` 的可比实验
- 更贴近指数增强场景的风险指标体系

这说明项目已经进入“可写正式实验章节”的阶段。

## 5. 当前局限

当前实验仍有以下限制：

- 股票覆盖不是严格 300/300，而是 `298/300`
- 虽然已经接入部分估值与财务因子，但基本面维度仍不够丰富
- 宏观外部数据尚未完整接入
- 高频分钟级数据接口已接好，但免费源在当前网络环境下稳定性较弱
- 组合优化已接入第一版约束框架，但风险模型和求解稳定性仍需进一步优化

## 6. 下一阶段建议

优先级建议如下：

1. 继续增强风险模型与优化器稳定性
2. 接入更多基本面与外部数据因子
3. 开展外部数据增益与消融实验
4. 将当前结果整理为论文正式实验章节
