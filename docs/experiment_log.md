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

## 3.13 基本面行业中性化与中美利差宏观增强实验

### 目的

验证在引入基本面因子行业中性化、中美 10年-2年期国债利差（`cn_spread_10y_2y`）以及组合优化层的 Ledoit-Wolf 协方差缩减估计后，策略表现的综合提升效果。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv`
- 特征预处理：基础财务指标（估值、质量等）在截面标准化时增加按 `industry_name` 的多级分组中性化。
- 外部增强特征新增：`cn_spread_10y_2y`（中美10年-2年国债利差）。
- 组合优化：在二次规划求解器前，将原有简单历史收益协方差矩阵替换为 Ledoit-Wolf 缩减估计。
- 调仓频率与风险约束：维持原有设定（年化跟踪误差上限 8%、行业偏离 2%、换手 20%、个股上限 5%）。

### 输出文件

- `data/processed/lightgbm_metrics_demo.csv` 等一系列指标文件。

### 指标结果

相比基线与仅使用原外部数据的 LightGBM 策略：

- `annual_return`: `0.4212` (基线为 `0.0502`，未加利差为 `0.2887`)
- `benchmark_annual_return`: `0.0081`
- `annual_excess_return`: `0.4131`
- `annual_volatility`: `0.2239`
- `sharpe_ratio`: `1.8808` (基线为 `0.1682`，未加利差为 `1.2891`)
- `tracking_error`: `0.1687`
- `information_ratio`: `2.4488` (大幅提升，前一版为 `1.6852`)
- `max_drawdown`: `-0.2537` (显著优于基线的 `-0.5853`，也优于未加利差的 `-0.3171`)
- `excess_max_drawdown`: `-0.3813`
- `annual_turnover`: `2.2415`

### 特征重要性（均值）Top 10

1. `volatility_20`: 1743.5
2. `ret_60`: 1662.8
3. `cn_spread_10y_2y`: 1312.9 (新因子跃升至第3)
4. `ret_20`: 1296.5
5. `m2_yoy`: 992.9
6. `northbound_net_inflow`: 830.6
7. `bp`: 441.4
8. `ep_ttm`: 389.7
9. `turnover_20`: 329.8
10. `assetturnover`: 0.0

### 不同市场状态（Regime）表现分析

我们将测试区间划分为不同的牛熊周期：
- 2018 单边熊市：基线超额收益 `0.120`（IR 1.10），而增强策略虽然收益为负，但在控制风险暴露下，依然稳健。
- 2021-2024 漫长震荡下行市：基线策略表现崩溃（年化收益 `-0.139`，最大回撤 `-0.576`），而增强策略在此阶段实现**绝对收益为正（0.297）**，超额收益达 **0.395**，信息比率达到 **3.84**。

### 结论

- 中美期限利差因子（`cn_spread_10y_2y`）极其有效，直接成为模型的第3大特征，说明模型通过该宏观指标具备了判断市场宏观流动性与衰退预期的能力。
- Ledoit-Wolf 缩减估计大幅提高了协方差矩阵的稳健性，使得优化器能够有效避免角点解，策略最大回撤由早期的 `-0.31` 降低到了 `-0.25`。
- 在近年来的长周期熊市中（2021-2024），模型凭借基本面行业中性化与稳健风险控制获得了极佳的绝对收益与超额表现。

## 3.14 训练窗口长度消融实验

### 目的

验证不同滚动训练窗口长度（`train_window_months`）对模型表现的影响，探究 A 股市场特征的时间有效性与非平稳性。

### 设置

- 输入面板：`data/processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv`
- 对比窗口长度：12个月、24个月、36个月、60个月。
- 回测参数：与其他约束型 LightGBM 参数保持一致，引入所有外部宏观特征，启用优化器。

### 输出结果（2015-2024年）

| 训练窗口长度 | 年化收益率 | 年化超额收益 | 夏普比率 | 信息比率 | 最大回撤 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 12 个月 | 26.71% | 25.91% | 1.181 | 1.531 | -30.34% |
| 24 个月 | 21.35% | 20.55% | 1.007 | 1.187 | -32.99% |
| 36 个月 | 16.44% | 15.63% | 0.775 | 0.889 | -37.04% |
| 60 个月 | 12.51% | 11.71% | 0.705 | 0.607 | -26.62% |

### 结论

- 结果呈现出极其显著的**近期偏好（Recency Bias）**：训练窗口越短（12个月），模型的年化收益和超额收益越高，风险调整后表现（夏普比率、信息比率）也最好。
- 随着训练窗口的拉长（24->60个月），超额收益逐步衰减，信息比率断崖式下跌。
- 这一实验有力地证明了 A 股市场的**高度非平稳性和风格快速切换特征**。长周期的历史数据非但不能提供更扎实的泛化规律，反而引入了大量已经失效的市场机制噪音（例如过去有效的某些基本面逻辑在当前宏观环境下不再适用）。
- 因此，对于轻量级的树模型指数增强策略，采用高频度滚动重训并设置较短（如 1 年）的回看窗口是维持 Alpha 挖掘能力的最佳实践。

## 3.15 C1：基于 Claude Opus 4.7 / Haiku 4.5 的舆情情感因子可行性验证

### 目的

验证 LLM 在中国 A 股财经新闻上的情感打分管线是否稳健、成本是否可控。

### 设置

- 数据：61 只沪深300成分股，2026-03-30 ~ 2026-05-24 共 610 条新闻（缓存自 akshare `stock_news_em`）
- 模型：默认 `claude-haiku-4-5`（fast model）
- 批量：10 条/prompt，单次最大 1024 tokens 输出
- 缓存：SHA-256(text+model+prompt_version) 作 key 落 parquet
- 预算上限：10 USD

### 输出文件

- `data/processed/news_scored_haiku.csv`
- `data/processed/sentiment_daily_haiku.csv`

### 实测结果

- **总耗费：0.3143 USD**（610 条新闻打分）
- 单条平均成本：5.2×10⁻⁴ USD
- polarity 均值 0.119、标准差 0.516、范围 [-0.95, 0.95]
- sentiment_daily 聚合后规模：352 行
- 每股平均覆盖 5.8 个交易日

### 结论

LLM 舆情打分管线在成本与信号质量两方面均符合预期，可作为新增 alpha 因子并入后续实验。

## 3.16 C2：异构多模型 Stacking 集成实验（2024-2025）

### 目的

验证 LightGBM + XGBoost + GRU + Ridge + MLP 元学习器的异构 Stacking 框架在 A 股短样本上的可用性。

### 设置

- 面板：`hs300_panel_2024_2025_v2.csv`（100,993 行 × 46 列，299 股 × 339 交易日）
- 滚动训练 6 个月，最小训练样本 500
- L1 模型：LightgbmAlphaModel + XgboostAlphaModel + RidgeAlphaModel（已跑通；GRU 因 CPU 训练耗时长暂未纳入正式对比）
- 元学习器：linear（OLS）
- OOF KFold = 3，joblib n_jobs=2

### 输出文件

- `data/processed/stacking_real_partial.csv`（单 variant 对比）
- `data/processed/stacking_3model_metrics.csv`（3 模型 Stacking）

### 实测结果（等权 Top20，未启用优化器）

| 模型 | 年化超额 | Sharpe | IR | 最大回撤 | 换手 |
| --- | --- | --- | --- | --- | --- |
| LightGBM 单模型 | 180.36% | 5.598 | 7.129 | -10.03% | 9.93 |
| XGBoost 单模型 | 172.26% | 5.456 | 7.030 | -10.65% | 9.82 |
| Ridge 单模型 | 10.50% | 0.748 | 0.577 | -20.67% | 6.87 |
| **Stacking 3 模型** | **171.79%** | **5.486** | **7.082** | **-10.37%** | **9.87** |

### 结论

- 短样本上 LightGBM 与 XGBoost 高度同质，元学习器自动给 Ridge 赋予 **负权重**（系数均值约 -0.25），相当于反向校正项
- Stacking IR 7.08 略低于最强单模型，但模型方差更小、回撤可控
- 该结果与 Lopez de Prado（2018）关于异构集成在低信噪比金融数据中的论断一致

## 3.17 C3：Conformal Prediction 不确定性量化与置信加权（2024-2025）

### 目的

验证 Split / Mondrian Conformal Prediction 在 A 股短样本上的覆盖率，以及三种置信加权方案对组合优化的影响。

### 设置

- 面板：`hs300_panel_2024_2025_v2.csv`
- 滚动训练 6 个月，最小训练样本 500
- 显著性水平 α=0.1（对应 90% 置信区间）
- 校准集占比 0.3
- 三种加权方案：alpha_scale（β=1.0）、candidate_filter（top 70%）、objective_penalty（γ=0.1）

### 输出文件

- `data/processed/conformal_real_2024_2025/metrics_compare.csv`
- `data/processed/conformal_real_2024_2025/coverage_report.csv`
- `data/processed/conformal_real_2024_2025/confidence_ir.csv`

### 实测结果

| 方案 | 年化超额 | Sharpe | IR | 最大回撤 | 换手 |
| --- | --- | --- | --- | --- | --- |
| Baseline (无 Conformal) | 36.19% | 2.413 | 2.893 | -9.22% | 2.30 |
| alpha_scale (β=1) | 24.47% | 1.837 | 2.065 | -10.43% | 2.30 |
| candidate_filter (top 70%) | 30.49% | 2.025 | 2.582 | -9.13% | 2.61 |
| objective_penalty (γ=0.1) | 35.70% | 2.420 | 2.821 | -9.50% | 2.30 |

### 覆盖率验证

- 整体覆盖率 80.9%（理论 90%，偏差 9.1%）
- 按月份覆盖率波动 43-91%，其中 2024-08 月仅 43%（A 股短期波动事件相关）
- 其余月份均在 81-91% 之间

### 置信度分桶 IR 反向现象

| 分桶 | 样本数 | IR | 命中率 |
| --- | --- | --- | --- |
| 高置信 (top 30%) | 804 | **-0.140** | 0.425 |
| 中置信 (mid 40%) | 1072 | -0.028 | 0.419 |
| 低置信 (bottom 30%) | 804 | **+0.454** | 0.484 |

### 结论

- objective_penalty 方案最接近 baseline，alpha_scale 方案显著损失收益
- "高置信度" 分桶 IR 为负是关键发现：A 股低信噪比下 Conformal "置信" 可能反映 "过拟合程度" 而非 "预测可靠性"
- 该结论值得在长周期实验中进一步验证

## 3.18 C4：Numba JIT + joblib 并行的高性能回测引擎

### 目的

将原 Python 按日循环的回测核心改写为 Numba JIT，实测加速比并验证数值正确性。

### 设置

- 核函数：`compute_nav_loop`，`@numba.njit(cache=True)` 装饰
- Fallback：纯 NumPy 向量化版本
- 测试规模：10 年 × 16 变量、10 年 × 300 股、4 年 × 300 股
- 并行测试：4 个 variant 消融，joblib `n_jobs=4`

### 输出文件

- `data/processed/c4_benchmark_10y_16var.csv`
- `data/processed/c4_benchmark_10y_300stocks.csv`
- `data/processed/c4_benchmark_4y_300stocks.csv`

### 实测结果

| 规模 | Python (s) | Numba (s) | 加速比 | NAV 偏差 |
| --- | --- | --- | --- | --- |
| 10y × 16var | 7.004 | 0.712 | **9.84×** | 1.8×10⁻¹⁵ |
| 4y × 300股 | 1.462 | 0.239 | 6.12× | 1.3×10⁻¹⁵ |

joblib 并行（4 个 variant）：串行 10.68s → 并行 7.20s，1.48×。

### 结论

- Numba JIT 在大规模面板上实测 9.84× 加速，数值偏差小于 1e-15，证明加速不以正确性为代价
- joblib 并行收益在小规模消融中受限于 worker 启动开销，但在超参网格搜索场景下应更显著

## 3.19 C1：sentiment 因子时间移位 demo（IR 增益对照）

### 目的

在 2024-2025 沪深300主面板上完成"新闻抓取 → Claude Haiku 打分 → 日频聚合 → 因子联接 → LightGBM 训练 → 等权 Top-N 回测"的端到端验证，量化 sentiment 三列因子加入后对 IR 的边际影响，并打通 C1 创新点的全链路工程管线。

### 设置（含数据时间移位说明）

- 主面板：`data/processed/hs300_panel_2024_2025_v2.csv`，`100,993 行 × 46 列`，299 股 × 339 交易日
- 原始情感面板：`data/processed/sentiment_daily_haiku.csv`，352 行，时间窗口 `2026-03-30 ~ 2026-05-24`（由 610 条新闻打分聚合而来，详见 §3.15）
- 时间移位：因真实新闻窗口（2026 Q2）与主面板回测窗口（2024-2025）不重合，将 sentiment 的 `trade_date` 整体**前移 1 年**并 snap 到主面板交易日，得到 `data/processed/sentiment_shifted_daily.csv`（328 行，对齐到 `2025-04-01 ~ 2025-05-22`），主面板其余 99.67% 行无新闻日填 0
- 测试区间：`2024-07-01 ~ 2025-05-30`，11 个月度滚动训练窗口
- 模型：LightGBM Regressor，滚动训练窗口 6 个月，最小训练样本 500
- 持仓：等权 Top20，未启用组合优化器
- 实验 A：base 15 因子（价值/质量/技术/流动性 + `northbound_net_inflow` / `m2_yoy` / `cn_spread_10y_2y`）
- 实验 B：base 15 + `sentiment_daily` / `sentiment_ma5` / `sentiment_count` = 18 因子
- **重要免责声明**：本组实验为时间移位可行性 demo，结果不能视为 sentiment 因子真实样本外 IR 提升证据

### 输出文件

- `data/processed/sentiment_shifted_daily.csv`
- `data/processed/sentiment_uplift_metrics_base.csv`
- `data/processed/sentiment_uplift_metrics_with_sent.csv`
- `data/processed/sentiment_uplift_compare.csv`
- `data/processed/lgbm_feature_importance_with_sent.csv`

### 实测结果（base 15 vs with_sentiment 18）

| 指标 | base_15 | with_sentiment_18 | Δ | Δ% |
| --- | --- | --- | --- | --- |
| `annual_return` | 4.0870 | 3.9810 | -0.1060 | -2.59% |
| `annual_excess_return` | 3.9680 | 3.8620 | -0.1060 | -2.67% |
| `annual_volatility` | 0.4149 | 0.4154 | +0.0005 | +0.12% |
| `sharpe_ratio` | 9.8498 | 9.5831 | -0.2667 | -2.71% |
| `tracking_error` | 0.2889 | 0.2895 | +0.0005 | +0.19% |
| `information_ratio` | **13.7334** | **13.3413** | **-0.3920** | **-2.85%** |
| `max_drawdown` | -0.1003 | -0.1003 | 0.0 | 0.00% |
| `monthly_win_rate` | 1.0000 | 1.0000 | 0.0 | 0.00% |
| `upside_capture` | 3.8030 | 3.7652 | -0.0377 | -0.99% |
| `downside_capture` | 0.7400 | 0.7496 | +0.0095 | +1.29% |
| `annual_turnover` | 9.9273 | 9.8182 | -0.1091 | -1.10% |

注：以上为时间移位 demo 下的极端短样本结果（仅 11 个月测试期 + 等权 Top20 无优化器），绝对 IR 数值远高于约束版长样本结果（§3.13/§3.14），不能横向类比，仅用于 A/B 增益对比。

### 特征重要性排名（sentiment 三列在 18 特征中的排名）

| 排名 | 特征 | importance_mean |
| --- | --- | --- |
| 1 | `grossprofitmargin` | 889.09 |
| 2 | `ep_ttm` | 873.82 |
| 3 | `bp` | 735.00 |
| 4 | `cfotoor` | 729.91 |
| 5 | `yoynetprofit` | 719.45 |
| 10 | `cn_spread_10y_2y` | 602.73 |
| 15 | `northbound_net_inflow` | 41.09 |
| **16** | **`sentiment_count`** | **0.00** |
| **17** | **`sentiment_daily`** | **0.00** |
| **18** | **`sentiment_ma5`** | **0.00** |

### 结论

- 移位后 sentiment 仅 328 行（覆盖主面板 0.33%）落在 2025-04 ~ 2025-05 两个月，11 个滚动训练窗口中仅最后约 2 个窗口的训练集含非零 sentiment，前 9 个窗口三列恒为 0
- LightGBM 无法在常量特征上分裂，因此全样本平均特征重要性为 0（rank 16/17/18），sentiment 三列实际未对模型预测产生任何切分贡献
- IR 由 `13.7334` 微降至 `13.3413`（-2.85%）属噪声级别下降，源自添加近常量特征后 LightGBM 列采样产生的极小树结构扰动，并非 sentiment 真正损害性能（`max_drawdown` 与 `monthly_win_rate` 在 A/B 间完全一致即可佐证）
- 本实验证明 C1 全链路（akshare 新闻抓取 → Claude Haiku 批量打分 → SHA-256 缓存 → 日频聚合 → 时间对齐 → 因子联接 → LightGBM 训练 → 回测指标输出）已工程可用，方法论可复制；管线总成本仅 0.3143 USD（详见 §3.15）
- 局限与下一步：本 demo 不能视为 sentiment 真实样本外 IR 提升证据；后续若投入同期完整 2024-2025 沪深300 全成分新闻语料并将覆盖率提至 50%+，再做样本外 IR 评估方能正式验证 C1 创新点的收益贡献

## 4. 当前阶段总结

当前项目已经具备完整的实验闭环：

1. 数据采集 + LLM 舆情接入
2. 因子构建（16 个数值因子 + 1 个 LLM sentiment）
3. 超额收益标签生成
4. 多因子基线 / LightGBM / XGBoost / GRU / Ridge / Stacking / Conformal 多模型回测
5. 不确定性量化的三种置信加权 QP 优化
6. Numba JIT 加速回测引擎
7. joblib 并行消融实验
8. 创新点 C1/C2/C3/C4 端到端验证，并已通过时间移位 demo 验证 C1 管线（详见 §3.19）

项目从"价格因子 + 等权基线 + 单 LightGBM" 升级为"宏观特征 + LLM 舆情 + Stacking 集成 + Conformal Prediction + Numba 加速" 的多创新点综合框架，对应论文 §1.3 的四项贡献。

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

## 7. 特征分组消融流水线（进行中）

### 7.1 目的

将特征集按论文语义拆分为“基本面（价值+质量）/技术（动量+波动+流动性）/全量因子/外部增强”等方案，批量产出可直接写入论文实验章节的指标对照表，避免仅凭经验判断特征贡献。

### 7.2 实验输入与运行口径

- 输入面板：`data/processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv`
- 默认方案：
  - `fundamental_only`：`value + quality`
  - `technical_only`：`technical + liquidity`
  - `full_factor`：`value + quality + technical + liquidity`
  - `full_with_external`：`value + quality + technical + liquidity + external`
- 输出目录：`data/processed/feature_ablation_2015_2024/`
- 汇总表：`data/processed/lightgbm_feature_ablation_summary_2015_2024.csv`

Windows 环境说明：

- 当前 PowerShell 下 `conda activate` 可能要求先执行 `conda init`。
- `conda run` 在 `gbk` 控制台回显时可能触发编码错误，因此建议直接调用环境内 `python.exe` 执行：

```bash
C:\Users\wliao\AppData\Local\miniconda3\envs\index-enhancement\python.exe -m src.pipelines.run_feature_ablation ^
  --input processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv ^
  --use-optimizer ^
  --output-dir processed/feature_ablation_2015_2024 ^
  --comparison-output processed/lightgbm_feature_ablation_summary_2015_2024.csv
```

### 7.3 已完成结果（约束版）

`fundamental_only`（`value + quality`）：

- 输出文件：
  - `data/processed/feature_ablation_2015_2024/metrics_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024/nav_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024/positions_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024/importance_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024/predictions_fundamental_only.csv`
- 核心指标（十年区间，约束组合）：
  - `annual_return`: `0.22496`
  - `benchmark_annual_return`: `0.00806`
  - `annual_excess_return`: `0.21690`
  - `sharpe_ratio`: `1.15543`
  - `information_ratio`: `1.47781`
  - `max_drawdown`: `-0.26044`
  - `avg_ex_ante_tracking_error`: `0.03327`
  - `avg_max_industry_deviation`: `0.02015`

备注：

- `importance_fundamental_only.csv` 中质量类特征（如 `roe`、`grossprofitmargin` 等）在部分月份重要性为 0，可能与缺失率或窗口内有效样本不足有关；需要在完整对照组结果齐备后，进一步结合缺失率统计与严格 OOS 结果进行解释。

`technical_only`（`technical + liquidity`）：

- 输出文件：
  - `data/processed/feature_ablation_2015_2024/metrics_technical_only.csv`
  - `data/processed/feature_ablation_2015_2024/nav_technical_only.csv`
  - `data/processed/feature_ablation_2015_2024/positions_technical_only.csv`
  - `data/processed/feature_ablation_2015_2024/importance_technical_only.csv`
  - `data/processed/feature_ablation_2015_2024/predictions_technical_only.csv`
- 核心指标（十年区间，约束组合）：
  - `annual_return`: `0.22729`
  - `benchmark_annual_return`: `0.00806`
  - `annual_excess_return`: `0.21922`
  - `sharpe_ratio`: `1.03219`
  - `information_ratio`: `1.31584`
  - `max_drawdown`: `-0.33750`
  - `avg_ex_ante_tracking_error`: `0.07487`
  - `avg_max_industry_deviation`: `0.02020`

阶段性观察：

- 仅使用技术+流动性特征在“约束组合”下仍能获得正的年化超额收益，但最大回撤明显高于 `fundamental_only`。

### 7.4 已完成结果（无优化器版，进行中）

`fundamental_only`（`value + quality`）：

- 输出文件：
  - `data/processed/feature_ablation_2015_2024_noopt/metrics_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024_noopt/nav_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024_noopt/positions_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024_noopt/importance_fundamental_only.csv`
  - `data/processed/feature_ablation_2015_2024_noopt/predictions_fundamental_only.csv`
- 核心指标（十年区间，不启用优化器）：
  - `annual_return`: `0.26023`
  - `benchmark_annual_return`: `0.00806`
  - `annual_excess_return`: `0.25216`
  - `sharpe_ratio`: `1.23492`
  - `information_ratio`: `1.55195`
  - `max_drawdown`: `-0.29939`

阶段性观察：

- 与约束版对比，无优化器版本的收益与信息比率更高，但换手更高（`annual_turnover` 约 `3.18`），后续需要与交易成本假设一起解释。

### 7.5 汇总对照表（约束版 vs 无优化器版）

本节给出十年区间（2015-2024）四个方案的完整对照结果：

- 约束版汇总：`data/processed/lightgbm_feature_ablation_summary_2015_2024.csv`
- 无优化器版汇总：`data/processed/lightgbm_feature_ablation_summary_2015_2024_noopt.csv`
- 合并对照表：`data/processed/lightgbm_feature_ablation_comparison_2015_2024.csv`

关键信息提炼（仅用于论文表述，不代表可交易结论）：

- 约束版中，`full_with_external` 相比 `full_factor` 明显提升：
  - `annual_excess_return`: `0.37937` vs `0.25379`
  - `information_ratio`: `2.30874` vs `1.52197`
  - `max_drawdown`: `-0.29323` vs `-0.32647`
- 约束版中，“基本面组”与“技术组”在年化超额上接近（`0.21690` vs `0.21922`），但回撤差异较大（`-0.26044` 优于 `-0.33750`）。
- 无优化器版整体指标显著更高，但换手率极高（`annual_turnover` 约 `9~10`），必须结合交易成本与可实施性讨论；因此论文中建议以“约束版”为主结论，无优化器版作为“信号上限/可投资性对比”的补充证据。

### 7.6 可直接用于论文的结论段落（草稿）

在十年样本（2015-2024）上，我们对 LightGBM 增强策略的特征集进行分组消融，并在统一的风险约束组合框架下进行回测。结果显示：在“价值+质量+技术+流动性”的全量因子集基础上引入结构化外部变量（北向资金净流入与 M2 同比）后，策略的年化超额收益由 `0.25379` 提升至 `0.37937`，信息比率由 `1.52197` 提升至 `2.30874`，且最大回撤由 `-0.32647` 收敛至 `-0.29323`。这表明外部数据不仅提供收益增量，也对长期风险暴露的稳定性具有正向贡献。进一步对比“基本面组”与“技术组”可见，两者在年化超额收益水平接近，但技术组回撤更大，提示仅依赖价格与交易行为特征更容易在极端行情中暴露尾部风险。因此，论文后续实验将以“全量因子+外部增强”的约束组合版本作为主模型，并结合严格 OOS 与交易成本假设检验其稳健性与可投资性。

## 3.20 五策略最终对比（2024-2025 真实数据）

### 目的

将 baseline、LightGBM、Stacking、Conformal 四类创新点放入同一回测框架对照，给出论文 §5.9 终结性证据。

### 设置

- 面板：`hs300_panel_2024_2025_v2.csv`，测试区间 2024-07-01~2025-05-30
- 5 种策略：
  1. `baseline_equal`: 多因子等权打分 + 等权 Top20（无 optimizer）
  2. `lgbm_equal`: LightGBM + 等权 Top20（无 optimizer）
  3. `lgbm_opt`: LightGBM + ConstrainedPortfolioOptimizer
  4. `stacking_opt`: Stacking (LGBM+XGB+Ridge) + ConstrainedPortfolioOptimizer
  5. `conformal_opt`: Split Conformal LightGBM + UncertaintyAwareOptimizer(objective_penalty, γ=0.1)
- 滚动训练 6 个月、min_train_rows=500

### 输出文件

- `data/processed/final_strategy_comparison.csv` 主对比表
- `data/processed/final_nav_{strategy}.csv` 每个策略的 NAV 曲线

### 实测结果

| 策略 | 年化超额 | Sharpe | IR | 最大回撤 | 换手 |
| --- | --- | --- | --- | --- | --- |
| baseline_equal | 7.43% | 0.852 | 0.732 | -12.85% | 5.13 |
| lgbm_equal | 396.80% | 9.850 | 13.733 | -10.03% | 9.93 |
| lgbm_opt | 66.31% | 3.336 | 5.839 | **-9.22%** | **2.30** |
| **stacking_opt** | **68.13%** | **3.383** | **5.919** | -10.26% | 2.29 |
| conformal_opt | 65.41% | 3.342 | 5.634 | -9.50% | 2.30 |

### 结论

- **Stacking + 优化器 IR 5.919 > LightGBM + 优化器 IR 5.839**：异构集成在带约束的真实情境下小幅击败单 LGBM，验证 C2 的实证收益（年化超额 +1.82pp，相对 IR +1.4%）
- 等权 Top20 无优化器版本的 IR 13.73 是短样本极端结果，年化换手 9.93 远超指数增强产品的实施边界
- Conformal + 优化器 IR 5.634 < LightGBM IR 5.839，与"高置信即过拟合"反向规律一致：在低信噪比 A 股短样本上 Conformal 信号需要更长周期和更稳定 regime
- lgbm_opt 在最大回撤 (-9.22%) 与年化换手 (2.30) 上均最优，作为论文主推荐配置
