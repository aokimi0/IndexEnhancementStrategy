# 实验记录

## 1. 记录目的

本文档用于汇总当前阶段已经完成的实验、关键参数、结果指标与阶段性结论。
当前记录的重点是验证研究链路是否跑通，并形成后续论文中“基线策略”与“AI 增强策略”的实验起点。

## 2. 当前实验环境

### 2.1 数据源

- 主数据源：`akshare`
- 说明：由于当前 `Tushare Pro` 账号权限不足，项目暂时采用 `akshare` 作为免 token 兜底数据源

### 2.2 运行环境

- 环境管理：`conda`
- Python 版本：`3.12`
- 主要依赖：
  - `pandas`
  - `numpy`
  - `akshare`
  - `lightgbm`
  - `scikit-learn`

### 2.3 当前数据特点

- 价格类字段已可稳定获取
- 动量、波动率等技术类因子已可构造
- 估值与财务类因子目前仍不完整，相关列在当前阶段大多为空
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

## 4. 当前阶段总结

当前项目已经具备完整的实验闭环：

1. 数据采集
2. 因子构建
3. 超额收益标签生成
4. 多因子基线回测
5. LightGBM 滚动预测
6. 机器学习增强回测
7. 基线与增强策略对比

在当前仅使用价格类因子的情况下，`LightGBM` 尚未显著拉开收益，但已经表现出：

- 更高的信息比率
- 更低的最大回撤
- 较好的风险调整后收益表现

这说明项目已经进入“可写实验结论”的阶段，而不是仅停留在工程验证阶段。

## 5. 当前局限

当前实验仍有以下限制：

- 股票覆盖不是严格 300/300，而是 `298/300`
- 当前因子仍以价格类因子为主
- 估值、财务和宏观外部数据尚未完整接入
- 高频分钟级数据接口已接好，但免费源在当前网络环境下稳定性较弱
- 组合构建仍是等权选股逻辑，尚未加入跟踪误差和行业偏离约束

## 6. 下一阶段建议

优先级建议如下：

1. 接入更多基本面与外部数据因子
2. 在现有 LightGBM 实验基础上扩展特征集
3. 引入组合优化与跟踪误差约束
4. 将实验结果整理为论文中的正式实验章节
