# 代码模块规划

后续代码建议按以下职责拆分：

- `src/data/`：行情、财务、外部数据的读取、清洗与对齐
- `src/factors/`：因子计算与截面预处理
- `src/models/`：LightGBM 训练、预测与特征重要性导出
- `src/portfolio/`：权重生成、约束优化、换手控制
- `src/backtest/`：净值回测、费用处理、绩效评估

当前仓库仍处于研究方案落地阶段，待数据接口确定后再逐步补全上述模块。

## 当前已实现

- `src/config.py`：项目路径和环境变量配置
- `src/data/tushare_client.py`：Tushare 轻量客户端
- `src/data/akshare_client.py`：Akshare 免 token 数据客户端
- `src/data/loaders.py`：沪深300研究面板读取与拼接
- `src/factors/engine.py`：基础因子计算与未来超额收益标签构造
- `src/factors/preprocess.py`：去极值与标准化
- `src/pipelines/build_factor_panel.py`：一键生成第一版研究因子面板
- `src/backtest/`：最小多因子基线回测
- `src/pipelines/run_baseline_backtest.py`：一键运行基线回测
- `src/pipelines/collect_higher_frequency_data.py`：采集分钟级高频数据

## 运行方式

当前默认使用 `akshare` 负责行情与指数成分，使用 `baostock` 补充日频估值和季度财务指标，无需 token。
在项目根目录执行：

```bash
python -m src.pipelines.build_factor_panel --start-date 20240101 --end-date 20241231
```

运行后会在 `data/processed/` 下生成第一版因子面板文件。
如果只想先验证链路是否稳定，建议先限制股票池规模：

```bash
python -m src.pipelines.build_factor_panel --data-source akshare --start-date 20240101 --end-date 20240331 --universe-limit 10 --output processed/hs300_factor_panel_sample.csv
```

运行过程中会自动：

- 将成功请求的数据缓存到 `data/cache/`
- 将失败的单只股票请求记录到 `logs/akshare_failures.log`
- 遇到单股票下载失败时跳过该股票，不中断整批任务

在得到因子面板后，可以继续运行最小多因子基线回测：

```bash
python -m src.pipelines.run_baseline_backtest --input processed/hs300_factor_panel_sample.csv --top-n 5
```

如果需要额外采集更高频的分钟级行情，可执行：

```bash
python -m src.pipelines.collect_higher_frequency_data --start-datetime "2024-03-01 09:30:00" --end-datetime "2024-03-05 15:00:00" --period 5 --universe-limit 50 --stocks-output processed/hs300_minute_5m_sample.csv --benchmark-output processed/hs300_benchmark_minute_5m_sample.csv
```


如果后续具备 `Tushare` 权限，可显式切换：

```bash
python -m src.pipelines.build_factor_panel --data-source tushare --start-date 20240101 --end-date 20241231
```
