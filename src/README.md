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
- `src/data/loaders.py`：沪深300研究面板读取与拼接
- `src/factors/engine.py`：基础因子计算与未来超额收益标签构造
- `src/factors/preprocess.py`：去极值与标准化
- `src/pipelines/build_factor_panel.py`：一键生成第一版研究因子面板

## 运行方式

先设置环境变量 `TUSHARE_TOKEN`，再在项目根目录执行：

```bash
python -m src.pipelines.build_factor_panel --start-date 20180101 --end-date 20241231
```

运行后会在 `data/processed/` 下生成第一版因子面板文件。
