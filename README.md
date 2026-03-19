# IndexEnhancementStrategy

## 项目定位

本项目服务于本科毕设《面向指数增强策略的量化算法优化设计与实现》。
研究对象为沪深300指数增强策略，目标是在控制跟踪误差的前提下，通过多因子、机器学习和外部数据提升组合相对基准的超额收益。

## 当前收敛后的主线

项目采用四阶段主线推进：

1. 多因子基线策略
2. 机器学习增强
3. 组合优化与风险约束
4. 外部数据增益验证

这样设计的原因是：

- 先保证研究闭环可复现
- 再逐步叠加 AI 与外部数据
- 保证毕设在个人电脑上可落地

## 已固定的最小可行研究方案
 
- 标的范围：按调仓日可得的沪深300成分股
- 调仓频率：月度调仓
- 基准指数：沪深300
- 目标函数：优先提升年化超额收益和信息比率，同时控制跟踪误差
- 核心因子：价值、质量、动量、低波动F
- 主模型：LightGBM
- 外部数据：北向资金净流入、M2同比增速
- 组合约束：跟踪误差、行业偏离、个股权重上限、换手率

## 当前已实现能力

当前代码已经支持以下实验链路：

1. `akshare + baostock` 十年级研究面板构建
2. 多因子基线与约束型基线回测
3. `LightGBM` 滚动训练、冻结训练和严格 OOS 测试
4. 外部数据增强与约束消融实验
5. 按因子分组批量运行特征消融实验

## 仓库结构

```text
IndexEnhancementStrategy/
├─ data/                  # 本地数据缓存和中间结果
├─ docs/                  # 研究范围、MVP定义、实验方案
├─ logs/                  # 训练与实验日志
├─ paper/                 # 毕业论文 LaTeX 源码与图表
├─ reference/             # 参考文献与毕业设计规范要求等资料
├─ src/                   # 策略实现代码
└─ README.md
```

## 当前文档

- `docs/scope.md`：最终研究边界与技术路线
- `docs/mvp.md`：最小可行研究原型定义
- `docs/experiments.md`：实验矩阵与对照方案
- `docs/references.md`：外部参考实现与模块映射
- `docs/experiment_log.md`：当前阶段实验记录与结果汇总

## 当前建议推进顺序

1. 继续补充基本面与外部数据因子
2. 基于分组消融与严格 OOS 结果筛选稳健特征集
3. 优化约束型组合构建与风险模型稳定性
4. 将现有结果沉淀为论文实验章节与图表

## 当前可运行入口

当前仓库已经补齐第一版“数据读取 -> 因子构建 -> 标签生成”骨架。
当前默认使用 `akshare` 行情 + `baostock` 基本面补充，无需 `TUSHARE_TOKEN`。
在项目根目录执行：

```bash
python -m src.pipelines.build_factor_panel --start-date 20180101 --end-date 20241231
```

运行后会在 `data/processed/` 下生成第一版研究因子面板。

约束型 `LightGBM` 实验示例：

```bash
python -m src.pipelines.run_lightgbm_experiment \
  --input processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv \
  --use-optimizer \
  --feature-groups value,quality,technical,liquidity,external \
  --prediction-output processed/lightgbm_predictions_demo.csv \
  --importance-output processed/lightgbm_importance_demo.csv \
  --nav-output processed/lightgbm_nav_demo.csv \
  --positions-output processed/lightgbm_positions_demo.csv \
  --metrics-output processed/lightgbm_metrics_demo.csv
```

特征分组消融实验示例：

```bash
python -m src.pipelines.run_feature_ablation \
  --input processed/hs300_factor_panel_constrained_fast_external_2015_2024.csv \
  --use-optimizer \
  --output-dir processed/feature_ablation_demo \
  --comparison-output processed/lightgbm_feature_ablation_demo.csv
```

## 约定

- `requirement.md` 只作为需求边界，不做修改
- 训练日志统一保存在 `logs/` 目录
- 优先做日频或月频研究，不引入高频复杂度
