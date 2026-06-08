# Quantify

这是一个量化学习项目，当前主要围绕 ETF 日线数据、60 日动量、月度轮动信号和简化回测展开。

项目目前实现的是一个基础版 ETF 月度动量轮动策略：

1. 下载多只 ETF 的历史日线行情。
2. 合并各 ETF 的收盘价。
3. 计算每只 ETF 的 60 日动量。
4. 每月最后一个交易日选择动量最强的 ETF。
5. 如果最强动量小于等于 0，则空仓。
6. 从下一个交易日开始持有选中的 ETF。
7. 计算策略每日收益、净值和最大回撤。

## 目录结构

```text
.
├── data/
│   ├── *_*.csv                 # 单只 ETF 的日线数据
│   ├── etf_close.csv           # 多只 ETF 收盘价宽表
│   ├── etf_momentum_60.csv     # 多只 ETF 的 60 日动量宽表
│   ├── monthly_signal.csv      # 月度轮动信号
│   └── backtest_daily.csv      # 每日回测结果
└── study/
    ├── etf_demo.py             # 单只 ETF 数据获取和指标计算示例
    ├── download_multi_etf.py   # 批量下载多只 ETF 数据
    ├── merge_etf_close.py      # 合并多只 ETF 的 close 收盘价
    ├── calc_etf_momentum.py    # 计算 60 日动量
    ├── select_monthly_signal.py # 生成月度轮动信号
    ├── simple_backtest.py      # 简化回测
    └── read_etf_csv.py         # 读取并检查 ETF CSV 数据
```

## 环境依赖

当前脚本使用 Python 3，主要依赖：

```bash
pip install pandas akshare
```

如果本机 `python` 指向 Python 2，请使用：

```bash
/usr/bin/python3 script.py
```

## 运行流程

建议按下面顺序运行。

### 1. 批量下载 ETF 数据

```bash
/usr/bin/python3 study/download_multi_etf.py
```

输出示例：

```text
data/hs300_sh510300.csv
data/zz500_sh510500.csv
data/cyb_sz159915.csv
...
```

当前 ETF 池在 `study/download_multi_etf.py` 的 `ETF_LIST` 中配置，包括沪深300、中证500、创业板、科创50、证券、消费、医药、新能源、红利等 ETF。

### 2. 合并 ETF 收盘价

```bash
/usr/bin/python3 study/merge_etf_close.py
```

生成：

```text
data/etf_close.csv
```

这个文件是宽表结构：

```text
date,hs300,zz500,cyb,kc50,securities,consumer,medicine,new_energy,dividend
```

每一列代表一只 ETF 的收盘价。

### 3. 计算 60 日动量

```bash
/usr/bin/python3 study/calc_etf_momentum.py
```

生成：

```text
data/etf_momentum_60.csv
```

60 日动量的计算公式：

```text
momentum_60 = 当前收盘价 / 60 个交易日前收盘价 - 1
```

### 4. 生成月度轮动信号

```bash
/usr/bin/python3 study/select_monthly_signal.py
```

生成：

```text
data/monthly_signal.csv
```

信号规则：

```text
每个月最后一个交易日，选择 60 日动量最高的 ETF。
如果最高动量 <= 0，则 selected_etf = cash。
```

### 5. 运行简化回测

```bash
/usr/bin/python3 study/simple_backtest.py
```

生成：

```text
data/backtest_daily.csv
```

回测结果包含：

```text
date              日期
position          当天持仓，cash 表示空仓
strategy_return   策略当天收益率
nav               策略净值
cummax_nav        历史最高净值
drawdown          当前回撤
```

## 策略说明

当前策略是单标的月度动量轮动。

每个月只选择一只 ETF：

```text
如果 selected_etf = hs300，则下个月持有沪深300ETF。
如果 selected_etf = cyb，则下个月持有创业板ETF。
如果 selected_etf = cash，则下个月空仓。
```

当前版本不会同时持有多只 ETF，也没有仓位权重。例如不会出现：

```text
50% hs300 + 50% cyb
```

策略收益计算方式：

```text
如果 position = cash：
    strategy_return = 0

如果 position = 某只 ETF：
    strategy_return = 这只 ETF 当天的日收益率
```

净值计算方式：

```text
nav = (1 + strategy_return).cumprod()
```

回撤计算方式：

```text
drawdown = nav / cummax_nav - 1
```

## 当前简化假设

当前回测是学习版简化模型，暂时做了这些假设：

- 每月最后一个交易日收盘后产生信号。
- 从下一个交易日开始持有选中的 ETF。
- `cash` 表示空仓，收益按 0 处理。
- 暂不考虑手续费、滑点、印花税、管理费等交易成本。
- 暂不考虑成交失败、涨跌停、停牌等交易限制。
- 新浪数据源返回的是未复权行情，严肃回测时需要进一步确认数据源和复权方式。

## 常见字段解释

`daily_return`：每日涨跌幅。

```text
daily_return = 今天收盘价 / 昨天收盘价 - 1
```

`ma20`：20 日均线。

```text
ma20 = 最近 20 个交易日收盘价的平均值
```

`ma60`：60 日均线。

```text
ma60 = 最近 60 个交易日收盘价的平均值
```

`momentum_60`：60 日动量。

```text
momentum_60 = 当前收盘价 / 60 个交易日前收盘价 - 1
```

`strategy_return`：策略当天收益率。

`nav`：策略净值，初始为 1。

`cummax_nav`：截至当天的历史最高净值。

`drawdown`：当前净值相对历史最高净值的回撤。

## 后续可以继续完善

- 增加手续费和滑点。
- 增加多 ETF 等权或按权重持仓。
- 使用前复权数据源。
- 增加基准指数对比，例如沪深300或中证全指。
- 增加夏普比率、波动率、胜率等绩效指标。
- 增加可视化图表，例如净值曲线和回撤曲线。
