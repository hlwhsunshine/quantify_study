#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETF 月度动量轮动：简化回测 + 仓位控制版本

输入：
1. data/etf_close.csv
   多只 ETF 的收盘价宽表

2. data/monthly_signal.csv
   每月最后一个交易日生成的轮动信号

输出：
1. data/backtest_daily.csv
   每日持仓、每日收益、策略净值、回撤

简化假设：
1. 每月最后一个交易日收盘后产生信号
2. 从下一个交易日开始持有 selected_etf
3. 如果 selected_etf 是 cash，则空仓，收益为 0
4. 加入仓位控制，例如只用 50% 仓位参与策略
5. 暂不考虑手续费、滑点、税费
6. 暂不考虑买入卖出成交失败
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
SIGNAL_FILE = os.path.join(DATA_DIR, "monthly_signal.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "backtest_daily.csv")

TRADING_DAYS_PER_YEAR = 252

# 仓位比例
# 1.0 表示满仓
# 0.5 表示只用 50% 资金买入 ETF，剩余 50% 现金不动
# 0.3 表示只用 30% 资金买入 ETF
POSITION_RATIO = 1


# =========================
# 读取数据
# =========================

def read_close_data():
    """
    读取 ETF 收盘价数据。

    格式类似：
    date, hs300, zz500, cyb, kc50, ...
    """
    if not os.path.exists(CLOSE_FILE):
        raise FileNotFoundError(f"文件不存在：{CLOSE_FILE}")

    df = pd.read_csv(CLOSE_FILE)

    if "date" not in df.columns:
        raise ValueError("etf_close.csv 缺少 date 字段")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


def read_signal_data():
    """
    读取月度调仓信号。

    必须包含：
    - date
    - selected_etf
    """
    if not os.path.exists(SIGNAL_FILE):
        raise FileNotFoundError(f"文件不存在：{SIGNAL_FILE}")

    df = pd.read_csv(SIGNAL_FILE)

    required_columns = ["date", "selected_etf"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"monthly_signal.csv 缺少字段：{col}")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 计算 ETF 每日收益率
# =========================

def calc_etf_daily_returns(close_df):
    """
    根据 ETF 收盘价计算每日涨跌幅。

    每日收益 = 今天收盘价 / 昨天收盘价 - 1
    """
    return_df = close_df.copy()

    etf_columns = [
        col for col in close_df.columns
        if col != "date"
    ]

    for col in etf_columns:
        return_df[col] = close_df[col].pct_change()

    return return_df


# =========================
# 生成每日持仓
# =========================

def build_daily_position(close_df, signal_df):
    """
    把月度信号转换成每日持仓。

    关键点：
    - 信号在月末交易日生成
    - 当天不能提前使用这个信号
    - 所以使用 allow_exact_matches=False
    - 表示只有 signal_date < 当前交易日，才会生效

    例如：
    2020-04-30 生成信号 cyb
    2020-05-06 开始持有 cyb
    """

    daily_dates = close_df[["date"]].copy()

    signal_for_merge = signal_df[["date", "selected_etf"]].copy()

    daily_position = pd.merge_asof(
        daily_dates,
        signal_for_merge,
        on="date",
        direction="backward",
        allow_exact_matches=False
    )

    # 第一条信号之前没有持仓，默认空仓
    daily_position["selected_etf"] = daily_position["selected_etf"].fillna("cash")

    daily_position = daily_position.rename(
        columns={"selected_etf": "position"}
    )

    return daily_position


# =========================
# 计算策略每日收益
# =========================

def calc_strategy_returns(return_df, position_df):
    """
    根据每日持仓，计算策略每日收益。

    如果 position = cash：
        strategy_return = 0

    如果 position = hs300：
        strategy_return = hs300 当天收益 * 仓位比例

    加入仓位控制后：
        例如 POSITION_RATIO = 0.5
        ETF 当天上涨 2%，账户只上涨 1%
        ETF 当天下跌 2%，账户只下跌 1%
    """

    result_df = position_df.copy()

    etf_columns = [
        col for col in return_df.columns
        if col != "date"
    ]

    # 合并每只 ETF 的每日收益率
    merged_df = pd.merge(
        result_df,
        return_df,
        on="date",
        how="left"
    )

    strategy_returns = []
    actual_positions = []

    for _, row in merged_df.iterrows():
        position = row["position"]

        # 空仓时，收益为 0，实际仓位也是 0
        if position == "cash":
            strategy_returns.append(0.0)
            actual_positions.append(0.0)
            continue

        if position not in etf_columns:
            raise ValueError(f"持仓标的 {position} 不在收益率表中")

        etf_return = row[position]

        # 如果该 ETF 当天收益为空，保守处理为空仓
        if pd.isna(etf_return):
            strategy_returns.append(0.0)
            actual_positions.append(0.0)
        else:
            # 核心修改：ETF 收益 * 仓位比例
            strategy_returns.append(etf_return * POSITION_RATIO)
            actual_positions.append(POSITION_RATIO)

    result_df["actual_position_ratio"] = actual_positions
    result_df["strategy_return"] = strategy_returns

    return result_df


# =========================
# 计算净值和回撤
# =========================

def calc_nav_and_drawdown(backtest_df):
    """
    计算策略净值和回撤。

    nav:
        策略净值。
        初始为 1。
        每天按 strategy_return 累乘。

    drawdown:
        当前净值相比历史最高净值跌了多少。
    """

    df = backtest_df.copy()

    df["nav"] = (1 + df["strategy_return"]).cumprod()

    df["cummax_nav"] = df["nav"].cummax()

    df["drawdown"] = df["nav"] / df["cummax_nav"] - 1

    return df


# =========================
# 计算绩效指标
# =========================

def calc_performance(backtest_df):
    """
    计算简化绩效指标：
    - 累计收益
    - 年化收益
    - 最大回撤
    - 交易日数量
    """

    df = backtest_df.copy()

    total_days = len(df)

    start_nav = df["nav"].iloc[0]
    end_nav = df["nav"].iloc[-1]

    total_return = end_nav / start_nav - 1

    annual_return = (end_nav / start_nav) ** (TRADING_DAYS_PER_YEAR / total_days) - 1

    max_drawdown = df["drawdown"].min()

    return {
        "total_days": total_days,
        "start_date": df["date"].min().date(),
        "end_date": df["date"].max().date(),
        "start_nav": start_nav,
        "end_nav": end_nav,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
    }


# =========================
# 主程序
# =========================

def main():
    # 1. 读取收盘价数据
    close_df = read_close_data()

    # 2. 读取月度信号
    signal_df = read_signal_data()

    # 3. 计算每只 ETF 的每日收益率
    return_df = calc_etf_daily_returns(close_df)

    # 4. 根据月度信号生成每日持仓
    position_df = build_daily_position(close_df, signal_df)

    # 5. 根据每日持仓计算策略每日收益
    backtest_df = calc_strategy_returns(return_df, position_df)

    # 6. 计算净值和回撤
    backtest_df = calc_nav_and_drawdown(backtest_df)

    # 7. 计算绩效指标
    performance = calc_performance(backtest_df)

    print("=" * 60)
    print("ETF 月度动量轮动：简化回测完成")
    print("=" * 60)

    print("\n回测参数：")
    print(f"仓位比例：{POSITION_RATIO:.0%}")

    print("\n回测区间：")
    print(f"{performance['start_date']} 到 {performance['end_date']}")

    print("\n核心指标：")
    print(f"交易日数量：{performance['total_days']}")
    print(f"初始净值：{performance['start_nav']:.4f}")
    print(f"最终净值：{performance['end_nav']:.4f}")
    print(f"累计收益：{performance['total_return']:.2%}")
    print(f"年化收益：{performance['annual_return']:.2%}")
    print(f"最大回撤：{performance['max_drawdown']:.2%}")

    print("\n前 10 行：")
    print(backtest_df.head(10))

    print("\n后 10 行：")
    print(backtest_df.tail(10))

    print("\n各持仓天数：")
    print(backtest_df["position"].value_counts())

    print("\n实际仓位天数：")
    print(backtest_df["actual_position_ratio"].value_counts())

    # 8. 保存每日回测结果
    backtest_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n每日回测结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()