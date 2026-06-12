#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动量参数测试：固定沪深300 MA200趋势过滤

测试不同动量窗口：
20 / 40 / 60 / 90 / 120

规则：
1. 每月最后一个交易日生成信号
2. 计算每只 ETF 过去 N 个交易日动量
3. 选择动量最高的 ETF
4. 如果最高动量 <= 0，则空仓
5. 如果沪深300ETF收盘价 <= 沪深300ETF MA200，则空仓
6. 否则买入动量最高 ETF
7. 信号下一个交易日生效，避免未来函数

输入：
data/etf_close.csv

输出：
data/momentum_param_test_with_ma200.csv
"""

import os
import pandas as pd


DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "momentum_param_test_with_ma200.csv")

BENCHMARK_COL = "hs300"

MA_WINDOW = 200
# MOMENTUM_WINDOWS = [20, 40, 60, 90, 120]
MOMENTUM_WINDOWS = [5,10, 20, 30, 40, 50, 60, 80, 100, 120]

def calc_momentum(close_df, momentum_window):
    momentum_df = close_df.copy()

    etf_cols = [col for col in close_df.columns if col != "date"]

    for col in etf_cols:
        momentum_df[col] = close_df[col] / close_df[col].shift(momentum_window) - 1

    return momentum_df


def generate_signal(close_df, momentum_window):
    close_df = close_df.copy()
    etf_cols = [col for col in close_df.columns if col != "date"]

    momentum_df = calc_momentum(close_df, momentum_window)

    close_df["hs300_ma200"] = close_df[BENCHMARK_COL].rolling(MA_WINDOW).mean()
    benchmark_df = close_df[["date", BENCHMARK_COL, "hs300_ma200"]].rename(
        columns={BENCHMARK_COL: "hs300_close"}
    )

    df = pd.merge(
        momentum_df,
        benchmark_df,
        on="date",
        how="inner"
    )

    df["year_month"] = df["date"].dt.to_period("M")
    month_end_df = df.groupby("year_month").tail(1).copy()

    signals = []

    for _, row in month_end_df.iterrows():
        date = row["date"]

        momentum_values = row[etf_cols]

        # 关键修复：
        # 去掉 NaN，避免某些 ETF 未上市或数据不足导致 idxmax 报错
        valid_momentum_values = momentum_values.dropna()

        if valid_momentum_values.empty:
            best_etf = None
            best_momentum = None
        else:
            best_etf = valid_momentum_values.idxmax()
            best_momentum = valid_momentum_values.max()

        hs300_close = row["hs300_close"]
        hs300_ma200 = row["hs300_ma200"]

        selected_etf = "cash"

        if best_etf is None:
            selected_etf = "cash"
        elif pd.isna(hs300_ma200):
            selected_etf = "cash"
        elif best_momentum <= 0:
            selected_etf = "cash"
        elif hs300_close <= hs300_ma200:
            selected_etf = "cash"
        else:
            selected_etf = best_etf

        signals.append({
            "date": date,
            "selected_etf": selected_etf,
            "best_etf": best_etf,
            "best_momentum": best_momentum,
            "hs300_close": hs300_close,
            "hs300_ma200": hs300_ma200,
        })

    signal_df = pd.DataFrame(signals)
    return signal_df

def run_backtest(close_df, signal_df):
    close_df = close_df.copy()
    signal_df = signal_df.copy()

    etf_cols = [col for col in close_df.columns if col != "date"]

    return_df = close_df.copy()

    for col in etf_cols:
        return_df[col] = return_df[col].pct_change(fill_method=None)

    return_df[etf_cols] = return_df[etf_cols].fillna(0)

    daily_df = pd.merge_asof(
        return_df,
        signal_df[["date", "selected_etf"]],
        on="date",
        direction="backward",
        allow_exact_matches=False
    )

    daily_df["selected_etf"] = daily_df["selected_etf"].fillna("cash")

    strategy_returns = []

    for _, row in daily_df.iterrows():
        selected_etf = row["selected_etf"]

        if selected_etf == "cash":
            strategy_return = 0
        else:
            strategy_return = row[selected_etf]

        strategy_returns.append(strategy_return)

    daily_df["strategy_return"] = strategy_returns
    daily_df["strategy_nav"] = (1 + daily_df["strategy_return"]).cumprod()

    daily_df["cummax_nav"] = daily_df["strategy_nav"].cummax()
    daily_df["drawdown"] = daily_df["strategy_nav"] / daily_df["cummax_nav"] - 1

    return daily_df


def calc_stats(daily_df):
    final_nav = daily_df["strategy_nav"].iloc[-1]
    total_return = final_nav - 1

    days = len(daily_df)
    annual_return = final_nav ** (252 / days) - 1

    max_drawdown = daily_df["drawdown"].min()

    cash_days = (daily_df["selected_etf"] == "cash").sum()
    cash_ratio = cash_days / days

    return {
        "final_nav": final_nav,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "cash_days": cash_days,
        "cash_ratio": cash_ratio,
    }


def main():
    close_df = pd.read_csv(CLOSE_FILE)

    close_df["date"] = pd.to_datetime(close_df["date"])
    close_df = close_df.sort_values("date").reset_index(drop=True)

    results = []

    for momentum_window in MOMENTUM_WINDOWS:
        signal_df = generate_signal(close_df, momentum_window)
        daily_df = run_backtest(close_df, signal_df)
        stats = calc_stats(daily_df)

        results.append({
            "momentum_window": momentum_window,
            **stats,
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 动量参数测试：固定 MA200 趋势过滤 =====")
    for _, row in result_df.iterrows():
        print(
            f"动量{int(row['momentum_window'])}日 | "
            f"最终净值：{row['final_nav']:.4f} | "
            f"累计收益：{row['total_return']:.2%} | "
            f"年化收益：{row['annual_return']:.2%} | "
            f"最大回撤：{row['max_drawdown']:.2%} | "
            f"空仓比例：{row['cash_ratio']:.2%}"
        )

    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
