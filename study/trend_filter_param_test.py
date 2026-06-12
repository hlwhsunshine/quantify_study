#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势过滤参数测试

测试不同 MA 窗口：
MA60 / MA120 / MA150 / MA200

规则：
1. 每月最后一个交易日生成信号
2. 选择 60 日动量最高 ETF
3. 如果最高动量 <= 0，则空仓
4. 如果沪深300ETF收盘价 <= 沪深300ETF MA，则空仓
5. 否则买入动量最高 ETF
6. 信号下一个交易日生效，避免未来函数

输入：
1. data/etf_close.csv
2. data/etf_momentum_60.csv

输出：
1. data/trend_filter_param_test.csv
"""

import os
import pandas as pd


DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
MOMENTUM_FILE = os.path.join(DATA_DIR, "etf_momentum_60.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "trend_filter_param_test.csv")

BENCHMARK_COL = "hs300"

MA_WINDOWS = [60, 120, 150, 200]


def generate_signal(close_df, momentum_df, ma_window):
    close_df = close_df.copy()
    momentum_df = momentum_df.copy()

    ma_col = f"hs300_ma{ma_window}"

    close_df[ma_col] = close_df[BENCHMARK_COL].rolling(ma_window).mean()
    benchmark_df = close_df[["date", BENCHMARK_COL, ma_col]].rename(
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

    etf_cols = [col for col in momentum_df.columns if col != "date"]

    signals = []

    for _, row in month_end_df.iterrows():
        date = row["date"]

        momentum_values = row[etf_cols].dropna()

        if momentum_values.empty:
            best_etf = "cash"
            best_momentum = float("nan")
        else:
            best_etf = momentum_values.idxmax()
            best_momentum = momentum_values.max()

        hs300_close = row["hs300_close"]
        hs300_ma = row[ma_col]

        selected_etf = "cash"

        if pd.isna(best_momentum):
            selected_etf = "cash"
        elif pd.isna(hs300_ma):
            selected_etf = "cash"
        elif best_momentum <= 0:
            selected_etf = "cash"
        elif hs300_close <= hs300_ma:
            selected_etf = "cash"
        else:
            selected_etf = best_etf

        signals.append({
            "date": date,
            "selected_etf": selected_etf,
            "best_etf": best_etf,
            "best_momentum": best_momentum,
            "hs300_close": hs300_close,
            "hs300_ma": hs300_ma,
        })

    signal_df = pd.DataFrame(signals)

    return signal_df


def run_backtest(close_df, signal_df):
    close_df = close_df.copy()
    signal_df = signal_df.copy()

    etf_cols = [col for col in close_df.columns if col != "date"]

    return_df = close_df.copy()

    for col in etf_cols:
        return_df[col] = return_df[col].pct_change()

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
    momentum_df = pd.read_csv(MOMENTUM_FILE)

    close_df["date"] = pd.to_datetime(close_df["date"])
    momentum_df["date"] = pd.to_datetime(momentum_df["date"])

    close_df = close_df.sort_values("date").reset_index(drop=True)
    momentum_df = momentum_df.sort_values("date").reset_index(drop=True)

    results = []

    for ma_window in MA_WINDOWS:
        signal_df = generate_signal(close_df, momentum_df, ma_window)
        daily_df = run_backtest(close_df, signal_df)
        stats = calc_stats(daily_df)

        results.append({
            "ma_window": ma_window,
            **stats,
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 趋势过滤参数测试 =====")
    for _, row in result_df.iterrows():
        print(
            f"MA{int(row['ma_window'])} | "
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
