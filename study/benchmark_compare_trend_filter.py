#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势过滤版策略净值 vs 基准净值对比

输入：
1. data/etf_close.csv
2. data/backtest_daily_trend_filter.csv

输出：
1. data/benchmark_compare_trend_filter.csv
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_daily_trend_filter.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "benchmark_compare_trend_filter.csv")

BENCHMARK_COL = "hs300"


def main():
    close_df = pd.read_csv(CLOSE_FILE)
    bt_df = pd.read_csv(BACKTEST_FILE)

    close_df["date"] = pd.to_datetime(close_df["date"])
    bt_df["date"] = pd.to_datetime(bt_df["date"])

    close_df = close_df.sort_values("date").reset_index(drop=True)
    bt_df = bt_df.sort_values("date").reset_index(drop=True)

    if BENCHMARK_COL not in close_df.columns:
        raise ValueError(f"找不到基准列：{BENCHMARK_COL}")

    benchmark_df = close_df[["date", BENCHMARK_COL]].copy()

    benchmark_df["benchmark_return"] = benchmark_df[BENCHMARK_COL].pct_change()
    benchmark_df["benchmark_return"] = benchmark_df["benchmark_return"].fillna(0)
    benchmark_df["benchmark_nav"] = (1 + benchmark_df["benchmark_return"]).cumprod()

    result = pd.merge(
        bt_df[["date", "strategy_nav", "drawdown"]],
        benchmark_df[["date", "benchmark_nav"]],
        on="date",
        how="inner"
    )

    result = result.rename(columns={
        "drawdown": "strategy_drawdown"
    })

    result["benchmark_cummax"] = result["benchmark_nav"].cummax()
    result["benchmark_drawdown"] = (
        result["benchmark_nav"] / result["benchmark_cummax"] - 1
    )

    result = result.drop(columns=["benchmark_cummax"])

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    strategy_final_nav = result["strategy_nav"].iloc[-1]
    benchmark_final_nav = result["benchmark_nav"].iloc[-1]

    strategy_total_return = strategy_final_nav - 1
    benchmark_total_return = benchmark_final_nav - 1

    strategy_max_drawdown = result["strategy_drawdown"].min()
    benchmark_max_drawdown = result["benchmark_drawdown"].min()

    print("===== 趋势过滤版策略 vs 基准 对比 =====")
    print(f"开始日期：{result['date'].min().date()}")
    print(f"结束日期：{result['date'].max().date()}")
    print()
    print(f"策略最终净值：{strategy_final_nav:.4f}")
    print(f"策略累计收益：{strategy_total_return:.2%}")
    print(f"策略最大回撤：{strategy_max_drawdown:.2%}")
    print()
    print(f"基准最终净值：{benchmark_final_nav:.4f}")
    print(f"基准累计收益：{benchmark_total_return:.2%}")
    print(f"基准最大回撤：{benchmark_max_drawdown:.2%}")
    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()