#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势过滤版年度收益分析：策略 vs 沪深300ETF基准

输入：
data/benchmark_compare_trend_filter.csv

输出：
data/yearly_return_analysis_trend_filter.csv
"""

import os
import pandas as pd


DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "benchmark_compare_trend_filter.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "yearly_return_analysis_trend_filter.csv")


def calc_year_return(group, nav_col):
    start_nav = group[nav_col].iloc[0]
    end_nav = group[nav_col].iloc[-1]
    return end_nav / start_nav - 1


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year

    results = []

    for year, group in df.groupby("year"):
        strategy_return = calc_year_return(group, "strategy_nav")
        benchmark_return = calc_year_return(group, "benchmark_nav")
        excess_return = strategy_return - benchmark_return

        results.append({
            "year": year,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 趋势过滤版年度收益分析：策略 vs 基准 =====")
    for _, row in result_df.iterrows():
        print(
            f"{int(row['year'])} | "
            f"策略：{row['strategy_return']:.2%} | "
            f"基准：{row['benchmark_return']:.2%} | "
            f"超额：{row['excess_return']:.2%}"
        )

    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()