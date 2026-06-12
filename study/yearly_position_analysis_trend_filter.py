#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势过滤版：按年份统计持仓天数和各 ETF 收益贡献

输入：
data/backtest_daily_trend_filter.csv

输出：
data/yearly_position_analysis_trend_filter.csv
"""

import os
import pandas as pd


DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "backtest_daily_trend_filter.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "yearly_position_analysis_trend_filter.csv")


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["year"] = df["date"].dt.year

    results = []

    for year, year_df in df.groupby("year"):
        total_days = len(year_df)

        for etf, group in year_df.groupby("selected_etf"):
            holding_days = len(group)
            holding_ratio = holding_days / total_days

            # 简化收益贡献：该持仓状态下每日策略收益累加
            simple_return_sum = group["strategy_return"].sum()

            results.append({
                "year": year,
                "selected_etf": etf,
                "holding_days": holding_days,
                "holding_ratio": holding_ratio,
                "simple_return_sum": simple_return_sum,
            })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        ["year", "holding_days"],
        ascending=[True, False]
    )

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 趋势过滤版：年度持仓分析 =====")

    for year, year_df in result_df.groupby("year"):
        print()
        print(f"===== {year} =====")
        print(year_df.to_string(index=False, formatters={
            "holding_ratio": "{:.2%}".format,
            "simple_return_sum": "{:.2%}".format,
        }))

    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()