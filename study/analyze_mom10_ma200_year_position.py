#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10日动量 + MA200 策略：年度持仓与收益来源分析

输入：
data/backtest_daily_mom10_ma200.csv

输出：
data/year_position_analysis_mom10_ma200.csv

分析内容：
1. 每年每个 ETF 的持仓天数
2. 每年每个 ETF 的持仓比例
3. 每年每个 ETF 持仓期间的收益贡献
4. 特别用于分析 2024 年亏损来源
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "backtest_daily_mom10_ma200.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "year_position_analysis_mom10_ma200.csv")


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

            # 简单收益贡献：持有该 ETF 期间，每日策略收益相加
            # 注意：这是近似贡献，不是严格复利拆解
            simple_return_sum = group["strategy_return"].sum()

            # 复利收益贡献：只看该 ETF 持仓期间的收益复利
            compound_return = (1 + group["strategy_return"]).prod() - 1

            results.append({
                "year": year,
                "selected_etf": etf,
                "holding_days": holding_days,
                "holding_ratio": holding_ratio,
                "simple_return_sum": simple_return_sum,
                "compound_return": compound_return,
            })

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        ["year", "holding_days"],
        ascending=[True, False]
    )

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 10日动量 + MA200：年度持仓与收益来源分析 =====")

    for year, year_df in result_df.groupby("year"):
        print()
        print(f"===== {year} =====")
        print(
            year_df.to_string(
                index=False,
                formatters={
                    "holding_ratio": "{:.2%}".format,
                    "simple_return_sum": "{:.2%}".format,
                    "compound_return": "{:.2%}".format,
                }
            )
        )

    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()