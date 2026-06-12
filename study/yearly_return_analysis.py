#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
年度收益分析：策略 vs 沪深300ETF基准

输入：
data/benchmark_compare.csv

输出：
data/yearly_return_analysis.csv
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "benchmark_compare.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "yearly_return_analysis.csv")


def calc_year_return(group, nav_col):
    """
    计算某一年的收益率：
    年末净值 / 年初净值 - 1
    """
    start_nav = group[nav_col].iloc[0]
    end_nav = group[nav_col].iloc[-1]
    return end_nav / start_nav - 1


def main():
    # 读取数据
    df = pd.read_csv(INPUT_FILE)

    # 转换日期
    df["date"] = pd.to_datetime(df["date"])

    # 排序
    df = df.sort_values("date").reset_index(drop=True)

    # 提取年份
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

    # 保存
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 年度收益分析：策略 vs 基准 =====")
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