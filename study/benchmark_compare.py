#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略净值 vs 基准净值对比

输入：
1. data/etf_close.csv
   多只 ETF 的收盘价宽表

2. data/backtest_daily.csv
   简化回测后的每日结果

输出：
1. data/benchmark_compare.csv
   策略与基准的净值、回撤对比
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_daily.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "benchmark_compare.csv")

# 这里用沪深300ETF作为基准
BENCHMARK_COL = "hs300"


# =========================
# 主逻辑
# =========================

def main():
    # 读取收盘价
    close_df = pd.read_csv(CLOSE_FILE)

    # 读取回测结果
    bt_df = pd.read_csv(BACKTEST_FILE)

    # 转换日期
    close_df["date"] = pd.to_datetime(close_df["date"])
    bt_df["date"] = pd.to_datetime(bt_df["date"])

    # 按日期排序
    close_df = close_df.sort_values("date").reset_index(drop=True)
    bt_df = bt_df.sort_values("date").reset_index(drop=True)

    # 检查基准列是否存在
    if BENCHMARK_COL not in close_df.columns:
        raise ValueError(f"找不到基准列：{BENCHMARK_COL}")

    # 提取基准价格
    benchmark_df = close_df[["date", BENCHMARK_COL]].copy()

    # 计算基准日收益率
    benchmark_df["benchmark_return"] = benchmark_df[BENCHMARK_COL].pct_change()

    # 第一天收益设为 0
    benchmark_df["benchmark_return"] = benchmark_df["benchmark_return"].fillna(0)

    # 计算基准净值
    benchmark_df["benchmark_nav"] = (1 + benchmark_df["benchmark_return"]).cumprod()

    # 合并策略净值和基准净值
    result = pd.merge(
        bt_df[["date", "nav", "drawdown"]],
        benchmark_df[["date", "benchmark_nav"]],
        on="date",
        how="inner"
    )

    # 回测结果里的 nav 是策略净值，这里重命名后方便和基准净值对比
    result = result.rename(columns={
        "nav": "strategy_nav",
        "drawdown": "strategy_drawdown"
    })

    # 计算基准回撤
    result["benchmark_cummax"] = result["benchmark_nav"].cummax()
    result["benchmark_drawdown"] = (
        result["benchmark_nav"] / result["benchmark_cummax"] - 1
    )

    # 删除辅助列
    result = result.drop(columns=["benchmark_cummax"])

    # 保存结果
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # =========================
    # 输出统计结果
    # =========================

    strategy_final_nav = result["strategy_nav"].iloc[-1]
    benchmark_final_nav = result["benchmark_nav"].iloc[-1]

    strategy_total_return = strategy_final_nav - 1
    benchmark_total_return = benchmark_final_nav - 1

    strategy_max_drawdown = result["strategy_drawdown"].min()
    benchmark_max_drawdown = result["benchmark_drawdown"].min()

    print("===== 策略 vs 基准 对比 =====")
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
