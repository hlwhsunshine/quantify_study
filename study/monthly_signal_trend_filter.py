#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETF 月度动量轮动信号生成：加入沪深300 MA200 趋势过滤

规则：
1. 每月最后一个交易日生成信号
2. 计算每只 ETF 过去 60 个交易日动量
3. 选择 60 日动量最高的 ETF
4. 如果最高动量 <= 0，则空仓
5. 如果沪深300ETF收盘价 <= 沪深300ETF MA200，则空仓
6. 否则买入动量最高 ETF

输入：
1. data/etf_close.csv
2. data/etf_momentum_60.csv

输出：
1. data/monthly_signal_trend_filter.csv
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
MOMENTUM_FILE = os.path.join(DATA_DIR, "etf_momentum_60.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "monthly_signal_trend_filter.csv")

BENCHMARK_COL = "hs300"
MA_WINDOW = 200


def main():
    close_df = pd.read_csv(CLOSE_FILE)
    momentum_df = pd.read_csv(MOMENTUM_FILE)

    close_df["date"] = pd.to_datetime(close_df["date"])
    momentum_df["date"] = pd.to_datetime(momentum_df["date"])

    close_df = close_df.sort_values("date").reset_index(drop=True)
    momentum_df = momentum_df.sort_values("date").reset_index(drop=True)

    if BENCHMARK_COL not in close_df.columns:
        raise ValueError(f"找不到基准列：{BENCHMARK_COL}")

    # 计算沪深300ETF MA200
    close_df["hs300_ma200"] = close_df[BENCHMARK_COL].rolling(MA_WINDOW).mean()
    benchmark_df = close_df[["date", BENCHMARK_COL, "hs300_ma200"]].rename(
        columns={BENCHMARK_COL: "hs300_close"}
    )

    # 合并动量和沪深300趋势数据
    df = pd.merge(
        momentum_df,
        benchmark_df,
        on="date",
        how="inner"
    )

    # 取每月最后一个交易日
    df["year_month"] = df["date"].dt.to_period("M")
    month_end_df = df.groupby("year_month").tail(1).copy()

    etf_cols = [
        col for col in momentum_df.columns
        if col != "date"
    ]

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
        hs300_ma200 = row["hs300_ma200"]

        # 默认空仓
        selected_etf = "cash"
        reason = ""

        if pd.isna(best_momentum):
            selected_etf = "cash"
            reason = "momentum_nan"

        elif pd.isna(hs300_ma200):
            selected_etf = "cash"
            reason = "ma200_nan"

        elif best_momentum <= 0:
            selected_etf = "cash"
            reason = "best_momentum_le_0"

        elif hs300_close <= hs300_ma200:
            selected_etf = "cash"
            reason = "hs300_below_ma200"

        else:
            selected_etf = best_etf
            reason = "buy"

        signals.append({
            "date": date,
            "selected_etf": selected_etf,
            "best_etf": best_etf,
            "best_momentum": best_momentum,
            "hs300_close": hs300_close,
            "hs300_ma200": hs300_ma200,
            "reason": reason,
        })

    signal_df = pd.DataFrame(signals)

    signal_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 月度信号生成完成：加入趋势过滤 =====")
    print(f"信号数量：{len(signal_df)}")
    print()
    print("信号原因统计：")
    print(signal_df["reason"].value_counts())
    print()
    print("最近10条信号：")
    print(signal_df.tail(10))
    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
