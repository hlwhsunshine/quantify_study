#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二阶段：计算多个 ETF 的 60 日动量

输入：
data/etf_close.csv

输出：
data/etf_momentum_60.csv

说明：
60 日动量 = 当前收盘价 / 60 个交易日前收盘价 - 1

这个指标后面会用于月度动量轮动：
每个月最后一个交易日，选择 momentum_60 最大的 ETF。
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "etf_close.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "etf_momentum_60.csv")

MOMENTUM_DAYS = 60


# =========================
# 读取收盘价数据
# =========================

def read_close_data():
    """
    读取 ETF 收盘价宽表。

    表格结构类似：
    date, hs300, zz500, cyb, kc50, ...
    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"文件不存在：{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "date" not in df.columns:
        raise ValueError("收盘价文件缺少 date 字段")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 计算 60 日动量
# =========================

def calc_momentum(close_df):
    """
    计算每只 ETF 的 60 日动量。

    对每一列 ETF 收盘价都计算：
    当前价格 / 60 个交易日前价格 - 1
    """
    momentum_df = close_df.copy()

    # ETF 列 = 除 date 以外的所有列
    etf_columns = [
        col for col in close_df.columns
        if col != "date"
    ]

    for col in etf_columns:
        momentum_df[col] = close_df[col] / close_df[col].shift(MOMENTUM_DAYS) - 1

    return momentum_df


# =========================
# 主程序
# =========================

def main():
    # 1. 读取收盘价
    close_df = read_close_data()

    # 2. 计算 60 日动量
    momentum_df = calc_momentum(close_df)

    # 3. 打印检查结果
    print("=" * 60)
    print("60 日动量计算完成")
    print("=" * 60)

    print("\n前 5 行：")
    print(momentum_df.head())

    print("\n后 5 行：")
    print(momentum_df.tail())

    print("\n字段：")
    print(momentum_df.columns)

    print("\n每列空值数量：")
    print(momentum_df.isna().sum())

    # 4. 查看最近一天各 ETF 的 60 日动量排名
    latest_row = momentum_df.dropna(how="all", subset=momentum_df.columns[1:]).tail(1)

    if not latest_row.empty:
        latest_date = latest_row["date"].iloc[0]

        ranking = (
            latest_row
            .drop(columns=["date"])
            .T
            .reset_index()
        )

        ranking.columns = ["etf", "momentum_60"]

        ranking = ranking.dropna().sort_values(
            "momentum_60",
            ascending=False
        )

        print("\n最近一个有效交易日：")
        print(latest_date.date())

        print("\n最近一个有效交易日的 60 日动量排名：")
        print(ranking)

    # 5. 保存结果
    momentum_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()