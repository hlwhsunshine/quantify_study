#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取已经保存好的 ETF CSV 文件，并检查数据质量。

当前目标：
1. 读取 CSV
2. 检查字段是否完整
3. 检查日期范围
4. 检查是否有空值
5. 查看 60 日动量最大的日期
"""

import pandas as pd


CSV_FILE = "data/510300_etf_daily.csv"


def main():
    # 1. 读取 CSV
    df = pd.read_csv(CSV_FILE)

    # 2. 把 date 转成日期格式
    df["date"] = pd.to_datetime(df["date"])

    # 3. 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    print("=" * 60)
    print("CSV 文件读取成功")
    print("=" * 60)

    # 4. 查看基本信息
    print("\n前 5 行：")
    print(df.head())

    print("\n后 5 行：")
    print(df.tail())

    print("\n字段：")
    print(df.columns)

    print("\n数据日期范围：")
    print(df["date"].min().date(), "到", df["date"].max().date())

    print("\n数据行数：")
    print(len(df))

    # 5. 检查空值
    print("\n每个字段的空值数量：")
    print(df.isna().sum())

    # 6. 查看 60 日动量最大的几天
    print("\n60 日动量最高的 10 天：")
    print(
        df[["date", "close", "momentum_60"]]
        .dropna()
        .sort_values("momentum_60", ascending=False)
        .head(10)
    )

    # 7. 查看最近 10 天的关键字段
    print("\n最近 10 天关键数据：")
    print(
        df[["date", "close", "daily_return", "ma20", "ma60", "momentum_60"]]
        .tail(10)
    )


if __name__ == "__main__":
    main()