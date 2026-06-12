#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查看异常 ETF 价格点前后若干交易日价格

输入：
data/etf_close.csv

输出：
终端打印异常点前后价格窗口
"""

import os
import pandas as pd


DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "etf_close.csv")

# 手动填入刚才发现的异常点
ABNORMAL_POINTS = [
    ("consumer", "2021-06-25"),
    ("new_energy", "2024-09-18"),
]

WINDOW = 10


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    for etf, date_str in ABNORMAL_POINTS:
        target_date = pd.to_datetime(date_str)

        matched = df.index[df["date"] == target_date].tolist()

        if not matched:
            print(f"找不到日期：{etf} {date_str}")
            continue

        idx = matched[0]

        start_idx = max(0, idx - WINDOW)
        end_idx = min(len(df) - 1, idx + WINDOW)

        sub_df = df.loc[start_idx:end_idx, ["date", etf]].copy()
        sub_df["daily_return"] = sub_df[etf].pct_change(fill_method=None)

        print()
        print("=" * 80)
        print(f"{etf} 异常点前后价格：{date_str}")
        print("=" * 80)

        for _, row in sub_df.iterrows():
            print(
                f"{row['date'].date()} | "
                f"price: {row[etf]} | "
                f"return: {row['daily_return']:.2%}"
            )


if __name__ == "__main__":
    main()