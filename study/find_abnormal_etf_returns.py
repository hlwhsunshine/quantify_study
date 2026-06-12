#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
定位 ETF 收盘价异常涨跌幅日期

输入：
data/etf_close.csv

输出：
data/abnormal_etf_returns.csv

功能：
1. 计算每只 ETF 的日收益率
2. 找出单日涨跌幅超过阈值的日期
3. 打印异常日期、前一日价格、当日价格、涨跌幅
"""

import os
import pandas as pd


DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "etf_close.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "abnormal_etf_returns.csv")

# 阈值：单日涨跌幅超过 15% 就列出来
THRESHOLD = 0.15


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    etf_cols = [col for col in df.columns if col != "date"]

    abnormal_rows = []

    print("===== ETF 异常涨跌幅检查 =====")
    print(f"阈值：单日涨跌幅超过 ±{THRESHOLD:.0%}")
    print()

    for col in etf_cols:
        price = df[col]

        daily_return = price.pct_change(fill_method=None)

        abnormal_mask = (daily_return > THRESHOLD) | (daily_return < -THRESHOLD)

        abnormal_indices = df.index[abnormal_mask].tolist()

        if len(abnormal_indices) == 0:
            continue

        print(f"===== {col} 异常记录 =====")

        for idx in abnormal_indices:
            date = df.loc[idx, "date"]
            current_price = df.loc[idx, col]

            prev_idx = idx - 1
            prev_date = df.loc[prev_idx, "date"] if prev_idx >= 0 else None
            prev_price = df.loc[prev_idx, col] if prev_idx >= 0 else None

            ret = daily_return.loc[idx]

            print(
                f"{date.date()} | "
                f"前一日：{prev_date.date() if prev_date is not None else None} | "
                f"前价：{prev_price} | "
                f"当价：{current_price} | "
                f"涨跌幅：{ret:.2%}"
            )

            abnormal_rows.append({
                "etf": col,
                "date": date.date(),
                "prev_date": prev_date.date() if prev_date is not None else None,
                "prev_price": prev_price,
                "current_price": current_price,
                "daily_return": ret,
            })

        print()

    abnormal_df = pd.DataFrame(abnormal_rows)
    abnormal_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 检查完成 =====")
    print(f"异常记录数量：{len(abnormal_df)}")
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()