#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETF 收盘价数据质量检查

输入：
data/etf_close.csv

输出：
data/etf_data_quality_report.csv

检查内容：
1. 每只 ETF 的起始日期
2. 每只 ETF 的结束日期
3. 有效交易日数量
4. 缺失值数量
5. 缺失值比例
6. 最大单日涨幅
7. 最大单日跌幅
8. 是否存在异常涨跌幅
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "etf_close_clean.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "etf_data_quality_report.csv")

# 单日涨跌幅超过这个阈值，认为需要重点检查
ABNORMAL_RETURN_THRESHOLD = 0.15


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    etf_cols = [col for col in df.columns if col != "date"]

    total_rows = len(df)

    results = []

    print("===== ETF 数据质量检查 =====")
    print(f"总交易日行数：{total_rows}")
    print(f"检查 ETF 数量：{len(etf_cols)}")
    print()

    for col in etf_cols:
        price = df[col]

        valid_price = price.dropna()

        if valid_price.empty:
            results.append({
                "etf": col,
                "start_date": None,
                "end_date": None,
                "valid_days": 0,
                "missing_days": total_rows,
                "missing_ratio": 1.0,
                "first_price": None,
                "last_price": None,
                "max_daily_return": None,
                "min_daily_return": None,
                "abnormal_up_days": 0,
                "abnormal_down_days": 0,
            })
            continue

        first_valid_index = valid_price.index[0]
        last_valid_index = valid_price.index[-1]

        start_date = df.loc[first_valid_index, "date"]
        end_date = df.loc[last_valid_index, "date"]

        valid_days = valid_price.count()
        missing_days = price.isna().sum()
        missing_ratio = missing_days / total_rows

        first_price = valid_price.iloc[0]
        last_price = valid_price.iloc[-1]

        daily_return = price.pct_change(fill_method=None)

        max_daily_return = daily_return.max()
        min_daily_return = daily_return.min()

        abnormal_up_days = (daily_return > ABNORMAL_RETURN_THRESHOLD).sum()
        abnormal_down_days = (daily_return < -ABNORMAL_RETURN_THRESHOLD).sum()

        results.append({
            "etf": col,
            "start_date": start_date.date(),
            "end_date": end_date.date(),
            "valid_days": valid_days,
            "missing_days": missing_days,
            "missing_ratio": missing_ratio,
            "first_price": first_price,
            "last_price": last_price,
            "max_daily_return": max_daily_return,
            "min_daily_return": min_daily_return,
            "abnormal_up_days": abnormal_up_days,
            "abnormal_down_days": abnormal_down_days,
        })

    report_df = pd.DataFrame(results)

    report_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 汇总报告 =====")
    for _, row in report_df.iterrows():
        print(
            f"{row['etf']} | "
            f"起始：{row['start_date']} | "
            f"结束：{row['end_date']} | "
            f"有效天数：{int(row['valid_days'])} | "
            f"缺失天数：{int(row['missing_days'])} | "
            f"缺失比例：{row['missing_ratio']:.2%} | "
            f"最大涨幅：{row['max_daily_return']:.2%} | "
            f"最大跌幅：{row['min_daily_return']:.2%} | "
            f"异常上涨天数：{int(row['abnormal_up_days'])} | "
            f"异常下跌天数：{int(row['abnormal_down_days'])}"
        )

    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()