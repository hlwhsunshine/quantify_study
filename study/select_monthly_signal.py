#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二阶段：生成 ETF 月度动量轮动信号

输入：
data/etf_momentum_60.csv

输出：
data/monthly_signal.csv

策略规则：
1. 每个月最后一个交易日观察所有 ETF 的 60 日动量
2. 选择 60 日动量最高的 ETF
3. 如果最高动量 <= 0，则空仓 cash
4. 信号在本月最后一个交易日生成
5. 后面回测时，从下一个交易日开始持有该 ETF
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "etf_momentum_60.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "monthly_signal.csv")


# =========================
# 读取动量数据
# =========================

def read_momentum_data():
    """
    读取 60 日动量宽表。

    表格结构类似：
    date, hs300, zz500, cyb, kc50, ...
    """
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"文件不存在：{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "date" not in df.columns:
        raise ValueError("动量文件缺少 date 字段")

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 找每个月最后一个交易日
# =========================

def get_month_end_rows(momentum_df):
    """
    找出每个月最后一个交易日。

    注意：
    这里不是自然月最后一天，而是数据里每个月最后出现的那个交易日。

    比如：
    2020-01-31 如果是交易日，就是 2020-01-31
    如果自然月最后一天是周末，则可能是 2020-01-30 或 2020-01-29
    """
    df = momentum_df.copy()

    # 生成月份字段，例如 2020-01、2020-02
    df["month"] = df["date"].dt.to_period("M")

    # 每个月取最后一行
    month_end_df = (
        df.groupby("month", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    return month_end_df


# =========================
# 生成月度信号
# =========================

def generate_signals(month_end_df):
    """
    在每个月最后一个交易日，选择动量最强的 ETF。

    如果所有 ETF 中最高动量 <= 0，则空仓 cash。
    """
    signals = []

    # ETF 列 = 除 date 和 month 之外的所有列
    etf_columns = [
        col for col in month_end_df.columns
        if col not in ["date", "month"]
    ]

    for _, row in month_end_df.iterrows():
        signal_date = row["date"]

        # 取出这一行所有 ETF 的动量
        momentum_values = row[etf_columns]

        # 去掉空值。上市时间不够、数据不足的 ETF 会是 NaN
        momentum_values = momentum_values.dropna()

        # 如果这一天所有 ETF 都没有有效动量，则跳过
        if momentum_values.empty:
            continue

        # 找到动量最大的 ETF
        best_etf = momentum_values.idxmax()
        best_momentum = momentum_values.max()

        # 如果最高动量 <= 0，说明所有 ETF 过去 60 日都不强，选择空仓
        if best_momentum <= 0:
            selected_etf = "cash"
        else:
            selected_etf = best_etf

        signals.append({
            "date": signal_date,
            "selected_etf": selected_etf,
            "best_etf": best_etf,
            "best_momentum_60": best_momentum,
        })

    signal_df = pd.DataFrame(signals)

    return signal_df


# =========================
# 主程序
# =========================

def main():
    # 1. 读取 60 日动量数据
    momentum_df = read_momentum_data()

    # 2. 找出每个月最后一个交易日
    month_end_df = get_month_end_rows(momentum_df)

    # 3. 生成月度信号
    signal_df = generate_signals(month_end_df)

    print("=" * 60)
    print("月度动量信号生成完成")
    print("=" * 60)

    print("\n前 10 条信号：")
    print(signal_df.head(10))

    print("\n后 10 条信号：")
    print(signal_df.tail(10))

    print("\n信号数量：")
    print(len(signal_df))

    print("\n各 ETF 被选中的次数：")
    print(signal_df["selected_etf"].value_counts())

    # 4. 保存结果
    signal_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()