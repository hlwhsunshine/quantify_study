#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二阶段：合并多个 ETF 的收盘价数据

当前目标：
1. 读取 data 目录下的多个 ETF CSV
2. 提取每只 ETF 的 close 收盘价
3. 按 date 日期合并成一张宽表
4. 保存为 data/etf_close.csv

后面月度动量轮动策略会基于这张表继续计算。
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

ETF_FILES = {
    "hs300": "hs300_sh510300.csv",
    "zz500": "zz500_sh510500.csv",
    "cyb": "cyb_sz159915.csv",
    "kc50": "kc50_sh588000.csv",
    "securities": "securities_sh512880.csv",
    "consumer": "consumer_sz159928.csv",
    "medicine": "medicine_sh512010.csv",
    "new_energy": "new_energy_sh516160.csv",
    "dividend": "dividend_sh510880.csv",
}

OUTPUT_FILE = os.path.join(DATA_DIR, "etf_close.csv")


# =========================
# 读取单个 ETF 的 close
# =========================

def read_one_close(name, file_name):
    """
    读取单只 ETF 的 CSV，并只保留：
    - date
    - close

    然后把 close 改名为 ETF 名称。
    例如 hs300 的 close 改成 hs300。
    """
    file_path = os.path.join(DATA_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    df = pd.read_csv(file_path)

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"{file_path} 缺少 date 或 close 字段")

    df = df[["date", "close"]].copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.rename(columns={"close": name})

    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 合并所有 ETF
# =========================

def merge_all_close():
    """
    逐个读取 ETF close，并按 date 外连接合并。

    outer merge 的意思是：
    - 只要某个日期任意 ETF 有数据，就保留这个日期
    - 某些 ETF 当天没数据，就显示 NaN
    """
    merged_df = None

    for name, file_name in ETF_FILES.items():
        print(f"读取：{name} - {file_name}")

        one_df = read_one_close(name, file_name)

        if merged_df is None:
            merged_df = one_df
        else:
            merged_df = pd.merge(
                merged_df,
                one_df,
                on="date",
                how="outer"
            )

    merged_df = merged_df.sort_values("date").reset_index(drop=True)

    return merged_df


# =========================
# 主程序
# =========================

def main():
    close_df = merge_all_close()

    print("=" * 60)
    print("合并完成")
    print("=" * 60)

    print("\n前 5 行：")
    print(close_df.head())

    print("\n后 5 行：")
    print(close_df.tail())

    print("\n字段：")
    print(close_df.columns)

    print("\n每列空值数量：")
    print(close_df.isna().sum())

    close_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()