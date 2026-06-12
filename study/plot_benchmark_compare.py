#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
画图：策略净值 vs 基准净值
画图：策略回撤 vs 基准回撤

输入：
data/benchmark_compare.csv

输出：
1. data/nav_compare.png
2. data/drawdown_compare.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 配置区
# =========================

DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "benchmark_compare.csv")

NAV_PNG = os.path.join(DATA_DIR, "nav_compare.png")
DRAWDOWN_PNG = os.path.join(DATA_DIR, "drawdown_compare.png")


def main():
    # 读取数据
    df = pd.read_csv(INPUT_FILE)

    # 转换日期
    df["date"] = pd.to_datetime(df["date"])

    # 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    # =========================
    # 图 1：净值对比
    # =========================

    plt.figure(figsize=(12, 6))

    plt.plot(df["date"], df["strategy_nav"], label="Strategy")
    plt.plot(df["date"], df["benchmark_nav"], label="HS300 ETF")

    plt.title("Strategy NAV vs HS300 ETF")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(NAV_PNG, dpi=150)
    plt.close()

    # =========================
    # 图 2：回撤对比
    # =========================

    plt.figure(figsize=(12, 6))

    plt.plot(df["date"], df["strategy_drawdown"], label="Strategy Drawdown")
    plt.plot(df["date"], df["benchmark_drawdown"], label="HS300 ETF Drawdown")

    plt.title("Strategy Drawdown vs HS300 ETF")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(DRAWDOWN_PNG, dpi=150)
    plt.close()

    print("画图完成：")
    print(f"净值对比图：{NAV_PNG}")
    print(f"回撤对比图：{DRAWDOWN_PNG}")


if __name__ == "__main__":
    main()