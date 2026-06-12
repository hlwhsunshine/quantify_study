#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查看简化回测结果：

1. 读取 data/backtest_daily.csv
2. 画策略净值曲线
3. 画策略回撤曲线
4. 找出最大回撤发生日期
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = "data"
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_daily.csv")
NAV_IMAGE_FILE = os.path.join(DATA_DIR, "backtest_nav.png")
DRAWDOWN_IMAGE_FILE = os.path.join(DATA_DIR, "backtest_drawdown.png")


def main():
    df = pd.read_csv(BACKTEST_FILE)

    df["date"] = pd.to_datetime(df["date"])

    print("=" * 60)
    print("回测结果检查")
    print("=" * 60)

    print("\n前 5 行：")
    print(df.head())

    print("\n后 5 行：")
    print(df.tail())

    # 找最大回撤
    max_drawdown_row = df.loc[df["drawdown"].idxmin()]

    print("\n最大回撤信息：")
    print("日期：", max_drawdown_row["date"].date())
    print("当日净值：", round(max_drawdown_row["nav"], 4))
    print("最大回撤：", "{:.2%}".format(max_drawdown_row["drawdown"]))
    print("当日持仓：", max_drawdown_row["position"])

    # 画净值曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["nav"], label="strategy nav")
    plt.title("ETF Momentum Strategy NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(NAV_IMAGE_FILE, dpi=150)
    plt.close()
    print("\n净值曲线已保存到：", NAV_IMAGE_FILE)

    # 画回撤曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["drawdown"], label="drawdown")
    plt.title("ETF Momentum Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DRAWDOWN_IMAGE_FILE, dpi=150)
    plt.close()
    print("回撤曲线已保存到：", DRAWDOWN_IMAGE_FILE)


if __name__ == "__main__":
    main()
