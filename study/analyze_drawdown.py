#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析最大回撤前后的持仓情况

目标：
1. 找到最大回撤日期
2. 找到最大回撤之前的净值高点日期
3. 查看这段时间的持仓变化
4. 统计这段时间每天持仓的 ETF
"""

import os
import pandas as pd


DATA_DIR = "data"
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_daily.csv")


def main():
    df = pd.read_csv(BACKTEST_FILE)
    df["date"] = pd.to_datetime(df["date"])

    # 最大回撤所在行
    max_dd_idx = df["drawdown"].idxmin()
    max_dd_row = df.loc[max_dd_idx]

    # 最大回撤之前的历史最高净值位置
    before_dd = df.loc[:max_dd_idx].copy()
    peak_idx = before_dd["nav"].idxmax()
    peak_row = df.loc[peak_idx]

    print("=" * 60)
    print("最大回撤分析")
    print("=" * 60)

    print("\n历史高点：")
    print("日期：", peak_row["date"].date())
    print("净值：", round(peak_row["nav"], 4))
    print("持仓：", peak_row["position"])

    print("\n最大回撤低点：")
    print("日期：", max_dd_row["date"].date())
    print("净值：", round(max_dd_row["nav"], 4))
    print("回撤：", "{:.2%}".format(max_dd_row["drawdown"]))
    print("持仓：", max_dd_row["position"])

    # 取出高点到低点之间的数据
    dd_period = df.loc[peak_idx:max_dd_idx].copy()

    print("\n回撤区间：")
    print(dd_period["date"].min().date(), "到", dd_period["date"].max().date())
    print("交易日数量：", len(dd_period))

    print("\n回撤区间各持仓天数：")
    print(dd_period["position"].value_counts())

    print("\n回撤区间每次持仓切换：")
    position_change = dd_period[
        dd_period["position"] != dd_period["position"].shift(1)
    ][["date", "position", "nav", "drawdown"]]

    print(position_change)

    print("\n回撤区间最后 20 行：")
    print(dd_period[["date", "position", "strategy_return", "nav", "drawdown"]].tail(20))


if __name__ == "__main__":
    main()