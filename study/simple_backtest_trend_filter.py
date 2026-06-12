#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETF 月度动量轮动：加入趋势过滤后的简化回测

输入：
1. data/etf_close.csv
   多只 ETF 的收盘价宽表

2. data/monthly_signal_trend_filter.csv
   加入沪深300 MA200 趋势过滤后的月度信号

输出：
1. data/backtest_daily_trend_filter.csv
   每日持仓、每日收益、策略净值、回撤

简化假设：
1. 每月最后一个交易日收盘后产生信号
2. 从下一个交易日开始持有 selected_etf
3. 如果 selected_etf 是 cash，则空仓，收益为 0
4. 暂不考虑手续费、滑点、税费
5. 暂不考虑买入卖出成交失败
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
SIGNAL_FILE = os.path.join(DATA_DIR, "monthly_signal_trend_filter.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "backtest_daily_trend_filter.csv")


def main():
    # 读取 ETF 收盘价
    close_df = pd.read_csv(CLOSE_FILE)

    # 读取月度信号
    signal_df = pd.read_csv(SIGNAL_FILE)

    # 转换日期
    close_df["date"] = pd.to_datetime(close_df["date"])
    signal_df["date"] = pd.to_datetime(signal_df["date"])

    # 排序
    close_df = close_df.sort_values("date").reset_index(drop=True)
    signal_df = signal_df.sort_values("date").reset_index(drop=True)

    # 计算每只 ETF 的日收益率
    etf_cols = [col for col in close_df.columns if col != "date"]
    return_df = close_df.copy()

    for col in etf_cols:
        return_df[col] = return_df[col].pct_change()

    return_df[etf_cols] = return_df[etf_cols].fillna(0)

    # 信号日期是月末收盘后生成
    # 因此不能当天使用信号，需要从下一个交易日开始生效
    # allow_exact_matches=False 可以避免未来函数
    daily_df = pd.merge_asof(
        return_df,
        signal_df[["date", "selected_etf"]],
        on="date",
        direction="backward",
        allow_exact_matches=False
    )

    # 没有信号前默认空仓
    daily_df["selected_etf"] = daily_df["selected_etf"].fillna("cash")

    # 根据每日持仓计算策略日收益
    strategy_returns = []

    for _, row in daily_df.iterrows():
        selected_etf = row["selected_etf"]

        if selected_etf == "cash":
            strategy_return = 0
        else:
            strategy_return = row[selected_etf]

        strategy_returns.append(strategy_return)

    daily_df["strategy_return"] = strategy_returns

    # 计算策略净值
    daily_df["strategy_nav"] = (1 + daily_df["strategy_return"]).cumprod()

    # 计算回撤
    daily_df["cummax_nav"] = daily_df["strategy_nav"].cummax()
    daily_df["drawdown"] = daily_df["strategy_nav"] / daily_df["cummax_nav"] - 1

    # 保存结果
    daily_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # 输出统计
    start_date = daily_df["date"].iloc[0]
    end_date = daily_df["date"].iloc[-1]

    final_nav = daily_df["strategy_nav"].iloc[-1]
    total_return = final_nav - 1
    max_drawdown = daily_df["drawdown"].min()

    days = len(daily_df)
    annual_return = final_nav ** (252 / days) - 1

    print("===== 趋势过滤版简化回测结果 =====")
    print(f"开始日期：{start_date.date()}")
    print(f"结束日期：{end_date.date()}")
    print(f"交易日数量：{days}")
    print()
    print(f"最终净值：{final_nav:.4f}")
    print(f"累计收益：{total_return:.2%}")
    print(f"年化收益：{annual_return:.2%}")
    print(f"最大回撤：{max_drawdown:.2%}")
    print()
    print("持仓天数统计：")
    print(daily_df["selected_etf"].value_counts())
    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()