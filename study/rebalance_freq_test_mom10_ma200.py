#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调仓周期参数测试：固定 10日动量 + 沪深300MA200 趋势过滤

测试：
1. weekly     每周最后一个交易日调仓
2. every_10d  每10个交易日调仓
3. every_15d  每15个交易日调仓
4. monthly    每月最后一个交易日调仓

规则：
1. 在调仓日收盘后生成信号
2. 计算每只 ETF 最近10个交易日动量
3. 选择动量最高的 ETF
4. 如果最高动量 <= 0，则空仓
5. 如果沪深300ETF <= MA200，则空仓
6. 否则买入动量最高 ETF
7. 信号从下一个交易日开始生效，避免未来函数

输入：
data/etf_close.csv

输出：
data/rebalance_freq_test_mom10_ma200.csv
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "rebalance_freq_test_mom10_ma200.csv")

BENCHMARK_COL = "hs300"

MOMENTUM_WINDOW = 10
MA_WINDOW = 200

REBALANCE_MODES = [
    "weekly",
    "every_10d",
    "every_15d",
    "monthly",
]


# =========================
# 工具函数
# =========================

def calc_annual_return(final_nav, days):
    return final_nav ** (252 / days) - 1


def get_rebalance_dates(df, mode):
    """
    根据调仓模式生成调仓日。

    注意：
    这里的调仓日是“信号生成日”。
    实际持仓从下一个交易日开始生效。
    """
    df = df.copy()

    if mode == "weekly":
        # 每周最后一个交易日
        df["year_week"] = df["date"].dt.to_period("W")
        rebalance_df = df.groupby("year_week").tail(1).copy()

    elif mode == "monthly":
        # 每月最后一个交易日
        df["year_month"] = df["date"].dt.to_period("M")
        rebalance_df = df.groupby("year_month").tail(1).copy()

    elif mode == "every_10d":
        # 每10个交易日调仓一次
        rebalance_df = df.iloc[::10].copy()

    elif mode == "every_15d":
        # 每15个交易日调仓一次
        rebalance_df = df.iloc[::15].copy()

    else:
        raise ValueError(f"未知调仓模式：{mode}")

    return rebalance_df[["date"]].copy()


def generate_signal(close_df, etf_cols, mode):
    """
    生成指定调仓周期下的信号。
    """
    close_df = close_df.copy()

    # 计算10日动量
    momentum_df = close_df[["date"]].copy()

    for col in etf_cols:
        momentum_df[col] = close_df[col] / close_df[col].shift(MOMENTUM_WINDOW) - 1

    # 计算沪深300 MA200
    trend_df = close_df[["date", BENCHMARK_COL]].copy()
    trend_df["hs300_ma200"] = trend_df[BENCHMARK_COL].rolling(MA_WINDOW).mean()

    # 合并
    signal_base = pd.merge(
        momentum_df,
        trend_df,
        on="date",
        how="inner",
        suffixes=("", "_close")
    )

    # 获取调仓日
    rebalance_dates = get_rebalance_dates(signal_base[["date"]].copy(), mode)

    # 只保留调仓日
    signal_base = pd.merge(
        signal_base,
        rebalance_dates,
        on="date",
        how="inner"
    )

    signals = []

    for _, row in signal_base.iterrows():
        date = row["date"]

        momentum_values = row[etf_cols]
        valid_momentum_values = momentum_values.dropna()

        if valid_momentum_values.empty:
            best_etf = None
            best_momentum = None
        else:
            best_etf = valid_momentum_values.idxmax()
            best_momentum = valid_momentum_values.max()

        # 注意：
        # hs300 是动量列
        # hs300_close 才是沪深300收盘价
        hs300_close = row[f"{BENCHMARK_COL}_close"]
        hs300_ma200 = row["hs300_ma200"]

        selected_etf = "cash"
        reason = ""

        if best_etf is None:
            selected_etf = "cash"
            reason = "momentum_all_nan"

        elif pd.isna(hs300_ma200):
            selected_etf = "cash"
            reason = "ma200_nan"

        elif best_momentum <= 0:
            selected_etf = "cash"
            reason = "best_momentum_le_0"

        elif hs300_close <= hs300_ma200:
            selected_etf = "cash"
            reason = "hs300_below_ma200"

        else:
            selected_etf = best_etf
            reason = "buy"

        signals.append({
            "date": date,
            "selected_etf": selected_etf,
            "best_etf": best_etf,
            "best_momentum": best_momentum,
            "hs300_close": hs300_close,
            "hs300_ma200": hs300_ma200,
            "reason": reason,
        })

    signal_df = pd.DataFrame(signals)

    return signal_df


def run_backtest(close_df, signal_df, etf_cols):
    """
    根据信号回测。
    """
    return_df = close_df[["date"] + etf_cols].copy()

    for col in etf_cols:
        return_df[col] = return_df[col].pct_change(fill_method=None)

    return_df[etf_cols] = return_df[etf_cols].fillna(0)

    # 信号从下一个交易日生效
    daily_df = pd.merge_asof(
        return_df,
        signal_df[["date", "selected_etf"]],
        on="date",
        direction="backward",
        allow_exact_matches=False
    )

    daily_df["selected_etf"] = daily_df["selected_etf"].fillna("cash")

    strategy_returns = []

    for _, row in daily_df.iterrows():
        selected_etf = row["selected_etf"]

        if selected_etf == "cash":
            strategy_return = 0

        elif selected_etf not in etf_cols:
            raise ValueError(f"selected_etf 不在 ETF 列中：{selected_etf}")

        else:
            strategy_return = row[selected_etf]

        strategy_returns.append(strategy_return)

    daily_df["strategy_return"] = strategy_returns
    daily_df["strategy_nav"] = (1 + daily_df["strategy_return"]).cumprod()

    daily_df["cummax_nav"] = daily_df["strategy_nav"].cummax()
    daily_df["drawdown"] = daily_df["strategy_nav"] / daily_df["cummax_nav"] - 1

    return daily_df


def calc_turnover_days(daily_df):
    """
    统计实际换仓次数。

    只要 selected_etf 和前一天不同，就算一次换仓。
    第一行不算。
    """
    position_change = daily_df["selected_etf"] != daily_df["selected_etf"].shift(1)
    turnover_days = position_change.sum() - 1

    if turnover_days < 0:
        turnover_days = 0

    return turnover_days


def calc_stats(daily_df, signal_df, mode):
    days = len(daily_df)

    final_nav = daily_df["strategy_nav"].iloc[-1]
    total_return = final_nav - 1
    annual_return = calc_annual_return(final_nav, days)
    max_drawdown = daily_df["drawdown"].min()

    cash_days = (daily_df["selected_etf"] == "cash").sum()
    cash_ratio = cash_days / days

    signal_count = len(signal_df)
    buy_signal_count = (signal_df["reason"] == "buy").sum()

    turnover_days = calc_turnover_days(daily_df)

    return {
        "rebalance_mode": mode,
        "final_nav": final_nav,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "cash_days": cash_days,
        "cash_ratio": cash_ratio,
        "signal_count": signal_count,
        "buy_signal_count": buy_signal_count,
        "turnover_days": turnover_days,
    }


def main():
    close_df = pd.read_csv(CLOSE_FILE)

    close_df["date"] = pd.to_datetime(close_df["date"])
    close_df = close_df.sort_values("date").reset_index(drop=True)

    etf_cols = [col for col in close_df.columns if col != "date"]

    if BENCHMARK_COL not in etf_cols:
        raise ValueError(f"找不到基准列：{BENCHMARK_COL}")

    results = []

    for mode in REBALANCE_MODES:
        signal_df = generate_signal(close_df, etf_cols, mode)
        daily_df = run_backtest(close_df, signal_df, etf_cols)
        stats = calc_stats(daily_df, signal_df, mode)

        results.append(stats)

    result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 调仓周期参数测试：10日动量 + MA200 =====")

    for _, row in result_df.iterrows():
        print(
            f"{row['rebalance_mode']} | "
            f"最终净值：{row['final_nav']:.4f} | "
            f"累计收益：{row['total_return']:.2%} | "
            f"年化收益：{row['annual_return']:.2%} | "
            f"最大回撤：{row['max_drawdown']:.2%} | "
            f"空仓比例：{row['cash_ratio']:.2%} | "
            f"信号数：{int(row['signal_count'])} | "
            f"买入信号数：{int(row['buy_signal_count'])} | "
            f"实际换仓次数：{int(row['turnover_days'])}"
        )

    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()