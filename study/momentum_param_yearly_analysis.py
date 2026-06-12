#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
不同动量参数的年度收益分析

固定：
1. 沪深300 MA200 趋势过滤
2. 月度调仓
3. 信号下一个交易日生效，避免未来函数

测试：
5 / 10 / 20 / 30 / 40 / 50 / 60 / 80 / 100 / 120 日动量

输入：
data/etf_close.csv

输出：
data/momentum_param_yearly_analysis.csv
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "momentum_param_yearly_analysis.csv")

# 用沪深300ETF作为市场趋势过滤基准
BENCHMARK_COL = "hs300"

# 趋势过滤均线
MA_WINDOW = 200

# 要测试的动量窗口
MOMENTUM_WINDOWS = [5, 10, 20, 30, 40, 50, 60, 80, 100, 120]


# =========================
# 计算动量
# =========================

def calc_momentum(close_df, momentum_window, etf_cols):
    """
    计算每只 ETF 的 N 日动量：

    动量 = 当前收盘价 / N日前收盘价 - 1

    注意：
    etf_cols 必须提前传入，避免把后续新增的辅助列误当成 ETF。
    """
    momentum_df = close_df[["date"] + etf_cols].copy()

    for col in etf_cols:
        momentum_df[col] = close_df[col] / close_df[col].shift(momentum_window) - 1

    return momentum_df


# =========================
# 生成月度信号
# =========================

def generate_signal(close_df, momentum_window, etf_cols):
    """
    生成某个动量窗口下的月度调仓信号。

    规则：
    1. 每月最后一个交易日生成信号
    2. 选择过去 N 日动量最高的 ETF
    3. 如果最高动量 <= 0，则空仓
    4. 如果沪深300ETF收盘价 <= 沪深300ETF MA200，则空仓
    5. 否则买入动量最高 ETF
    """
    close_df = close_df.copy()

    # 计算动量
    momentum_df = calc_momentum(close_df, momentum_window, etf_cols)

    # 计算沪深300 MA200
    close_df["hs300_ma200"] = close_df[BENCHMARK_COL].rolling(MA_WINDOW).mean()

    # 合并动量数据和趋势过滤数据
    df = pd.merge(
        momentum_df,
        close_df[["date", BENCHMARK_COL, "hs300_ma200"]],
        on="date",
        how="inner",
        suffixes=("", "_close")
    )

    # 取每月最后一个交易日
    df["year_month"] = df["date"].dt.to_period("M")
    month_end_df = df.groupby("year_month").tail(1).copy()

    signals = []

    for _, row in month_end_df.iterrows():
        date = row["date"]

        # 只在真正的 ETF 列中寻找最大动量
        momentum_values = row[etf_cols]

        # 去掉 NaN，避免某些 ETF 未上市或数据不足导致 idxmax 报错
        valid_momentum_values = momentum_values.dropna()

        if valid_momentum_values.empty:
            best_etf = None
            best_momentum = None
        else:
            best_etf = valid_momentum_values.idxmax()
            best_momentum = valid_momentum_values.max()

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

    return pd.DataFrame(signals)


# =========================
# 回测
# =========================

def run_backtest(close_df, signal_df, etf_cols):
    """
    使用月度信号进行简化回测。

    关键点：
    信号在月末收盘后生成，所以从下一个交易日开始生效。
    使用 allow_exact_matches=False 避免未来函数。
    """
    close_df = close_df.copy()
    signal_df = signal_df.copy()

    # 计算每日收益率
    return_df = close_df[["date"] + etf_cols].copy()

    for col in etf_cols:
        return_df[col] = return_df[col].pct_change(fill_method=None)

    return_df[etf_cols] = return_df[etf_cols].fillna(0)

    # 信号下一个交易日生效
    daily_df = pd.merge_asof(
        return_df,
        signal_df[["date", "selected_etf"]],
        on="date",
        direction="backward",
        allow_exact_matches=False
    )

    # 没有信号前默认空仓
    daily_df["selected_etf"] = daily_df["selected_etf"].fillna("cash")

    # 根据每日持仓计算策略收益
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

    # 计算策略净值
    daily_df["strategy_nav"] = (1 + daily_df["strategy_return"]).cumprod()

    # 计算回撤
    daily_df["cummax_nav"] = daily_df["strategy_nav"].cummax()
    daily_df["drawdown"] = daily_df["strategy_nav"] / daily_df["cummax_nav"] - 1

    return daily_df


# =========================
# 年度收益计算
# =========================

def calc_year_return(group):
    """
    计算某一年的收益率：
    年末净值 / 年初净值 - 1
    """
    start_nav = group["strategy_nav"].iloc[0]
    end_nav = group["strategy_nav"].iloc[-1]
    return end_nav / start_nav - 1


# =========================
# 主函数
# =========================

def main():
    # 读取收盘价数据
    close_df = pd.read_csv(CLOSE_FILE)

    close_df["date"] = pd.to_datetime(close_df["date"])
    close_df = close_df.sort_values("date").reset_index(drop=True)

    # 提前确定真正的 ETF 列
    # 后面新增 hs300_ma200 等辅助字段时，不会影响 ETF 列表
    etf_cols = [col for col in close_df.columns if col != "date"]

    if BENCHMARK_COL not in etf_cols:
        raise ValueError(f"找不到基准列：{BENCHMARK_COL}")

    results = []

    for momentum_window in MOMENTUM_WINDOWS:
        signal_df = generate_signal(close_df, momentum_window, etf_cols)
        daily_df = run_backtest(close_df, signal_df, etf_cols)

        daily_df["year"] = daily_df["date"].dt.year

        for year, group in daily_df.groupby("year"):
            year_return = calc_year_return(group)

            # 注意：
            # 这里的 year_max_drawdown 是相对于整个策略历史高点的回撤，
            # 不是单独以该年年初为基准重新计算的年内回撤。
            year_max_drawdown = group["drawdown"].min()

            cash_ratio = (group["selected_etf"] == "cash").sum() / len(group)

            results.append({
                "momentum_window": momentum_window,
                "year": year,
                "strategy_return": year_return,
                "year_max_drawdown": year_max_drawdown,
                "cash_ratio": cash_ratio,
            })

    result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("===== 不同动量参数年度收益分析 =====")

    for momentum_window in MOMENTUM_WINDOWS:
        print()
        print(f"===== 动量{momentum_window}日 =====")

        sub_df = result_df[result_df["momentum_window"] == momentum_window]

        for _, row in sub_df.iterrows():
            print(
                f"{int(row['year'])} | "
                f"收益：{row['strategy_return']:.2%} | "
                f"年内最大回撤：{row['year_max_drawdown']:.2%} | "
                f"空仓比例：{row['cash_ratio']:.2%}"
            )

    print()
    print(f"结果已保存到：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()