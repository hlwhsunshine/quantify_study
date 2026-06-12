#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动量周期 + 趋势均线 + 调仓周期 三维参数测试

测试参数：
1. 动量周期：
   5 / 10 / 15 / 20 / 30

2. 趋势过滤均线：
   MA60 / MA120 / MA150 / MA200

3. 调仓周期：
   weekly     每周最后一个交易日调仓
   every_10d  每10个交易日调仓
   every_15d  每15个交易日调仓
   monthly    每月最后一个交易日调仓

策略规则：
1. 在调仓日收盘后生成信号
2. 计算每只 ETF 过去 N 个交易日动量
3. 选择动量最高的 ETF
4. 如果最高动量 <= 0，则空仓
5. 如果沪深300ETF收盘价 <= 沪深300ETF MA，则空仓
6. 否则买入动量最高 ETF
7. 信号从下一个交易日开始生效，避免未来函数

输入：
data/etf_close.csv

输出：
data/param_grid_test_momentum_ma_rebalance.csv

暂不考虑：
1. 手续费
2. 滑点
3. 买卖价差
4. 成交失败
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close_clean.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "param_grid_test_momentum_ma_rebalance.csv")

BENCHMARK_COL = "hs300"

MOMENTUM_WINDOWS = [5, 10, 15, 20, 30]

MA_WINDOWS = [60, 120, 150, 200]

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
    """
    按 252 个交易日计算年化收益。
    """
    return final_nav ** (252 / days) - 1


def calc_max_drawdown(nav_series):
    """
    根据净值序列计算最大回撤。
    """
    cummax_nav = nav_series.cummax()
    drawdown = nav_series / cummax_nav - 1
    return drawdown.min()


def get_rebalance_dates(df, mode):
    """
    根据调仓模式生成调仓日。

    注意：
    这里的调仓日是信号生成日。
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


# =========================
# 信号生成
# =========================

def generate_signal(close_df, etf_cols, momentum_window, ma_window, rebalance_mode):
    """
    生成某组参数下的信号。
    """
    close_df = close_df.copy()

    # 计算动量
    momentum_df = close_df[["date"]].copy()

    for col in etf_cols:
        momentum_df[col] = close_df[col] / close_df[col].shift(momentum_window) - 1

    # 计算趋势过滤均线
    trend_df = close_df[["date", BENCHMARK_COL]].copy()
    ma_col = f"hs300_ma{ma_window}"
    trend_df[ma_col] = trend_df[BENCHMARK_COL].rolling(ma_window).mean()

    # 合并动量和趋势数据
    # 注意：
    # momentum_df 里的 hs300 是沪深300动量
    # trend_df 里的 hs300 是沪深300收盘价
    # 合并后收盘价列会变成 hs300_close
    signal_base = pd.merge(
        momentum_df,
        trend_df,
        on="date",
        how="inner",
        suffixes=("", "_close")
    )

    # 获取调仓日
    rebalance_dates = get_rebalance_dates(
        signal_base[["date"]].copy(),
        rebalance_mode
    )

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

        hs300_close = row[f"{BENCHMARK_COL}_close"]
        hs300_ma = row[ma_col]

        selected_etf = "cash"
        reason = ""

        if best_etf is None:
            selected_etf = "cash"
            reason = "momentum_all_nan"

        elif pd.isna(hs300_ma):
            selected_etf = "cash"
            reason = "ma_nan"

        elif best_momentum <= 0:
            selected_etf = "cash"
            reason = "best_momentum_le_0"

        elif hs300_close <= hs300_ma:
            selected_etf = "cash"
            reason = "hs300_below_ma"

        else:
            selected_etf = best_etf
            reason = "buy"

        signals.append({
            "date": date,
            "selected_etf": selected_etf,
            "best_etf": best_etf,
            "best_momentum": best_momentum,
            "hs300_close": hs300_close,
            "hs300_ma": hs300_ma,
            "reason": reason,
        })

    signal_df = pd.DataFrame(signals)

    return signal_df


# =========================
# 回测
# =========================

def run_backtest(close_df, signal_df, etf_cols):
    """
    根据信号进行每日回测。
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


def calc_turnover_count(daily_df):
    """
    统计实际换仓次数。

    selected_etf 和前一天不同，就算一次变化。
    第一行不算换仓。
    """
    position_change = daily_df["selected_etf"] != daily_df["selected_etf"].shift(1)

    turnover_count = position_change.sum() - 1

    if turnover_count < 0:
        turnover_count = 0

    return turnover_count


def calc_stats(
    daily_df,
    signal_df,
    momentum_window,
    ma_window,
    rebalance_mode
):
    """
    计算某组参数的统计指标。
    """
    days = len(daily_df)

    final_nav = daily_df["strategy_nav"].iloc[-1]
    total_return = final_nav - 1
    annual_return = calc_annual_return(final_nav, days)
    max_drawdown = daily_df["drawdown"].min()

    cash_days = (daily_df["selected_etf"] == "cash").sum()
    cash_ratio = cash_days / days

    signal_count = len(signal_df)
    buy_signal_count = (signal_df["reason"] == "buy").sum()

    turnover_count = calc_turnover_count(daily_df)

    # 简单收益回撤比：累计收益 / 最大回撤绝对值
    # 用来辅助排序，不是专业指标
    if max_drawdown == 0:
        return_drawdown_ratio = None
    else:
        return_drawdown_ratio = total_return / abs(max_drawdown)

    return {
        "momentum_window": momentum_window,
        "ma_window": ma_window,
        "rebalance_mode": rebalance_mode,
        "final_nav": final_nav,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "return_drawdown_ratio": return_drawdown_ratio,
        "cash_days": cash_days,
        "cash_ratio": cash_ratio,
        "signal_count": signal_count,
        "buy_signal_count": buy_signal_count,
        "turnover_count": turnover_count,
    }


# =========================
# 主函数
# =========================

def main():
    close_df = pd.read_csv(CLOSE_FILE)

    close_df["date"] = pd.to_datetime(close_df["date"])
    close_df = close_df.sort_values("date").reset_index(drop=True)

    etf_cols = [col for col in close_df.columns if col != "date"]

    if BENCHMARK_COL not in etf_cols:
        raise ValueError(f"找不到基准列：{BENCHMARK_COL}")

    results = []

    total_tests = (
        len(MOMENTUM_WINDOWS)
        * len(MA_WINDOWS)
        * len(REBALANCE_MODES)
    )

    test_index = 0

    for momentum_window in MOMENTUM_WINDOWS:
        for ma_window in MA_WINDOWS:
            for rebalance_mode in REBALANCE_MODES:
                test_index += 1

                print(
                    f"正在测试 {test_index}/{total_tests}："
                    f"动量{momentum_window}日 + "
                    f"MA{ma_window} + "
                    f"{rebalance_mode}"
                )

                signal_df = generate_signal(
                    close_df=close_df,
                    etf_cols=etf_cols,
                    momentum_window=momentum_window,
                    ma_window=ma_window,
                    rebalance_mode=rebalance_mode
                )

                daily_df = run_backtest(
                    close_df=close_df,
                    signal_df=signal_df,
                    etf_cols=etf_cols
                )

                stats = calc_stats(
                    daily_df=daily_df,
                    signal_df=signal_df,
                    momentum_window=momentum_window,
                    ma_window=ma_window,
                    rebalance_mode=rebalance_mode
                )

                results.append(stats)

    result_df = pd.DataFrame(results)

    # 保存完整结果
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print()
    print("===== 参数测试完成 =====")
    print(f"总测试组合数：{len(result_df)}")
    print(f"结果已保存到：{OUTPUT_FILE}")

    # =========================
    # 输出 Top 结果
    # =========================

    print()
    print("===== 按累计收益排序 Top 15 =====")
    top_return = result_df.sort_values(
        "total_return",
        ascending=False
    ).head(15)

    for _, row in top_return.iterrows():
        print(
            f"动量{int(row['momentum_window'])}日 | "
            f"MA{int(row['ma_window'])} | "
            f"{row['rebalance_mode']} | "
            f"累计收益：{row['total_return']:.2%} | "
            f"年化：{row['annual_return']:.2%} | "
            f"最大回撤：{row['max_drawdown']:.2%} | "
            f"收益回撤比：{row['return_drawdown_ratio']:.2f} | "
            f"空仓：{row['cash_ratio']:.2%} | "
            f"换仓：{int(row['turnover_count'])}"
        )

    print()
    print("===== 按收益回撤比排序 Top 15 =====")
    top_ratio = result_df.dropna(subset=["return_drawdown_ratio"]).sort_values(
        "return_drawdown_ratio",
        ascending=False
    ).head(15)

    for _, row in top_ratio.iterrows():
        print(
            f"动量{int(row['momentum_window'])}日 | "
            f"MA{int(row['ma_window'])} | "
            f"{row['rebalance_mode']} | "
            f"累计收益：{row['total_return']:.2%} | "
            f"年化：{row['annual_return']:.2%} | "
            f"最大回撤：{row['max_drawdown']:.2%} | "
            f"收益回撤比：{row['return_drawdown_ratio']:.2f} | "
            f"空仓：{row['cash_ratio']:.2%} | "
            f"换仓：{int(row['turnover_count'])}"
        )

    print()
    print("===== 最大回撤小于20%的组合，按累计收益排序 Top 15 =====")
    low_dd = result_df[result_df["max_drawdown"] > -0.20].sort_values(
        "total_return",
        ascending=False
    ).head(15)

    if len(low_dd) == 0:
        print("没有最大回撤小于20%的组合。")
    else:
        for _, row in low_dd.iterrows():
            print(
                f"动量{int(row['momentum_window'])}日 | "
                f"MA{int(row['ma_window'])} | "
                f"{row['rebalance_mode']} | "
                f"累计收益：{row['total_return']:.2%} | "
                f"年化：{row['annual_return']:.2%} | "
                f"最大回撤：{row['max_drawdown']:.2%} | "
                f"收益回撤比：{row['return_drawdown_ratio']:.2f} | "
                f"空仓：{row['cash_ratio']:.2%} | "
                f"换仓：{int(row['turnover_count'])}"
            )


if __name__ == "__main__":
    main()