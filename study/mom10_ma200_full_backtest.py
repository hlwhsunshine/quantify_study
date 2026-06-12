#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10日动量 + 沪深300 MA200趋势过滤 + 月度调仓 完整回测

策略规则：
1. 每月最后一个交易日收盘后生成信号
2. 计算每只 ETF 过去 10 个交易日动量
3. 选择动量最高的 ETF
4. 如果最高动量 <= 0，则空仓
5. 如果沪深300ETF收盘价 <= 沪深300ETF MA200，则空仓
6. 否则买入动量最高 ETF
7. 信号从下一个交易日开始生效，避免未来函数

输入：
data/etf_close.csv

输出：
1. data/monthly_signal_mom10_ma200.csv
2. data/backtest_daily_mom10_ma200.csv
3. data/benchmark_compare_mom10_ma200.csv
4. data/yearly_return_mom10_ma200.csv

暂不考虑：
1. 手续费
2. 滑点
3. 成交失败
4. 跟踪误差
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close_clean.csv")

SIGNAL_FILE = os.path.join(DATA_DIR, "monthly_signal_mom10_ma200.csv")
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_daily_mom10_ma200.csv")
COMPARE_FILE = os.path.join(DATA_DIR, "benchmark_compare_mom10_ma200.csv")
YEARLY_FILE = os.path.join(DATA_DIR, "yearly_return_mom10_ma200.csv")

BENCHMARK_COL = "hs300"

MOMENTUM_WINDOW = 10
MA_WINDOW = 200


# =========================
# 工具函数
# =========================

def calc_max_drawdown(nav_series):
    """
    根据净值序列计算最大回撤。
    """
    cummax_nav = nav_series.cummax()
    drawdown = nav_series / cummax_nav - 1
    return drawdown.min()


def calc_annual_return(final_nav, days):
    """
    按 252 个交易日计算年化收益。
    """
    return final_nav ** (252 / days) - 1


# =========================
# 生成月度信号
# =========================

def generate_monthly_signal(close_df, etf_cols):
    """
    生成 10日动量 + MA200趋势过滤 的月度信号。
    """
    df = close_df[["date"] + etf_cols].copy()

    # 计算10日动量
    momentum_df = df[["date"]].copy()

    for col in etf_cols:
        momentum_df[col] = df[col] / df[col].shift(MOMENTUM_WINDOW) - 1

    # 计算沪深300 MA200
    trend_df = close_df[["date", BENCHMARK_COL]].copy()
    trend_df["hs300_ma200"] = trend_df[BENCHMARK_COL].rolling(MA_WINDOW).mean()

    # 合并动量和趋势过滤数据
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

    # 取每月最后一个交易日
    signal_base["year_month"] = signal_base["date"].dt.to_period("M")
    month_end_df = signal_base.groupby("year_month").tail(1).copy()

    signals = []

    for _, row in month_end_df.iterrows():
        date = row["date"]

        momentum_values = row[etf_cols]
        valid_momentum_values = momentum_values.dropna()

        if valid_momentum_values.empty:
            best_etf = None
            best_momentum = None
        else:
            best_etf = valid_momentum_values.idxmax()
            best_momentum = valid_momentum_values.max()

        # 关键修复：
        # 这里必须使用 hs300_close，而不是 hs300
        # 因为 hs300 是动量，hs300_close 才是收盘价
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

# =========================
# 回测
# =========================

def run_backtest(close_df, signal_df, etf_cols):
    """
    根据月度信号进行每日回测。
    """
    return_df = close_df[["date"] + etf_cols].copy()

    # 计算每日收益率
    for col in etf_cols:
        return_df[col] = return_df[col].pct_change(fill_method=None)

    return_df[etf_cols] = return_df[etf_cols].fillna(0)

    # 月末信号从下一个交易日生效
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


# =========================
# 基准对比
# =========================

def build_benchmark_compare(close_df, backtest_df):
    """
    生成策略和沪深300ETF基准的净值、回撤对比。
    """
    benchmark_df = close_df[["date", BENCHMARK_COL]].copy()

    benchmark_df["benchmark_return"] = benchmark_df[BENCHMARK_COL].pct_change(fill_method=None)
    benchmark_df["benchmark_return"] = benchmark_df["benchmark_return"].fillna(0)
    benchmark_df["benchmark_nav"] = (1 + benchmark_df["benchmark_return"]).cumprod()

    result = pd.merge(
        backtest_df[["date", "strategy_nav", "drawdown"]],
        benchmark_df[["date", "benchmark_nav"]],
        on="date",
        how="inner"
    )

    result = result.rename(columns={
        "drawdown": "strategy_drawdown"
    })

    result["benchmark_cummax"] = result["benchmark_nav"].cummax()
    result["benchmark_drawdown"] = result["benchmark_nav"] / result["benchmark_cummax"] - 1
    result = result.drop(columns=["benchmark_cummax"])

    return result


# =========================
# 年度收益分析
# =========================

def build_yearly_analysis(compare_df):
    """
    生成年度收益分析。

    这里的年度最大回撤是真正的年内最大回撤：
    每年都从该年第一天净值重新归一化计算。
    """
    df = compare_df.copy()
    df["year"] = df["date"].dt.year

    results = []

    for year, group in df.groupby("year"):
        group = group.sort_values("date").copy()

        strategy_start_nav = group["strategy_nav"].iloc[0]
        strategy_end_nav = group["strategy_nav"].iloc[-1]

        benchmark_start_nav = group["benchmark_nav"].iloc[0]
        benchmark_end_nav = group["benchmark_nav"].iloc[-1]

        strategy_return = strategy_end_nav / strategy_start_nav - 1
        benchmark_return = benchmark_end_nav / benchmark_start_nav - 1
        excess_return = strategy_return - benchmark_return

        # 年内净值重新归一化
        strategy_year_nav = group["strategy_nav"] / strategy_start_nav
        benchmark_year_nav = group["benchmark_nav"] / benchmark_start_nav

        strategy_year_max_drawdown = calc_max_drawdown(strategy_year_nav)
        benchmark_year_max_drawdown = calc_max_drawdown(benchmark_year_nav)

        results.append({
            "year": year,
            "strategy_return": strategy_return,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "strategy_year_max_drawdown": strategy_year_max_drawdown,
            "benchmark_year_max_drawdown": benchmark_year_max_drawdown,
        })

    yearly_df = pd.DataFrame(results)

    return yearly_df


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

    # 1. 生成信号
    signal_df = generate_monthly_signal(close_df, etf_cols)
    signal_df.to_csv(SIGNAL_FILE, index=False, encoding="utf-8-sig")

    # 2. 回测
    backtest_df = run_backtest(close_df, signal_df, etf_cols)
    backtest_df.to_csv(BACKTEST_FILE, index=False, encoding="utf-8-sig")

    # 3. 基准对比
    compare_df = build_benchmark_compare(close_df, backtest_df)
    compare_df.to_csv(COMPARE_FILE, index=False, encoding="utf-8-sig")

    # 4. 年度收益
    yearly_df = build_yearly_analysis(compare_df)
    yearly_df.to_csv(YEARLY_FILE, index=False, encoding="utf-8-sig")

    # =========================
    # 汇总输出
    # =========================

    start_date = backtest_df["date"].iloc[0]
    end_date = backtest_df["date"].iloc[-1]
    days = len(backtest_df)

    strategy_final_nav = compare_df["strategy_nav"].iloc[-1]
    benchmark_final_nav = compare_df["benchmark_nav"].iloc[-1]

    strategy_total_return = strategy_final_nav - 1
    benchmark_total_return = benchmark_final_nav - 1

    strategy_annual_return = calc_annual_return(strategy_final_nav, days)
    benchmark_annual_return = calc_annual_return(benchmark_final_nav, days)

    strategy_max_drawdown = compare_df["strategy_drawdown"].min()
    benchmark_max_drawdown = compare_df["benchmark_drawdown"].min()

    cash_days = (backtest_df["selected_etf"] == "cash").sum()
    cash_ratio = cash_days / days

    print("===== 10日动量 + 沪深300MA200 完整回测结果 =====")
    print(f"开始日期：{start_date.date()}")
    print(f"结束日期：{end_date.date()}")
    print(f"交易日数量：{days}")
    print()

    print("===== 策略表现 =====")
    print(f"策略最终净值：{strategy_final_nav:.4f}")
    print(f"策略累计收益：{strategy_total_return:.2%}")
    print(f"策略年化收益：{strategy_annual_return:.2%}")
    print(f"策略最大回撤：{strategy_max_drawdown:.2%}")
    print(f"空仓天数：{cash_days}")
    print(f"空仓比例：{cash_ratio:.2%}")
    print()

    print("===== 基准表现：沪深300ETF =====")
    print(f"基准最终净值：{benchmark_final_nav:.4f}")
    print(f"基准累计收益：{benchmark_total_return:.2%}")
    print(f"基准年化收益：{benchmark_annual_return:.2%}")
    print(f"基准最大回撤：{benchmark_max_drawdown:.2%}")
    print()

    print("===== 信号原因统计 =====")
    print(signal_df["reason"].value_counts())
    print()

    print("===== 持仓天数统计 =====")
    print(backtest_df["selected_etf"].value_counts())
    print()

    print("===== 年度收益分析 =====")
    for _, row in yearly_df.iterrows():
        print(
            f"{int(row['year'])} | "
            f"策略：{row['strategy_return']:.2%} | "
            f"基准：{row['benchmark_return']:.2%} | "
            f"超额：{row['excess_return']:.2%} | "
            f"策略年内回撤：{row['strategy_year_max_drawdown']:.2%} | "
            f"基准年内回撤：{row['benchmark_year_max_drawdown']:.2%}"
        )

    print()
    print("===== 文件已保存 =====")
    print(f"月度信号：{SIGNAL_FILE}")
    print(f"每日回测：{BACKTEST_FILE}")
    print(f"基准对比：{COMPARE_FILE}")
    print(f"年度分析：{YEARLY_FILE}")


if __name__ == "__main__":
    main()