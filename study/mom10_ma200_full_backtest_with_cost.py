#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
10日动量 + 沪深300 MA200趋势过滤 + 月度调仓
加入手续费和滑点后的完整回测

策略规则：
1. 每月最后一个交易日收盘后生成信号
2. 计算每只 ETF 过去10个交易日动量
3. 选择动量最高的 ETF
4. 如果最高动量 <= 0，则空仓
5. 如果沪深300ETF收盘价 <= 沪深300ETF MA200，则空仓
6. 否则买入动量最高 ETF
7. 信号从下一个交易日开始生效，避免未来函数

成本规则：
1. cash -> ETF：扣买入成本
2. ETF -> cash：扣卖出成本
3. ETF A -> ETF B：扣卖出成本 + 买入成本
4. 持仓不变：不扣成本

输入：
data/etf_close_clean.csv

输出：
1. data/monthly_signal_mom10_ma200_cost.csv
2. data/backtest_daily_mom10_ma200_cost.csv
3. data/benchmark_compare_mom10_ma200_cost.csv
4. data/yearly_return_mom10_ma200_cost.csv
"""

import os
import pandas as pd


# =========================
# 配置区
# =========================

DATA_DIR = "data"

CLOSE_FILE = os.path.join(DATA_DIR, "etf_close_clean.csv")

SIGNAL_FILE = os.path.join(DATA_DIR, "monthly_signal_mom10_ma200_cost.csv")
BACKTEST_FILE = os.path.join(DATA_DIR, "backtest_daily_mom10_ma200_cost.csv")
COMPARE_FILE = os.path.join(DATA_DIR, "benchmark_compare_mom10_ma200_cost.csv")
YEARLY_FILE = os.path.join(DATA_DIR, "yearly_return_mom10_ma200_cost.csv")

BENCHMARK_COL = "hs300"

MOMENTUM_WINDOW = 10
MA_WINDOW = 200

# 单边交易成本
BUY_COST = 0.001   # 0.10%
SELL_COST = 0.001  # 0.10%


# =========================
# 工具函数
# =========================

def calc_annual_return(final_nav, days):
    return final_nav ** (252 / days) - 1


def calc_max_drawdown(nav_series):
    cummax_nav = nav_series.cummax()
    drawdown = nav_series / cummax_nav - 1
    return drawdown.min()


# =========================
# 生成月度信号
# =========================

def generate_monthly_signal(close_df, etf_cols):
    df = close_df[["date"] + etf_cols].copy()

    # 计算10日动量
    momentum_df = df[["date"]].copy()

    for col in etf_cols:
        momentum_df[col] = df[col] / df[col].shift(MOMENTUM_WINDOW) - 1

    # 计算沪深300 MA200
    trend_df = close_df[["date", BENCHMARK_COL]].copy()
    trend_df["hs300_ma200"] = trend_df[BENCHMARK_COL].rolling(MA_WINDOW).mean()

    # 合并动量和趋势数据
    # momentum_df 里的 hs300 是动量
    # trend_df 里的 hs300 是收盘价
    # merge 后收盘价列会变成 hs300_close
    signal_base = pd.merge(
        momentum_df,
        trend_df,
        on="date",
        how="inner",
        suffixes=("", "_close")
    )

    # 每月最后一个交易日
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
# 成本计算
# =========================

def calc_trade_cost(prev_position, current_position):
    """
    根据前一日持仓和当前持仓计算交易成本。

    返回值是当天需要扣掉的收益率成本。
    """
    if prev_position == current_position:
        return 0.0

    # cash -> ETF
    if prev_position == "cash" and current_position != "cash":
        return BUY_COST

    # ETF -> cash
    if prev_position != "cash" and current_position == "cash":
        return SELL_COST

    # ETF A -> ETF B
    if prev_position != "cash" and current_position != "cash":
        return SELL_COST + BUY_COST

    return 0.0


# =========================
# 回测
# =========================

def run_backtest_with_cost(close_df, signal_df, etf_cols):
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

    strategy_returns_before_cost = []
    trade_costs = []
    strategy_returns_after_cost = []

    prev_position = "cash"

    for _, row in daily_df.iterrows():
        current_position = row["selected_etf"]

        if current_position == "cash":
            raw_return = 0.0

        elif current_position not in etf_cols:
            raise ValueError(f"selected_etf 不在 ETF 列中：{current_position}")

        else:
            raw_return = row[current_position]

        trade_cost = calc_trade_cost(prev_position, current_position)

        net_return = raw_return - trade_cost

        strategy_returns_before_cost.append(raw_return)
        trade_costs.append(trade_cost)
        strategy_returns_after_cost.append(net_return)

        prev_position = current_position

    daily_df["strategy_return_before_cost"] = strategy_returns_before_cost
    daily_df["trade_cost"] = trade_costs
    daily_df["strategy_return"] = strategy_returns_after_cost

    daily_df["strategy_nav"] = (1 + daily_df["strategy_return"]).cumprod()

    daily_df["cummax_nav"] = daily_df["strategy_nav"].cummax()
    daily_df["drawdown"] = daily_df["strategy_nav"] / daily_df["cummax_nav"] - 1

    return daily_df


# =========================
# 基准对比
# =========================

def build_benchmark_compare(close_df, backtest_df):
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
# 年度分析
# =========================

def build_yearly_analysis(compare_df):
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

    return pd.DataFrame(results)


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

    signal_df = generate_monthly_signal(close_df, etf_cols)
    signal_df.to_csv(SIGNAL_FILE, index=False, encoding="utf-8-sig")

    backtest_df = run_backtest_with_cost(close_df, signal_df, etf_cols)
    backtest_df.to_csv(BACKTEST_FILE, index=False, encoding="utf-8-sig")

    compare_df = build_benchmark_compare(close_df, backtest_df)
    compare_df.to_csv(COMPARE_FILE, index=False, encoding="utf-8-sig")

    yearly_df = build_yearly_analysis(compare_df)
    yearly_df.to_csv(YEARLY_FILE, index=False, encoding="utf-8-sig")

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

    trade_days = (backtest_df["trade_cost"] > 0).sum()
    total_trade_cost = backtest_df["trade_cost"].sum()

    print("===== 10日动量 + 沪深300MA200 完整回测结果：含交易成本 =====")
    print(f"开始日期：{start_date.date()}")
    print(f"结束日期：{end_date.date()}")
    print(f"交易日数量：{days}")
    print()

    print("===== 成本设置 =====")
    print(f"买入成本：{BUY_COST:.2%}")
    print(f"卖出成本：{SELL_COST:.2%}")
    print()

    print("===== 策略表现：含成本 =====")
    print(f"策略最终净值：{strategy_final_nav:.4f}")
    print(f"策略累计收益：{strategy_total_return:.2%}")
    print(f"策略年化收益：{strategy_annual_return:.2%}")
    print(f"策略最大回撤：{strategy_max_drawdown:.2%}")
    print(f"空仓天数：{cash_days}")
    print(f"空仓比例：{cash_ratio:.2%}")
    print(f"发生交易成本的天数：{trade_days}")
    print(f"累计交易成本近似值：{total_trade_cost:.2%}")
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

    print("===== 年度收益分析：含成本 =====")
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