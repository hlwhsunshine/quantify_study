import pandas as pd
import numpy as np
from pathlib import Path


# =========================
# 基础配置
# =========================

DATA_PATH = "data/etf_close_clean.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_ETFS = [
    "hs300",
    "zz500",
    "cyb",
    "kc50",
    "securities",
    "consumer",
    "medicine",
    "new_energy",
    "dividend",
]

MARKET_ETF = "hs300"
MOMENTUM_WINDOW = 10
MA_WINDOW = 200

BUY_COST = 0.001
SELL_COST = 0.001

TRADING_DAYS = 252


# v1.3 当前重点候选
CANDIDATES = [
    {
        "candidate": "Top3_4.00",
        "top_n": 3,
        "momentum_threshold": 0.04,
    },
    {
        "candidate": "Top4_3.75",
        "top_n": 4,
        "momentum_threshold": 0.0375,
    },
    {
        "candidate": "Top4_4.00",
        "top_n": 4,
        "momentum_threshold": 0.04,
    },
    {
        "candidate": "Top1_4.00",
        "top_n": 1,
        "momentum_threshold": 0.04,
    },
]


# =========================
# 工具函数
# =========================

def load_price_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    else:
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")

    needed_cols = list(set(ALL_ETFS + [MARKET_ETF]))
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少这些列: {missing}")

    return df


def get_month_end_signal_dates(index: pd.DatetimeIndex):
    s = pd.Series(index, index=index)
    month_end_dates = s.groupby(s.index.to_period("M")).max().tolist()
    return month_end_dates


def calc_max_drawdown(nav: pd.Series):
    peak = nav.cummax()
    drawdown = nav / peak - 1
    max_dd = drawdown.min()
    end_date = drawdown.idxmin()
    start_date = nav.loc[:end_date].idxmax()
    return max_dd, start_date, end_date


def calc_metrics(nav: pd.Series, daily_ret: pd.Series):
    total_days = len(nav)
    years = total_days / TRADING_DAYS

    final_value = nav.iloc[-1]
    cumulative_return = final_value - 1
    annual_return = final_value ** (1 / years) - 1

    annual_volatility = daily_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = annual_return / annual_volatility if annual_volatility != 0 else np.nan

    max_dd, dd_start, dd_end = calc_max_drawdown(nav)
    calmar = annual_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "final_value": final_value,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "max_dd_start": dd_start,
        "max_dd_end": dd_end,
    }


def run_backtest(close: pd.DataFrame, invest_pool, top_n, momentum_threshold):
    dates = close.index

    ret = close[invest_pool].pct_change().fillna(0)
    momentum = close[invest_pool].pct_change(MOMENTUM_WINDOW)

    market_ma = close[MARKET_ETF].rolling(MA_WINDOW).mean()

    month_end_dates = get_month_end_signal_dates(dates)

    # 生成“下一个交易日执行”的目标仓位
    target_weights_by_exec_date = {}

    for signal_date in month_end_dates:
        if signal_date not in dates:
            continue

        pos = dates.get_loc(signal_date)
        if pos + 1 >= len(dates):
            continue

        exec_date = dates[pos + 1]

        target = pd.Series(0.0, index=invest_pool)

        # MA200 市场过滤
        market_price = close.loc[signal_date, MARKET_ETF]
        market_ma_value = market_ma.loc[signal_date]

        if pd.isna(market_ma_value) or market_price <= market_ma_value:
            target_weights_by_exec_date[exec_date] = target
            continue

        mom = momentum.loc[signal_date].dropna()
        mom = mom[mom > momentum_threshold]
        mom = mom.sort_values(ascending=False)

        selected = mom.head(top_n).index.tolist()

        if len(selected) > 0:
            equal_weight = 1.0 / len(selected)
            target.loc[selected] = equal_weight

        target_weights_by_exec_date[exec_date] = target

    nav = []
    daily_returns = []
    cash_ratios = []
    holding_counts = []
    turnover_records = []
    total_cost_rate = 0.0
    rebalance_days = 0

    value = 1.0
    weights = pd.Series(0.0, index=invest_pool)

    for i, date in enumerate(dates):
        if i == 0:
            nav.append(value)
            daily_returns.append(0.0)
            cash_ratios.append(1.0)
            holding_counts.append(0)
            continue

        # 月度调仓：当天开盘前按目标仓位切换，然后承受当天收益
        if date in target_weights_by_exec_date:
            target = target_weights_by_exec_date[date]

            buy_turnover = (target - weights).clip(lower=0).sum()
            sell_turnover = (weights - target).clip(lower=0).sum()

            cost_rate = buy_turnover * BUY_COST + sell_turnover * SELL_COST

            if buy_turnover + sell_turnover > 1e-12:
                rebalance_days += 1

            total_cost_rate += cost_rate
            value *= (1 - cost_rate)

            turnover_records.append({
                "date": date,
                "buy_turnover": buy_turnover,
                "sell_turnover": sell_turnover,
                "cost_rate": cost_rate,
                "holding_count_after_rebalance": int((target > 0).sum()),
            })

            weights = target.copy()

        # 当天收益
        asset_ret = ret.loc[date]
        portfolio_ret = (weights * asset_ret).sum()

        old_value = value
        value *= (1 + portfolio_ret)

        # 持仓权重随价格涨跌自然漂移
        if value > 0:
            gross_asset_weights = weights * (1 + asset_ret)
            weights = gross_asset_weights / (1 + portfolio_ret)

        nav.append(value)
        daily_returns.append(value / old_value - 1)

        cash_ratio = 1 - weights.sum()
        cash_ratios.append(cash_ratio)
        holding_counts.append(int((weights > 1e-8).sum()))

    nav = pd.Series(nav, index=dates, name="nav")
    daily_returns = pd.Series(daily_returns, index=dates, name="daily_return")

    metrics = calc_metrics(nav, daily_returns)

    metrics.update({
        "cash_ratio": np.mean(cash_ratios),
        "avg_holding_count": np.mean(holding_counts),
        "avg_holding_count_when_invested": (
            np.mean([x for x in holding_counts if x > 0])
            if any(x > 0 for x in holding_counts)
            else 0
        ),
        "rebalance_days": rebalance_days,
        "total_cost_rate": total_cost_rate,
    })

    return metrics


def build_pool_tests():
    tests = []

    # 完整 ETF 池
    tests.append({
        "pool_test": "full_pool",
        "invest_pool": ALL_ETFS.copy(),
    })

    # 逐个删除
    for etf in ALL_ETFS:
        pool = [x for x in ALL_ETFS if x != etf]
        tests.append({
            "pool_test": f"without_{etf}",
            "invest_pool": pool,
        })

    # 几组重点删除测试
    custom_tests = {
        "without_growth_group": ["cyb", "kc50", "new_energy", "securities"],
        "without_defensive_group": ["consumer", "medicine", "dividend"],
        "broad_index_only": ["hs300", "zz500"],
        "broad_plus_dividend": ["hs300", "zz500", "dividend"],
        "no_sector_etf": ["hs300", "zz500", "dividend"],
    }

    for name, keep_or_remove in custom_tests.items():
        if name.startswith("without_"):
            pool = [x for x in ALL_ETFS if x not in keep_or_remove]
        else:
            pool = keep_or_remove

        tests.append({
            "pool_test": name,
            "invest_pool": pool,
        })

    return tests


def main():
    close = load_price_data(DATA_PATH)

    results = []
    pool_tests = build_pool_tests()

    for candidate in CANDIDATES:
        for pool_test in pool_tests:
            invest_pool = pool_test["invest_pool"]

            # 如果投资池为空，跳过
            if len(invest_pool) == 0:
                continue

            metrics = run_backtest(
                close=close,
                invest_pool=invest_pool,
                top_n=candidate["top_n"],
                momentum_threshold=candidate["momentum_threshold"],
            )

            row = {
                "candidate": candidate["candidate"],
                "top_n": candidate["top_n"],
                "momentum_threshold": candidate["momentum_threshold"],
                "pool_test": pool_test["pool_test"],
                "invest_pool": ",".join(invest_pool),
                "pool_size": len(invest_pool),
            }
            row.update(metrics)

            results.append(row)

    result_df = pd.DataFrame(results)

    # 百分比格式列保留原始小数，方便后续分析
    result_df = result_df.sort_values(
        by=["candidate", "final_value"],
        ascending=[True, False],
    )

    output_path = OUTPUT_DIR / "v14_etf_pool_sensitivity.csv"
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n========== v1.4 ETF 池敏感性测试完成 ==========")
    print(f"结果已保存: {output_path}")

    display_cols = [
        "candidate",
        "pool_test",
        "pool_size",
        "final_value",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "cash_ratio",
        "avg_holding_count_when_invested",
        "rebalance_days",
        "total_cost_rate",
    ]

    print("\n========== 结果预览 ==========")
    print(result_df[display_cols].to_string(index=False))

    print("\n========== 每个候选版本的 full_pool 基准 ==========")
    baseline = result_df[result_df["pool_test"] == "full_pool"]
    print(baseline[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()