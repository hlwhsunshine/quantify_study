import pandas as pd
import numpy as np


DATA_PATH = "data/etf_close_clean.csv"

ETF_LIST = [
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

MOMENTUM_DAYS = 10
MARKET_FILTER_ETF = "hs300"
MARKET_MA_DAYS = 200

BUY_COST = 0.001
SELL_COST = 0.001


CASH_STRATEGIES = [
    {
        "strategy_name": "no_cash_filter",
        "use_momentum_cash": False,
        "momentum_threshold": None,
        "use_ma_cash": False,
    },
    {
        "strategy_name": "momentum_only",
        "use_momentum_cash": True,
        "momentum_threshold": 0.0,
        "use_ma_cash": False,
    },
    {
        "strategy_name": "ma_only",
        "use_momentum_cash": False,
        "momentum_threshold": None,
        "use_ma_cash": True,
    },
    {
        "strategy_name": "momentum_and_ma",
        "use_momentum_cash": True,
        "momentum_threshold": 0.0,
        "use_ma_cash": True,
    },
    {
        "strategy_name": "strict_momentum_1pct_and_ma",
        "use_momentum_cash": True,
        "momentum_threshold": 0.01,
        "use_ma_cash": True,
    },
    {
        "strategy_name": "strict_momentum_2pct_and_ma",
        "use_momentum_cash": True,
        "momentum_threshold": 0.02,
        "use_ma_cash": True,
    },
    {
        "strategy_name": "strict_momentum_3pct_and_ma",
        "use_momentum_cash": True,
        "momentum_threshold": 0.03,
        "use_ma_cash": True,
    },
]


def load_data():
    """
    读取清洗后的 ETF 收盘价数据。
    """
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")
    return df


def get_actual_month_end_dates(close):
    """
    获取每个月实际存在的最后一个交易日。
    使用 ME，兼容新版 pandas。
    """
    month_end_dates = close.resample("ME").last().index

    actual_month_end_dates = []

    for d in month_end_dates:
        available_dates = close.index[close.index <= d]

        if len(available_dates) > 0:
            actual_month_end_dates.append(available_dates[-1])

    return actual_month_end_dates


def generate_daily_position(
    close,
    use_momentum_cash,
    momentum_threshold,
    use_ma_cash,
):
    """
    根据不同空仓规则生成每日持仓。

    参数说明：
    use_momentum_cash:
        是否启用动量空仓。

    momentum_threshold:
        动量空仓阈值。
        例如 0.0 表示 best_mom <= 0 空仓。
        0.01 表示 best_mom <= 1% 空仓。

    use_ma_cash:
        是否启用 MA200 市场过滤空仓。
    """

    momentum = close[ETF_LIST].pct_change(MOMENTUM_DAYS)
    market_ma = close[MARKET_FILTER_ETF].rolling(MARKET_MA_DAYS).mean()

    actual_month_end_dates = get_actual_month_end_dates(close)

    signals = pd.Series(index=close.index, dtype="object")

    for d in actual_month_end_dates:
        mom_row = momentum.loc[d, ETF_LIST].dropna()

        if len(mom_row) == 0:
            signals.loc[d] = "cash"
            continue

        best_etf = mom_row.idxmax()
        best_mom = mom_row.max()

        should_cash = False

        # 动量空仓规则
        if use_momentum_cash:
            if best_mom <= momentum_threshold:
                should_cash = True

        # MA200 市场过滤规则
        if use_ma_cash:
            market_ma_value = market_ma.loc[d]

            # MA 数据不足时，默认先不通过市场过滤，保持谨慎空仓
            if pd.isna(market_ma_value):
                should_cash = True
            else:
                market_ok = close.loc[d, MARKET_FILTER_ETF] > market_ma_value

                if not market_ok:
                    should_cash = True

        if should_cash:
            signals.loc[d] = "cash"
        else:
            signals.loc[d] = best_etf

    # 月末信号，下一个交易日生效
    position = signals.shift(1).ffill().fillna("cash")

    return position


def run_backtest(close, position, buy_cost=BUY_COST, sell_cost=SELL_COST):
    """
    执行带交易成本的回测。
    """

    returns = close[ETF_LIST].pct_change().fillna(0)
    strategy_returns = pd.Series(index=close.index, data=0.0)

    prev_position = "cash"

    for date in close.index:
        current_position = position.loc[date]

        if current_position == "cash":
            daily_ret = 0.0
        else:
            daily_ret = returns.loc[date, current_position]

        cost = 0.0

        if current_position != prev_position:
            if prev_position != "cash":
                cost += sell_cost

            if current_position != "cash":
                cost += buy_cost

        strategy_returns.loc[date] = daily_ret - cost

        prev_position = current_position

    equity = (1 + strategy_returns).cumprod()

    return strategy_returns, equity


def calculate_metrics(strategy_returns, equity, position):
    """
    计算策略评价指标。
    """

    final_value = equity.iloc[-1]
    total_return = final_value - 1

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25

    if years <= 0:
        annual_return = np.nan
    else:
        annual_return = final_value ** (1 / years) - 1

    annual_vol = strategy_returns.std() * np.sqrt(252)

    if annual_vol == 0 or np.isnan(annual_vol):
        sharpe = np.nan
    else:
        sharpe = annual_return / annual_vol

    max_drawdown = (equity / equity.cummax() - 1).min()

    trade_count = (position != position.shift(1)).sum()

    cash_ratio = (position == "cash").mean()

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "cash_ratio": cash_ratio,
    }


def main():
    close = load_data()

    print("数据读取完成")
    print("数据起始日期:", close.index[0].date())
    print("数据结束日期:", close.index[-1].date())
    print("数据行数:", len(close))

    results = []

    for config in CASH_STRATEGIES:
        position = generate_daily_position(
            close=close,
            use_momentum_cash=config["use_momentum_cash"],
            momentum_threshold=config["momentum_threshold"],
            use_ma_cash=config["use_ma_cash"],
        )

        strategy_returns, equity = run_backtest(
            close=close,
            position=position,
            buy_cost=BUY_COST,
            sell_cost=SELL_COST,
        )

        metrics = calculate_metrics(
            strategy_returns=strategy_returns,
            equity=equity,
            position=position,
        )

        results.append({
            "strategy_name": config["strategy_name"],
            "use_momentum_cash": config["use_momentum_cash"],
            "momentum_threshold": config["momentum_threshold"],
            "use_ma_cash": config["use_ma_cash"],
            **metrics,
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        "data/cash_filter_strategy_result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_df = result_df.copy()

    percent_cols = [
        "momentum_threshold",
        "total_return",
        "annual_return",
        "annual_vol",
        "max_drawdown",
        "cash_ratio",
    ]

    for col in percent_cols:
        display_df[col] = display_df[col].apply(
            lambda x: "" if pd.isna(x) else f"{x:.2%}"
        )

    display_df["final_value"] = display_df["final_value"].apply(lambda x: f"{x:.4f}")
    display_df["sharpe"] = display_df["sharpe"].apply(lambda x: f"{x:.2f}")

    print("\n===== 空仓策略测试结果 =====\n")
    print(display_df.to_string(index=False))

    print("\n结果已保存到：data/cash_filter_strategy_result.csv")


if __name__ == "__main__":
    main()