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
MOMENTUM_THRESHOLD = 0.04

MARKET_FILTER_ETF = "hs300"
MARKET_MA_DAYS = 200

BUY_COST = 0.001
SELL_COST = 0.001


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")
    return df


def get_actual_month_end_dates(close):
    month_end_dates = close.resample("ME").last().index

    actual_month_end_dates = []

    for d in month_end_dates:
        available_dates = close.index[close.index <= d]

        if len(available_dates) > 0:
            actual_month_end_dates.append(available_dates[-1])

    return actual_month_end_dates


def generate_daily_position(close):
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

        # v1.1：最高 10 日动量 <= 4% 时空仓
        if best_mom <= MOMENTUM_THRESHOLD:
            should_cash = True

        # MA200 市场过滤
        market_ma_value = market_ma.loc[d]

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

    position = signals.shift(1).ffill().fillna("cash")

    return position


def run_backtest(close, position, buy_cost=BUY_COST, sell_cost=SELL_COST):
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


def calculate_total_metrics(strategy_returns, equity, position):
    final_value = equity.iloc[-1]
    total_return = final_value - 1

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25

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


def calculate_yearly_metrics(strategy_returns, equity, position):
    """
    计算每年收益、年度最大回撤、年度交易次数、年度空仓比例。
    """

    results = []

    years = sorted(strategy_returns.index.year.unique())

    for year in years:
        year_mask = strategy_returns.index.year == year

        year_returns = strategy_returns.loc[year_mask]
        year_equity_global = equity.loc[year_mask]
        year_position = position.loc[year_mask]

        if len(year_returns) == 0:
            continue

        # 年度收益：这一年内从 1 开始重新累乘
        year_equity = (1 + year_returns).cumprod()
        yearly_return = year_equity.iloc[-1] - 1

        # 年度最大回撤：只看该年度内部
        yearly_max_drawdown = (year_equity / year_equity.cummax() - 1).min()

        # 年度波动
        yearly_vol = year_returns.std() * np.sqrt(252)

        # 年度交易次数
        # 注意：为了统计年初第一天是否发生换仓，需要和全局 position 的前一天比较
        year_dates = year_position.index
        trade_count = 0

        for d in year_dates:
            current_pos = position.loc[d]
            previous_dates = position.index[position.index < d]

            if len(previous_dates) == 0:
                prev_pos = "cash"
            else:
                prev_pos = position.loc[previous_dates[-1]]

            if current_pos != prev_pos:
                trade_count += 1

        cash_ratio = (year_position == "cash").mean()

        results.append({
            "year": year,
            "yearly_return": yearly_return,
            "yearly_vol": yearly_vol,
            "yearly_max_drawdown": yearly_max_drawdown,
            "trade_count": trade_count,
            "cash_ratio": cash_ratio,
        })

    return pd.DataFrame(results)


def main():
    close = load_data()

    print("数据读取完成")
    print("数据起始日期:", close.index[0].date())
    print("数据结束日期:", close.index[-1].date())
    print("数据行数:", len(close))

    position = generate_daily_position(close)

    strategy_returns, equity = run_backtest(
        close=close,
        position=position,
        buy_cost=BUY_COST,
        sell_cost=SELL_COST,
    )

    total_metrics = calculate_total_metrics(
        strategy_returns=strategy_returns,
        equity=equity,
        position=position,
    )

    yearly_df = calculate_yearly_metrics(
        strategy_returns=strategy_returns,
        equity=equity,
        position=position,
    )

    yearly_df.to_csv(
        "data/v11_yearly_return_analysis.csv",
        index=False,
        encoding="utf-8-sig",
    )

    total_df = pd.DataFrame([total_metrics])
    total_df.to_csv(
        "data/v11_total_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== v1.1 总体结果 =====\n")

    display_total = total_df.copy()

    percent_cols_total = [
        "total_return",
        "annual_return",
        "annual_vol",
        "max_drawdown",
        "cash_ratio",
    ]

    for col in percent_cols_total:
        display_total[col] = display_total[col].apply(lambda x: f"{x:.2%}")

    display_total["final_value"] = display_total["final_value"].apply(lambda x: f"{x:.4f}")
    display_total["sharpe"] = display_total["sharpe"].apply(lambda x: f"{x:.2f}")

    print(display_total.to_string(index=False))

    print("\n===== v1.1 年度收益分解 =====\n")

    display_yearly = yearly_df.copy()

    percent_cols_yearly = [
        "yearly_return",
        "yearly_vol",
        "yearly_max_drawdown",
        "cash_ratio",
    ]

    for col in percent_cols_yearly:
        display_yearly[col] = display_yearly[col].apply(lambda x: f"{x:.2%}")

    print(display_yearly.to_string(index=False))

    print("\n结果已保存到：")
    print("data/v11_yearly_return_analysis.csv")
    print("data/v11_total_metrics.csv")


if __name__ == "__main__":
    main()