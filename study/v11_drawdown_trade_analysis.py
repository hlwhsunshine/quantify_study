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
    costs = pd.Series(index=close.index, data=0.0)

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
        costs.loc[date] = cost

        prev_position = current_position

    equity = (1 + strategy_returns).cumprod()

    return strategy_returns, equity, costs


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

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

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


def analyze_max_drawdown(equity):
    """
    分析全局最大回撤区间。

    peak_date:
        最大回撤对应的前期净值高点日期。

    trough_date:
        最大回撤最低点日期。
    """

    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    trough_date = drawdown.idxmin()
    max_drawdown = drawdown.loc[trough_date]

    # 在 trough_date 之前，找到对应的历史最高点
    equity_before_trough = equity.loc[:trough_date]
    peak_date = equity_before_trough.idxmax()

    peak_equity = equity.loc[peak_date]
    trough_equity = equity.loc[trough_date]

    result = {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "peak_equity": peak_equity,
        "trough_equity": trough_equity,
        "max_drawdown": max_drawdown,
    }

    return result, drawdown


def build_trade_detail(equity, position):
    """
    构建每段持仓明细。

    注意：
    这里的 trade_return 是从持仓开始日净值到持仓结束日净值的变化。
    它已经包含了回测里的买卖成本影响。
    """

    trades = []

    current_holding = None
    entry_date = None

    dates = list(position.index)

    for i, date in enumerate(dates):
        holding = position.loc[date]

        if current_holding is None:
            current_holding = holding
            entry_date = date
            continue

        # 持仓发生变化，上一段持仓结束
        if holding != current_holding:
            exit_date = dates[i - 1]

            entry_equity = equity.loc[entry_date]
            exit_equity = equity.loc[exit_date]

            if entry_equity == 0:
                trade_return = np.nan
            else:
                trade_return = exit_equity / entry_equity - 1

            holding_days = (exit_date - entry_date).days

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "holding": current_holding,
                "holding_days": holding_days,
                "entry_equity": entry_equity,
                "exit_equity": exit_equity,
                "trade_return": trade_return,
            })

            current_holding = holding
            entry_date = date

    # 处理最后一段持仓
    exit_date = dates[-1]

    entry_equity = equity.loc[entry_date]
    exit_equity = equity.loc[exit_date]

    if entry_equity == 0:
        trade_return = np.nan
    else:
        trade_return = exit_equity / entry_equity - 1

    holding_days = (exit_date - entry_date).days

    trades.append({
        "entry_date": entry_date,
        "exit_date": exit_date,
        "holding": current_holding,
        "holding_days": holding_days,
        "entry_equity": entry_equity,
        "exit_equity": exit_equity,
        "trade_return": trade_return,
    })

    trade_df = pd.DataFrame(trades)

    return trade_df


def add_trade_year_info(trade_df):
    trade_df = trade_df.copy()
    trade_df["entry_year"] = trade_df["entry_date"].dt.year
    trade_df["exit_year"] = trade_df["exit_date"].dt.year
    return trade_df


def main():
    close = load_data()

    print("数据读取完成")
    print("数据起始日期:", close.index[0].date())
    print("数据结束日期:", close.index[-1].date())
    print("数据行数:", len(close))

    position = generate_daily_position(close)

    strategy_returns, equity, costs = run_backtest(
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

    max_dd_info, drawdown = analyze_max_drawdown(equity)

    trade_df = build_trade_detail(
        equity=equity,
        position=position,
    )

    trade_df = add_trade_year_info(trade_df)

    # 保存数据
    total_df = pd.DataFrame([total_metrics])
    max_dd_df = pd.DataFrame([max_dd_info])

    total_df.to_csv(
        "data/v11_drawdown_total_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    max_dd_df.to_csv(
        "data/v11_max_drawdown_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    trade_df.to_csv(
        "data/v11_trade_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    drawdown_df = pd.DataFrame({
        "date": drawdown.index,
        "equity": equity.values,
        "drawdown": drawdown.values,
        "position": position.values,
    })

    drawdown_df.to_csv(
        "data/v11_daily_drawdown.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 打印总体结果
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

    # 打印最大回撤信息
    print("\n===== v1.1 最大回撤区间 =====\n")

    display_dd = max_dd_df.copy()

    display_dd["peak_date"] = display_dd["peak_date"].dt.date
    display_dd["trough_date"] = display_dd["trough_date"].dt.date
    display_dd["peak_equity"] = display_dd["peak_equity"].apply(lambda x: f"{x:.4f}")
    display_dd["trough_equity"] = display_dd["trough_equity"].apply(lambda x: f"{x:.4f}")
    display_dd["max_drawdown"] = display_dd["max_drawdown"].apply(lambda x: f"{x:.2%}")

    print(display_dd.to_string(index=False))

    # 打印全部交易明细
    print("\n===== v1.1 交易明细 =====\n")

    display_trade = trade_df.copy()

    display_trade["entry_date"] = display_trade["entry_date"].dt.date
    display_trade["exit_date"] = display_trade["exit_date"].dt.date
    display_trade["entry_equity"] = display_trade["entry_equity"].apply(lambda x: f"{x:.4f}")
    display_trade["exit_equity"] = display_trade["exit_equity"].apply(lambda x: f"{x:.4f}")
    display_trade["trade_return"] = display_trade["trade_return"].apply(lambda x: f"{x:.2%}")

    print(display_trade.to_string(index=False))

    # 打印亏损最大的 10 段持仓
    print("\n===== 亏损最大的 10 段持仓 =====\n")

    worst_trades = trade_df.sort_values("trade_return").head(10).copy()

    worst_trades["entry_date"] = worst_trades["entry_date"].dt.date
    worst_trades["exit_date"] = worst_trades["exit_date"].dt.date
    worst_trades["entry_equity"] = worst_trades["entry_equity"].apply(lambda x: f"{x:.4f}")
    worst_trades["exit_equity"] = worst_trades["exit_equity"].apply(lambda x: f"{x:.4f}")
    worst_trades["trade_return"] = worst_trades["trade_return"].apply(lambda x: f"{x:.2%}")

    print(worst_trades.to_string(index=False))

    # 打印盈利最大的 10 段持仓
    print("\n===== 盈利最大的 10 段持仓 =====\n")

    best_trades = trade_df.sort_values("trade_return", ascending=False).head(10).copy()

    best_trades["entry_date"] = best_trades["entry_date"].dt.date
    best_trades["exit_date"] = best_trades["exit_date"].dt.date
    best_trades["entry_equity"] = best_trades["entry_equity"].apply(lambda x: f"{x:.4f}")
    best_trades["exit_equity"] = best_trades["exit_equity"].apply(lambda x: f"{x:.4f}")
    best_trades["trade_return"] = best_trades["trade_return"].apply(lambda x: f"{x:.2%}")

    print(best_trades.to_string(index=False))

    print("\n结果已保存到：")
    print("data/v11_drawdown_total_metrics.csv")
    print("data/v11_max_drawdown_info.csv")
    print("data/v11_trade_detail.csv")
    print("data/v11_daily_drawdown.csv")


if __name__ == "__main__":
    main()