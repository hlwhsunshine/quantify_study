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

STOP_LOSS_LIST = [
    None,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
]


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


def generate_monthly_signal(close):
    """
    生成 v1.1 月末信号。

    信号仍然只在月末产生：
    - 满足条件则信号为某只 ETF
    - 不满足则 cash

    止损逻辑不在这里处理，而是在日度回测中处理。
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

        if best_mom <= MOMENTUM_THRESHOLD:
            should_cash = True

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

    # 月末信号下一个交易日生效
    base_position = signals.shift(1).ffill().fillna("cash")

    return base_position


def run_backtest_with_stop_loss(close, base_position, stop_loss):
    """
    单 ETF 止损回测。

    逻辑：
    1. base_position 是原 v1.1 月度目标持仓；
    2. 如果没有止损，实际持仓 = base_position；
    3. 如果有止损：
       - 买入某 ETF 后记录入场价格；
       - 持仓期间，如果 ETF 当前价格相对入场价格跌幅 <= -stop_loss，
         则当天触发止损，实际持仓转为 cash；
       - 止损后，在 base_position 没有变化之前，不再重新买入；
       - 等到下次月度信号发生变化，再按新信号进入。
    """

    returns = close[ETF_LIST].pct_change().fillna(0.0)

    strategy_returns = pd.Series(index=close.index, data=0.0)
    actual_position = pd.Series(index=close.index, dtype="object")
    stop_loss_flag = pd.Series(index=close.index, data=False)

    prev_actual_position = "cash"
    prev_base_position = "cash"

    entry_price = None
    stopped_out = False
    stop_count = 0

    for date in close.index:
        target_position = base_position.loc[date]

        # 如果月度目标持仓发生变化，解除上一个止损锁定
        if target_position != prev_base_position:
            stopped_out = False
            entry_price = None

        current_position = target_position

        # 如果之前已经止损，且月度目标持仓还没变，则继续空仓
        if stopped_out:
            current_position = "cash"

        # 检查是否新进入某只 ETF
        if prev_actual_position == "cash" and current_position != "cash":
            entry_price = close.loc[date, current_position]

        # 检查止损
        if (
            stop_loss is not None
            and current_position != "cash"
            and entry_price is not None
        ):
            current_price = close.loc[date, current_position]
            holding_return = current_price / entry_price - 1

            if holding_return <= -stop_loss:
                current_position = "cash"
                stopped_out = True
                stop_loss_flag.loc[date] = True
                stop_count += 1

        # 计算当日收益
        if current_position == "cash":
            daily_ret = 0.0
        else:
            daily_ret = returns.loc[date, current_position]

        # 计算交易成本
        cost = 0.0

        if current_position != prev_actual_position:
            if prev_actual_position != "cash":
                cost += SELL_COST

            if current_position != "cash":
                cost += BUY_COST

        strategy_returns.loc[date] = daily_ret - cost
        actual_position.loc[date] = current_position

        # 如果持仓变化为新的 ETF，更新 entry_price
        if current_position != prev_actual_position:
            if current_position != "cash":
                entry_price = close.loc[date, current_position]
            else:
                entry_price = None

        prev_actual_position = current_position
        prev_base_position = target_position

    equity = (1 + strategy_returns).cumprod()

    return strategy_returns, equity, actual_position, stop_loss_flag, stop_count


def calculate_metrics(strategy_returns, equity, actual_position, stop_count):
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

    trade_count = (actual_position != actual_position.shift(1)).sum()
    cash_ratio = (actual_position == "cash").mean()

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "cash_ratio": cash_ratio,
        "stop_count": stop_count,
    }


def analyze_max_drawdown(equity):
    running_max = equity.cummax()
    drawdown = equity / running_max - 1

    trough_date = drawdown.idxmin()
    max_drawdown = drawdown.loc[trough_date]

    equity_before_trough = equity.loc[:trough_date]
    peak_date = equity_before_trough.idxmax()

    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "peak_equity": equity.loc[peak_date],
        "trough_equity": equity.loc[trough_date],
        "max_drawdown": max_drawdown,
    }


def main():
    close = load_data()

    print("数据读取完成")
    print("数据起始日期:", close.index[0].date())
    print("数据结束日期:", close.index[-1].date())
    print("数据行数:", len(close))

    base_position = generate_monthly_signal(close)

    results = []
    dd_results = []

    for stop_loss in STOP_LOSS_LIST:
        strategy_returns, equity, actual_position, stop_loss_flag, stop_count = (
            run_backtest_with_stop_loss(
                close=close,
                base_position=base_position,
                stop_loss=stop_loss,
            )
        )

        metrics = calculate_metrics(
            strategy_returns=strategy_returns,
            equity=equity,
            actual_position=actual_position,
            stop_count=stop_count,
        )

        dd_info = analyze_max_drawdown(equity)

        results.append({
            "stop_loss": stop_loss,
            **metrics,
        })

        dd_results.append({
            "stop_loss": stop_loss,
            **dd_info,
        })

        # 保存每日结果
        daily_df = pd.DataFrame({
            "date": close.index,
            "base_position": base_position.values,
            "actual_position": actual_position.values,
            "strategy_return": strategy_returns.values,
            "equity": equity.values,
            "stop_loss_flag": stop_loss_flag.values,
        })

        label = "none" if stop_loss is None else str(int(stop_loss * 100))
        daily_df.to_csv(
            f"data/v12_stop_loss_{label}_daily.csv",
            index=False,
            encoding="utf-8-sig",
        )

    result_df = pd.DataFrame(results)
    dd_df = pd.DataFrame(dd_results)

    result_df.to_csv(
        "data/v12_stop_loss_result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    dd_df.to_csv(
        "data/v12_stop_loss_drawdown_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_df = result_df.copy()

    percent_cols = [
        "stop_loss",
        "total_return",
        "annual_return",
        "annual_vol",
        "max_drawdown",
        "cash_ratio",
    ]

    for col in percent_cols:
        display_df[col] = display_df[col].apply(
            lambda x: "None" if pd.isna(x) else f"{x:.2%}"
        )

    display_df["final_value"] = display_df["final_value"].apply(lambda x: f"{x:.4f}")
    display_df["sharpe"] = display_df["sharpe"].apply(lambda x: f"{x:.2f}")

    print("\n===== v1.2 单 ETF 止损测试结果 =====\n")
    print(display_df.to_string(index=False))

    display_dd = dd_df.copy()

    display_dd["stop_loss"] = display_dd["stop_loss"].apply(
        lambda x: "None" if pd.isna(x) else f"{x:.2%}"
    )

    display_dd["peak_date"] = display_dd["peak_date"].dt.date
    display_dd["trough_date"] = display_dd["trough_date"].dt.date
    display_dd["peak_equity"] = display_dd["peak_equity"].apply(lambda x: f"{x:.4f}")
    display_dd["trough_equity"] = display_dd["trough_equity"].apply(lambda x: f"{x:.4f}")
    display_dd["max_drawdown"] = display_dd["max_drawdown"].apply(lambda x: f"{x:.2%}")

    print("\n===== v1.2 不同止损参数最大回撤区间 =====\n")
    print(display_dd.to_string(index=False))

    print("\n结果已保存到：")
    print("data/v12_stop_loss_result.csv")
    print("data/v12_stop_loss_drawdown_info.csv")
    print("data/v12_stop_loss_none_daily.csv")
    print("data/v12_stop_loss_5_daily.csv")
    print("data/v12_stop_loss_8_daily.csv")
    print("data/v12_stop_loss_10_daily.csv")
    print("data/v12_stop_loss_12_daily.csv")
    print("data/v12_stop_loss_15_daily.csv")


if __name__ == "__main__":
    main()