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
    生成 v1.1 月度目标持仓。
    """

    momentum = close[ETF_LIST].pct_change(MOMENTUM_DAYS)
    market_ma = close[MARKET_FILTER_ETF].rolling(MARKET_MA_DAYS).mean()

    actual_month_end_dates = get_actual_month_end_dates(close)

    signals = pd.Series(index=close.index, dtype="object")

    signal_info_rows = []

    for d in actual_month_end_dates:
        mom_row = momentum.loc[d, ETF_LIST].dropna()

        if len(mom_row) == 0:
            signals.loc[d] = "cash"

            signal_info_rows.append({
                "signal_date": d,
                "best_etf": "cash",
                "best_mom": np.nan,
                "market_ok": False,
                "signal": "cash",
            })
            continue

        best_etf = mom_row.idxmax()
        best_mom = mom_row.max()

        market_ma_value = market_ma.loc[d]

        if pd.isna(market_ma_value):
            market_ok = False
        else:
            market_ok = close.loc[d, MARKET_FILTER_ETF] > market_ma_value

        if best_mom <= MOMENTUM_THRESHOLD:
            signal = "cash"
        elif not market_ok:
            signal = "cash"
        else:
            signal = best_etf

        signals.loc[d] = signal

        signal_info_rows.append({
            "signal_date": d,
            "best_etf": best_etf,
            "best_mom": best_mom,
            "market_ok": market_ok,
            "signal": signal,
        })

    base_position = signals.shift(1).ffill().fillna("cash")
    signal_info_df = pd.DataFrame(signal_info_rows)

    return base_position, signal_info_df


def run_backtest_with_stop_loss(close, base_position, stop_loss):
    """
    回测并记录止损事件。

    止损逻辑：
    - 月度目标仓位来自 base_position；
    - 入场后记录 entry_date 和 entry_price；
    - 如果持仓 ETF 相对 entry_price 跌幅达到 stop_loss，则当天转 cash；
    - 本轮月度目标持仓不变前，不重新买入；
    - 等下次月度目标持仓变化时，解除止损锁定。
    """

    returns = close[ETF_LIST].pct_change().fillna(0.0)

    strategy_returns = pd.Series(index=close.index, data=0.0)
    actual_position = pd.Series(index=close.index, dtype="object")
    equity = pd.Series(index=close.index, data=np.nan)

    stop_events = []

    prev_actual_position = "cash"
    prev_base_position = "cash"

    entry_date = None
    entry_price = None
    entry_etf = None

    stopped_out = False

    current_equity = 1.0

    for date in close.index:
        target_position = base_position.loc[date]

        # 月度目标持仓变化时，解除止损锁定
        if target_position != prev_base_position:
            stopped_out = False
            entry_date = None
            entry_price = None
            entry_etf = None

        current_position = target_position

        # 如果已经止损，而月度目标持仓未变化，则继续 cash
        if stopped_out:
            current_position = "cash"

        # 新入场：从 cash 进入 ETF
        if prev_actual_position == "cash" and current_position != "cash":
            entry_date = date
            entry_price = close.loc[date, current_position]
            entry_etf = current_position

        # 如果 ETF 切换，也视为新入场
        if (
            prev_actual_position != "cash"
            and current_position != "cash"
            and current_position != prev_actual_position
        ):
            entry_date = date
            entry_price = close.loc[date, current_position]
            entry_etf = current_position

        triggered = False
        holding_return_at_stop = np.nan
        stop_price = np.nan

        # 止损判断
        if (
            current_position != "cash"
            and entry_price is not None
            and stop_loss is not None
        ):
            current_price = close.loc[date, current_position]
            holding_return = current_price / entry_price - 1

            if holding_return <= -stop_loss:
                triggered = True
                holding_return_at_stop = holding_return
                stop_price = current_price

                stop_events.append({
                    "stop_loss": stop_loss,
                    "stop_date": date,
                    "holding": current_position,
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "holding_return_at_stop": holding_return_at_stop,
                    "equity_before_stop": current_equity,
                })

                current_position = "cash"
                stopped_out = True

        # 计算当日收益
        if current_position == "cash":
            daily_ret = 0.0
        else:
            daily_ret = returns.loc[date, current_position]

        # 成本
        cost = 0.0

        if current_position != prev_actual_position:
            if prev_actual_position != "cash":
                cost += SELL_COST

            if current_position != "cash":
                cost += BUY_COST

        daily_strategy_ret = daily_ret - cost
        current_equity = current_equity * (1 + daily_strategy_ret)

        strategy_returns.loc[date] = daily_strategy_ret
        actual_position.loc[date] = current_position
        equity.loc[date] = current_equity

        # 如果发生了止损，清空入场信息
        if triggered:
            entry_date = None
            entry_price = None
            entry_etf = None

        # 如果持仓发生变化且新持仓为 ETF，更新入场信息
        if current_position != prev_actual_position:
            if current_position != "cash":
                entry_date = date
                entry_price = close.loc[date, current_position]
                entry_etf = current_position
            else:
                entry_date = None
                entry_price = None
                entry_etf = None

        prev_actual_position = current_position
        prev_base_position = target_position

    stop_events_df = pd.DataFrame(stop_events)

    return strategy_returns, equity, actual_position, stop_events_df


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


def main():
    close = load_data()

    print("数据读取完成")
    print("数据起始日期:", close.index[0].date())
    print("数据结束日期:", close.index[-1].date())
    print("数据行数:", len(close))

    base_position, signal_info_df = generate_monthly_signal(close)

    all_events = []
    all_metrics = []

    for stop_loss in STOP_LOSS_LIST:
        strategy_returns, equity, actual_position, stop_events_df = (
            run_backtest_with_stop_loss(
                close=close,
                base_position=base_position,
                stop_loss=stop_loss,
            )
        )

        stop_count = len(stop_events_df)

        metrics = calculate_metrics(
            strategy_returns=strategy_returns,
            equity=equity,
            actual_position=actual_position,
            stop_count=stop_count,
        )

        all_metrics.append({
            "stop_loss": stop_loss,
            **metrics,
        })

        if stop_count > 0:
            stop_events_df["stop_loss_label"] = f"{stop_loss:.0%}"
            all_events.append(stop_events_df)

    metrics_df = pd.DataFrame(all_metrics)

    if len(all_events) > 0:
        events_df = pd.concat(all_events, ignore_index=True)
    else:
        events_df = pd.DataFrame()

    # 增加止损月份字段，方便看是否集中在某些月份
    if len(events_df) > 0:
        events_df["stop_year"] = events_df["stop_date"].dt.year
        events_df["stop_month"] = events_df["stop_date"].dt.to_period("M").astype(str)
        events_df["entry_month"] = events_df["entry_date"].dt.to_period("M").astype(str)

    # 保存
    metrics_df.to_csv(
        "data/v12_stop_loss_event_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    events_df.to_csv(
        "data/v12_stop_loss_events.csv",
        index=False,
        encoding="utf-8-sig",
    )

    signal_info_df.to_csv(
        "data/v12_stop_loss_signal_info.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 打印指标
    print("\n===== v1.2 止损参数复算结果 =====\n")

    display_metrics = metrics_df.copy()

    percent_cols = [
        "stop_loss",
        "total_return",
        "annual_return",
        "annual_vol",
        "max_drawdown",
        "cash_ratio",
    ]

    for col in percent_cols:
        display_metrics[col] = display_metrics[col].apply(lambda x: f"{x:.2%}")

    display_metrics["final_value"] = display_metrics["final_value"].apply(lambda x: f"{x:.4f}")
    display_metrics["sharpe"] = display_metrics["sharpe"].apply(lambda x: f"{x:.2f}")

    print(display_metrics.to_string(index=False))

    # 打印止损事件
    print("\n===== v1.2 止损触发事件明细 =====\n")

    if len(events_df) == 0:
        print("没有止损事件。")
    else:
        display_events = events_df.copy()

        display_events["stop_loss"] = display_events["stop_loss"].apply(lambda x: f"{x:.2%}")
        display_events["stop_date"] = display_events["stop_date"].dt.date
        display_events["entry_date"] = display_events["entry_date"].dt.date
        display_events["entry_price"] = display_events["entry_price"].apply(lambda x: f"{x:.4f}")
        display_events["stop_price"] = display_events["stop_price"].apply(lambda x: f"{x:.4f}")
        display_events["holding_return_at_stop"] = display_events["holding_return_at_stop"].apply(lambda x: f"{x:.2%}")
        display_events["equity_before_stop"] = display_events["equity_before_stop"].apply(lambda x: f"{x:.4f}")

        cols = [
            "stop_loss",
            "stop_date",
            "holding",
            "entry_date",
            "entry_price",
            "stop_price",
            "holding_return_at_stop",
            "equity_before_stop",
            "stop_month",
        ]

        print(display_events[cols].to_string(index=False))

    # 按止损参数统计事件
    print("\n===== 各止损参数触发次数统计 =====\n")

    if len(events_df) == 0:
        print("没有止损事件。")
    else:
        event_count = (
            events_df
            .groupby(["stop_loss", "holding"])
            .size()
            .reset_index(name="count")
            .sort_values(["stop_loss", "count"], ascending=[True, False])
        )

        display_count = event_count.copy()
        display_count["stop_loss"] = display_count["stop_loss"].apply(lambda x: f"{x:.2%}")

        print(display_count.to_string(index=False))

    print("\n结果已保存到：")
    print("data/v12_stop_loss_event_metrics.csv")
    print("data/v12_stop_loss_events.csv")
    print("data/v12_stop_loss_signal_info.csv")


if __name__ == "__main__":
    main()