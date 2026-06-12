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

    for d in actual_month_end_dates:
        mom_row = momentum.loc[d, ETF_LIST].dropna()

        if len(mom_row) == 0:
            signals.loc[d] = "cash"
            continue

        best_etf = mom_row.idxmax()
        best_mom = mom_row.max()

        market_ma_value = market_ma.loc[d]

        if pd.isna(market_ma_value):
            market_ok = False
        else:
            market_ok = close.loc[d, MARKET_FILTER_ETF] > market_ma_value

        if best_mom <= MOMENTUM_THRESHOLD:
            signals.loc[d] = "cash"
        elif not market_ok:
            signals.loc[d] = "cash"
        else:
            signals.loc[d] = best_etf

    base_position = signals.shift(1).ffill().fillna("cash")

    return base_position


def find_planned_exit_date(base_position, entry_date, holding):
    """
    找到如果不止损，原计划持有到哪一天。

    逻辑：
    从 entry_date 往后看，直到 base_position 不再等于 holding。
    退出日取变化前的最后一个交易日。
    """

    dates = list(base_position.index)
    start_idx = dates.index(entry_date)

    planned_exit_date = dates[-1]

    for i in range(start_idx + 1, len(dates)):
        d = dates[i]

        if base_position.loc[d] != holding:
            planned_exit_date = dates[i - 1]
            break

    return planned_exit_date


def run_backtest_and_collect_stop_events(close, base_position, stop_loss):
    """
    回测并收集止损事件。

    返回：
    stop_events_df
    """

    returns = close[ETF_LIST].pct_change().fillna(0.0)

    prev_actual_position = "cash"
    prev_base_position = "cash"

    entry_date = None
    entry_price = None

    stopped_out = False

    current_equity = 1.0

    stop_events = []

    for date in close.index:
        target_position = base_position.loc[date]

        # 月度目标持仓变化时，解除止损锁定
        if target_position != prev_base_position:
            stopped_out = False
            entry_date = None
            entry_price = None

        current_position = target_position

        if stopped_out:
            current_position = "cash"

        # 新入场
        if prev_actual_position == "cash" and current_position != "cash":
            entry_date = date
            entry_price = close.loc[date, current_position]

        # ETF 切换，视为新入场
        if (
            prev_actual_position != "cash"
            and current_position != "cash"
            and current_position != prev_actual_position
        ):
            entry_date = date
            entry_price = close.loc[date, current_position]

        triggered = False

        # 止损判断
        if (
            current_position != "cash"
            and entry_price is not None
            and stop_loss is not None
        ):
            current_price = close.loc[date, current_position]
            holding_return = current_price / entry_price - 1

            if holding_return <= -stop_loss:
                planned_exit_date = find_planned_exit_date(
                    base_position=base_position,
                    entry_date=entry_date,
                    holding=current_position,
                )

                planned_exit_price = close.loc[planned_exit_date, current_position]
                holding_return_to_planned_exit = planned_exit_price / entry_price - 1

                saved_or_missed_return = (
                    holding_return_to_planned_exit - holding_return
                )

                if holding_return_to_planned_exit < holding_return:
                    stop_effect = "effective_saved_loss"
                elif holding_return_to_planned_exit > 0:
                    stop_effect = "missed_profit"
                elif holding_return_to_planned_exit > holding_return:
                    stop_effect = "reduced_loss_but_missed_rebound"
                else:
                    stop_effect = "neutral"

                stop_events.append({
                    "stop_loss": stop_loss,
                    "holding": current_position,
                    "entry_date": entry_date,
                    "stop_date": date,
                    "planned_exit_date": planned_exit_date,
                    "entry_price": entry_price,
                    "stop_price": current_price,
                    "planned_exit_price": planned_exit_price,
                    "holding_return_at_stop": holding_return,
                    "holding_return_to_planned_exit": holding_return_to_planned_exit,
                    "saved_or_missed_return": saved_or_missed_return,
                    "stop_effect": stop_effect,
                    "equity_before_stop": current_equity,
                })

                current_position = "cash"
                stopped_out = True
                triggered = True

        # 计算日收益和成本，只为保持 equity 近似一致
        if current_position == "cash":
            daily_ret = 0.0
        else:
            daily_ret = returns.loc[date, current_position]

        cost = 0.0

        if current_position != prev_actual_position:
            if prev_actual_position != "cash":
                cost += SELL_COST

            if current_position != "cash":
                cost += BUY_COST

        current_equity = current_equity * (1 + daily_ret - cost)

        if triggered:
            entry_date = None
            entry_price = None

        if current_position != prev_actual_position:
            if current_position != "cash":
                entry_date = date
                entry_price = close.loc[date, current_position]
            else:
                entry_date = None
                entry_price = None

        prev_actual_position = current_position
        prev_base_position = target_position

    return pd.DataFrame(stop_events)


def summarize_by_stop_loss(events_df):
    rows = []

    for stop_loss, group in events_df.groupby("stop_loss"):
        event_count = len(group)

        effective_count = (group["stop_effect"] == "effective_saved_loss").sum()
        missed_profit_count = (group["stop_effect"] == "missed_profit").sum()
        reduced_loss_count = (
            group["stop_effect"] == "reduced_loss_but_missed_rebound"
        ).sum()

        avg_return_at_stop = group["holding_return_at_stop"].mean()
        avg_return_to_exit = group["holding_return_to_planned_exit"].mean()
        avg_saved_or_missed = group["saved_or_missed_return"].mean()

        total_saved_or_missed = group["saved_or_missed_return"].sum()

        rows.append({
            "stop_loss": stop_loss,
            "event_count": event_count,
            "effective_count": effective_count,
            "missed_profit_count": missed_profit_count,
            "reduced_loss_count": reduced_loss_count,
            "avg_return_at_stop": avg_return_at_stop,
            "avg_return_to_planned_exit": avg_return_to_exit,
            "avg_saved_or_missed": avg_saved_or_missed,
            "total_saved_or_missed": total_saved_or_missed,
        })

    return pd.DataFrame(rows)


def main():
    close = load_data()

    print("数据读取完成")
    print("数据起始日期:", close.index[0].date())
    print("数据结束日期:", close.index[-1].date())
    print("数据行数:", len(close))

    base_position = generate_monthly_signal(close)

    all_events = []

    for stop_loss in STOP_LOSS_LIST:
        events_df = run_backtest_and_collect_stop_events(
            close=close,
            base_position=base_position,
            stop_loss=stop_loss,
        )

        if len(events_df) > 0:
            all_events.append(events_df)

    if len(all_events) > 0:
        all_events_df = pd.concat(all_events, ignore_index=True)
    else:
        all_events_df = pd.DataFrame()

    if len(all_events_df) == 0:
        print("没有止损事件。")
        return

    all_events_df["entry_month"] = all_events_df["entry_date"].dt.to_period("M").astype(str)
    all_events_df["stop_month"] = all_events_df["stop_date"].dt.to_period("M").astype(str)
    all_events_df["planned_exit_month"] = all_events_df["planned_exit_date"].dt.to_period("M").astype(str)

    summary_df = summarize_by_stop_loss(all_events_df)

    all_events_df.to_csv(
        "data/v12_stop_loss_opportunity_events.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        "data/v12_stop_loss_opportunity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== v1.2 止损机会损失事件明细 =====\n")

    display_events = all_events_df.copy()

    percent_cols_event = [
        "stop_loss",
        "holding_return_at_stop",
        "holding_return_to_planned_exit",
        "saved_or_missed_return",
    ]

    for col in percent_cols_event:
        display_events[col] = display_events[col].apply(lambda x: f"{x:.2%}")

    display_events["entry_date"] = display_events["entry_date"].dt.date
    display_events["stop_date"] = display_events["stop_date"].dt.date
    display_events["planned_exit_date"] = display_events["planned_exit_date"].dt.date

    for col in ["entry_price", "stop_price", "planned_exit_price", "equity_before_stop"]:
        display_events[col] = display_events[col].apply(lambda x: f"{x:.4f}")

    cols = [
        "stop_loss",
        "holding",
        "entry_date",
        "stop_date",
        "planned_exit_date",
        "holding_return_at_stop",
        "holding_return_to_planned_exit",
        "saved_or_missed_return",
        "stop_effect",
    ]

    print(display_events[cols].to_string(index=False))

    print("\n===== v1.2 止损机会损失汇总 =====\n")

    display_summary = summary_df.copy()

    percent_cols_summary = [
        "stop_loss",
        "avg_return_at_stop",
        "avg_return_to_planned_exit",
        "avg_saved_or_missed",
        "total_saved_or_missed",
    ]

    for col in percent_cols_summary:
        display_summary[col] = display_summary[col].apply(lambda x: f"{x:.2%}")

    print(display_summary.to_string(index=False))

    print("\n结果已保存到：")
    print("data/v12_stop_loss_opportunity_events.csv")
    print("data/v12_stop_loss_opportunity_summary.csv")


if __name__ == "__main__":
    main()