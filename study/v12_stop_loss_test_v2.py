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
    df = df.sort_values("date").set_index("date")

    missing_cols = [col for col in ETF_LIST if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少 ETF 列：{missing_cols}")

    return df


def get_actual_month_end_dates(close):
    """
    获取每个月实际存在的最后一个交易日。
    """
    return close.groupby(close.index.to_period("M")).apply(
        lambda x: x.index[-1]
    ).tolist()


def generate_base_position(close):
    """
    生成 v1.1 的基础月度持仓。

    规则：
    - 10 日动量；
    - 最强动量必须 > 4%；
    - hs300 必须位于 MA200 上方；
    - 月末生成信号；
    - 下一个交易日生效。
    """
    momentum = close[ETF_LIST].pct_change(MOMENTUM_DAYS)
    market_ma = close[MARKET_FILTER_ETF].rolling(MARKET_MA_DAYS).mean()

    month_end_dates = get_actual_month_end_dates(close)

    signals = pd.Series(index=close.index, dtype="object")

    for date in month_end_dates:
        mom_row = momentum.loc[date, ETF_LIST].dropna()

        if mom_row.empty:
            signals.loc[date] = "cash"
            continue

        best_etf = mom_row.idxmax()
        best_momentum = mom_row.max()

        ma_value = market_ma.loc[date]

        if pd.isna(ma_value):
            market_ok = False
        else:
            market_ok = (
                close.loc[date, MARKET_FILTER_ETF] > ma_value
            )

        if best_momentum <= MOMENTUM_THRESHOLD:
            signals.loc[date] = "cash"
        elif not market_ok:
            signals.loc[date] = "cash"
        else:
            signals.loc[date] = best_etf

    base_position = signals.shift(1).ffill().fillna("cash")

    return base_position


def run_backtest_with_stop_loss(close, base_position, stop_loss):
    """
    使用日线收盘价进行保守的止损回测。

    时点规则：
    1. 当天先按持仓计算完整日收益；
    2. 收盘后检查是否达到止损线；
    3. 若触发，则在当天收盘卖出并扣卖出成本；
    4. 下一个交易日起为空仓；
    5. 月度目标持仓改变后，解除止损锁定。

    注意：
    只有日线收盘数据，因此不假设能精确按止损价格成交。
    """
    etf_returns = close[ETF_LIST].pct_change().fillna(0.0)

    strategy_returns = pd.Series(
        0.0, index=close.index, dtype=float
    )

    # 记录每日收盘后的实际持仓
    end_position = pd.Series(
        "cash", index=close.index, dtype="object"
    )

    stop_flag = pd.Series(
        False, index=close.index, dtype=bool
    )

    stop_events = []

    previous_end_position = "cash"
    previous_target = "cash"

    entry_date = None
    entry_price = None
    stopped_out = False

    for date in close.index:
        target = base_position.loc[date]

        # 月度目标发生变化，允许重新入场
        if target != previous_target:
            stopped_out = False

        # 决定当天开始时的持仓
        if stopped_out:
            start_position = "cash"
        else:
            start_position = target

        transaction_cost = 0.0

        # 当天开始时根据目标调整持仓
        if start_position != previous_end_position:
            if previous_end_position != "cash":
                transaction_cost += SELL_COST

            if start_position != "cash":
                transaction_cost += BUY_COST
                entry_date = date
                entry_price = close.loc[date, start_position]
            else:
                entry_date = None
                entry_price = None

        # 当日完整收益：止损判断前先承担当日涨跌
        if start_position == "cash":
            daily_gross_return = 0.0
        else:
            daily_gross_return = etf_returns.loc[
                date, start_position
            ]

        closing_position = start_position

        # 收盘后检查止损
        if (
            stop_loss is not None
            and start_position != "cash"
            and entry_price is not None
        ):
            current_price = close.loc[date, start_position]
            holding_return = current_price / entry_price - 1

            if holding_return <= -stop_loss:
                # 当天完整收益已经计入；
                # 收盘卖出，再扣一次卖出成本
                transaction_cost += SELL_COST
                closing_position = "cash"
                stopped_out = True
                stop_flag.loc[date] = True

                stop_events.append({
                    "stop_loss": stop_loss,
                    "entry_date": entry_date,
                    "stop_date": date,
                    "holding": start_position,
                    "entry_price": entry_price,
                    "stop_close_price": current_price,
                    "holding_return_at_stop": holding_return,
                })

                entry_date = None
                entry_price = None

        strategy_returns.loc[date] = (
            daily_gross_return - transaction_cost
        )

        end_position.loc[date] = closing_position

        previous_end_position = closing_position
        previous_target = target

    equity = (1 + strategy_returns).cumprod()
    stop_events_df = pd.DataFrame(stop_events)

    return (
        strategy_returns,
        equity,
        end_position,
        stop_flag,
        stop_events_df,
    )


def calculate_metrics(
    strategy_returns,
    equity,
    end_position,
    stop_count,
):
    final_value = equity.iloc[-1]
    total_return = final_value - 1

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25

    annual_return = (
        final_value ** (1 / years) - 1
        if years > 0
        else np.nan
    )

    annual_vol = strategy_returns.std() * np.sqrt(252)

    sharpe = (
        annual_return / annual_vol
        if annual_vol > 0 and not np.isnan(annual_vol)
        else np.nan
    )

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min()

    # 收盘后持仓发生变化的次数
    trade_count = (
        end_position != end_position.shift(1).fillna("cash")
    ).sum()

    cash_ratio = (end_position == "cash").mean()

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
    peak_date = equity.loc[:trough_date].idxmax()

    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "peak_equity": equity.loc[peak_date],
        "trough_equity": equity.loc[trough_date],
        "max_drawdown": drawdown.loc[trough_date],
    }


def main():
    close = load_data()
    base_position = generate_base_position(close)

    print("数据读取完成")
    print("数据起始日期:", close.index[0].date())
    print("数据结束日期:", close.index[-1].date())
    print("数据行数:", len(close))

    metrics_rows = []
    drawdown_rows = []
    all_stop_events = []

    for stop_loss in STOP_LOSS_LIST:
        (
            strategy_returns,
            equity,
            end_position,
            stop_flag,
            stop_events_df,
        ) = run_backtest_with_stop_loss(
            close=close,
            base_position=base_position,
            stop_loss=stop_loss,
        )

        metrics = calculate_metrics(
            strategy_returns=strategy_returns,
            equity=equity,
            end_position=end_position,
            stop_count=len(stop_events_df),
        )

        drawdown_info = analyze_max_drawdown(equity)

        metrics_rows.append({
            "stop_loss": stop_loss,
            **metrics,
        })

        drawdown_rows.append({
            "stop_loss": stop_loss,
            **drawdown_info,
        })

        if not stop_events_df.empty:
            all_stop_events.append(stop_events_df)

        label = (
            "none"
            if stop_loss is None
            else str(int(stop_loss * 100))
        )

        daily_df = pd.DataFrame({
            "date": close.index,
            "base_position": base_position.values,
            "end_position": end_position.values,
            "strategy_return": strategy_returns.values,
            "equity": equity.values,
            "stop_flag": stop_flag.values,
        })

        daily_df.to_csv(
            f"data/v12_stop_loss_v2_{label}_daily.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics_df = pd.DataFrame(metrics_rows)
    drawdown_df = pd.DataFrame(drawdown_rows)

    if all_stop_events:
        events_df = pd.concat(
            all_stop_events,
            ignore_index=True,
        )
    else:
        events_df = pd.DataFrame()

    metrics_df.to_csv(
        "data/v12_stop_loss_v2_result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    drawdown_df.to_csv(
        "data/v12_stop_loss_v2_drawdown.csv",
        index=False,
        encoding="utf-8-sig",
    )

    events_df.to_csv(
        "data/v12_stop_loss_v2_events.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_metrics = metrics_df.copy()

    for col in [
        "stop_loss",
        "total_return",
        "annual_return",
        "annual_vol",
        "max_drawdown",
        "cash_ratio",
    ]:
        display_metrics[col] = display_metrics[col].apply(
            lambda x: "None" if pd.isna(x) else f"{x:.2%}"
        )

    display_metrics["final_value"] = (
        display_metrics["final_value"]
        .map(lambda x: f"{x:.4f}")
    )

    display_metrics["sharpe"] = (
        display_metrics["sharpe"]
        .map(lambda x: f"{x:.2f}")
    )

    print("\n===== 修正版 v1.2 单 ETF 止损测试 =====\n")
    print(display_metrics.to_string(index=False))

    display_dd = drawdown_df.copy()

    display_dd["stop_loss"] = display_dd["stop_loss"].apply(
        lambda x: "None" if pd.isna(x) else f"{x:.2%}"
    )
    display_dd["peak_date"] = display_dd["peak_date"].dt.date
    display_dd["trough_date"] = display_dd["trough_date"].dt.date
    display_dd["peak_equity"] = display_dd["peak_equity"].map(
        lambda x: f"{x:.4f}"
    )
    display_dd["trough_equity"] = display_dd[
        "trough_equity"
    ].map(lambda x: f"{x:.4f}")
    display_dd["max_drawdown"] = display_dd[
        "max_drawdown"
    ].map(lambda x: f"{x:.2%}")

    print("\n===== 修正版最大回撤区间 =====\n")
    print(display_dd.to_string(index=False))

    if not events_df.empty:
        display_events = events_df.copy()

        display_events["stop_loss"] = display_events[
            "stop_loss"
        ].map(lambda x: f"{x:.2%}")
        display_events["entry_date"] = display_events[
            "entry_date"
        ].dt.date
        display_events["stop_date"] = display_events[
            "stop_date"
        ].dt.date
        display_events["entry_price"] = display_events[
            "entry_price"
        ].map(lambda x: f"{x:.4f}")
        display_events["stop_close_price"] = display_events[
            "stop_close_price"
        ].map(lambda x: f"{x:.4f}")
        display_events["holding_return_at_stop"] = (
            display_events["holding_return_at_stop"]
            .map(lambda x: f"{x:.2%}")
        )

        print("\n===== 修正版止损触发事件 =====\n")
        print(display_events.to_string(index=False))

    print("\n结果已保存到：")
    print("data/v12_stop_loss_v2_result.csv")
    print("data/v12_stop_loss_v2_drawdown.csv")
    print("data/v12_stop_loss_v2_events.csv")


if __name__ == "__main__":
    main()