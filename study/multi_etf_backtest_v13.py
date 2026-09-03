from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 参数配置
# ============================================================

DATA_FILE = Path("data/etf_close_clean.csv")
OUTPUT_DIR = Path("results/v1_3_multi_etf")

MOMENTUM_DAYS = 10
MA_DAYS = 200
MOMENTUM_THRESHOLD = 0.04

BUY_COST = 0.001
SELL_COST = 0.001

MARKET_FILTER_ETF = "hs300"

TOP_N_LIST = [1, 2, 3, 4]


# ============================================================
# 数据读取
# ============================================================

def load_close_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{file_path}")

    close = pd.read_csv(file_path, index_col=0, parse_dates=True)

    close = close.sort_index()
    close = close.apply(pd.to_numeric, errors="coerce")

    if MARKET_FILTER_ETF not in close.columns:
        raise ValueError(
            f"数据中缺少市场过滤 ETF：{MARKET_FILTER_ETF}"
        )

    if close.index.has_duplicates:
        raise ValueError("日期索引存在重复值，请先检查数据。")

    return close


# ============================================================
# 获取每月最后一个交易日
# ============================================================

def get_month_end_dates(close: pd.DataFrame) -> pd.DatetimeIndex:
    month_period = close.index.to_period("M")

    month_end_dates = (
        close.groupby(month_period)
        .apply(lambda df: df.index[-1])
    )

    return pd.DatetimeIndex(month_end_dates.values)


# ============================================================
# 生成月末目标权重
# ============================================================

def generate_monthly_target_weights(
    close: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """
    在每月最后一个交易日收盘后生成目标权重。

    规则：
    1. hs300 必须高于 MA200；
    2. ETF 10日动量必须大于 4%；
    3. 选择动量排名前 top_n 的 ETF；
    4. 每个持仓槽位固定权重为 1 / top_n；
    5. 候选不足时，剩余仓位保持现金。
    """

    momentum = close.pct_change(MOMENTUM_DAYS, fill_method=None)
    hs300_ma = close[MARKET_FILTER_ETF].rolling(MA_DAYS).mean()

    month_end_dates = get_month_end_dates(close)

    target_weights = pd.DataFrame(
        0.0,
        index=month_end_dates,
        columns=close.columns,
    )


    for signal_date in month_end_dates:
        hs300_price = close.at[signal_date, MARKET_FILTER_ETF]
        hs300_ma_value = hs300_ma.at[signal_date]

        # MA200尚未形成，或者市场处于MA200下方
        if (
            pd.isna(hs300_ma_value)
            or pd.isna(hs300_price)
            or hs300_price <= hs300_ma_value
        ):
            continue

        current_momentum = momentum.loc[signal_date].dropna()

        # 只保留动量严格大于4%的ETF
        qualified = current_momentum[
            current_momentum > MOMENTUM_THRESHOLD
        ]

        if qualified.empty:
            continue

        selected = (
            qualified
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        selected_count = len(selected)

        if selected_count > 0:
            equal_weight = 1.0 / selected_count
            target_weights.loc[signal_date, selected] = equal_weight

    return target_weights


# ============================================================
# 将月末信号转换为每日实际持仓
# ============================================================

def build_daily_weights(
    close: pd.DataFrame,
    monthly_target_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    月末信号从下一个交易日开始生效。

    例如：
    1月31日收盘后生成信号；
    2月第一个交易日使用新权重。
    """

    daily_weights = pd.DataFrame(
        np.nan,
        index=close.index,
        columns=close.columns,
    )

    for signal_date, target_weight in monthly_target_weights.iterrows():
        signal_position = close.index.get_indexer([signal_date])[0]

        if signal_position == -1:
            continue

        next_position = signal_position + 1

        if next_position >= len(close.index):
            continue

        effective_date = close.index[next_position]
        daily_weights.loc[effective_date] = target_weight.values

    # 信号生效后，一直持有到下一次信号生效
    daily_weights = daily_weights.ffill().fillna(0.0)

    return daily_weights


# ============================================================
# 计算交易成本
# ============================================================

def calculate_daily_costs(
    daily_weights: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    根据每日目标权重变化计算买卖成本。

    权重增加视为买入；
    权重减少视为卖出。
    """

    previous_weights = daily_weights.shift(1).fillna(0.0)
    weight_change = daily_weights - previous_weights

    buy_turnover = weight_change.clip(lower=0).sum(axis=1)
    sell_turnover = (-weight_change.clip(upper=0)).sum(axis=1)

    daily_cost = (
        buy_turnover * BUY_COST
        + sell_turnover * SELL_COST
    )

    return daily_cost, buy_turnover, sell_turnover


# ============================================================
# 回测
# ============================================================

def run_backtest(
    close: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_weights = generate_monthly_target_weights(
        close=close,
        top_n=top_n,
    )

    daily_weights = build_daily_weights(
        close=close,
        monthly_target_weights=monthly_weights,
    )

    asset_returns = close.pct_change(fill_method=None).fillna(0.0)

    # 当日收益由当日持仓承担
    gross_return = (
        daily_weights * asset_returns
    ).sum(axis=1)

    daily_cost, buy_turnover, sell_turnover = calculate_daily_costs(
        daily_weights
    )

    net_return = gross_return - daily_cost
    net_value = (1.0 + net_return).cumprod()

    invested_weight = daily_weights.sum(axis=1)
    cash_weight = 1.0 - invested_weight

    result = pd.DataFrame({
        "gross_return": gross_return,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "cost": daily_cost,
        "net_return": net_return,
        "net_value": net_value,
        "invested_weight": invested_weight,
        "cash_weight": cash_weight,
    })

    return result, daily_weights


# ============================================================
# 绩效统计
# ============================================================

def calculate_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
    top_n: int,
) -> dict:
    net_return = result["net_return"]
    net_value = result["net_value"]

    trading_days = len(result)
    years = trading_days / 252

    final_value = net_value.iloc[-1]
    cumulative_return = final_value - 1.0

    if years > 0 and final_value > 0:
        annual_return = final_value ** (1 / years) - 1
    else:
        annual_return = np.nan

    annual_volatility = net_return.std(ddof=1) * np.sqrt(252)

    if annual_volatility > 0:
        sharpe = annual_return / annual_volatility
    else:
        sharpe = np.nan

    rolling_max = net_value.cummax()
    drawdown = net_value / rolling_max - 1.0
    max_drawdown = drawdown.min()

    cash_ratio = (
        result["cash_weight"]
        .clip(lower=0, upper=1)
        .mean()
    )

    average_invested_weight = (
        result["invested_weight"]
        .clip(lower=0, upper=1)
        .mean()
    )

    # 任意权重变化都视为一次调仓日
    weight_change = daily_weights.diff().abs().sum(axis=1)
    rebalance_days = int((weight_change > 1e-12).sum())

    total_buy_turnover = result["buy_turnover"].sum()
    total_sell_turnover = result["sell_turnover"].sum()
    total_cost = result["cost"].sum()

    # 平均实际持有ETF数量
    holding_count = (daily_weights > 1e-12).sum(axis=1)
    average_holding_count = holding_count.mean()
    maximum_holding_count = holding_count.max()

    return {
        "top_n": top_n,
        "final_value": final_value,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "cash_ratio": cash_ratio,
        "average_invested_weight": average_invested_weight,
        "average_holding_count": average_holding_count,
        "maximum_holding_count": maximum_holding_count,
        "rebalance_days": rebalance_days,
        "total_buy_turnover": total_buy_turnover,
        "total_sell_turnover": total_sell_turnover,
        "total_cost": total_cost,
    }


# ============================================================
# 打印结果
# ============================================================

def print_metrics(metrics: dict) -> None:
    print("=" * 60)
    print(f"Top {metrics['top_n']} 多ETF持仓")
    print("=" * 60)

    print(f"最终净值：       {metrics['final_value']:.4f}")
    print(f"累计收益：       {metrics['cumulative_return']:.2%}")
    print(f"年化收益：       {metrics['annual_return']:.2%}")
    print(f"年化波动：       {metrics['annual_volatility']:.2%}")
    print(f"夏普比率：       {metrics['sharpe']:.2f}")
    print(f"最大回撤：       {metrics['max_drawdown']:.2%}")
    print(f"平均现金比例：   {metrics['cash_ratio']:.2%}")
    print(
        f"平均投入仓位：   "
        f"{metrics['average_invested_weight']:.2%}"
    )
    print(
        f"平均持仓数量：   "
        f"{metrics['average_holding_count']:.2f}"
    )
    print(
        f"最大持仓数量：   "
        f"{metrics['maximum_holding_count']}"
    )
    print(f"调仓交易日数：   {metrics['rebalance_days']}")
    print(
        f"累计买入换手：   "
        f"{metrics['total_buy_turnover']:.2f}"
    )
    print(
        f"累计卖出换手：   "
        f"{metrics['total_sell_turnover']:.2f}"
    )
    print(f"累计成本率：     {metrics['total_cost']:.2%}")


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    close = load_close_data(DATA_FILE)

    print(f"数据开始日期：{close.index.min().date()}")
    print(f"数据结束日期：{close.index.max().date()}")
    print(f"交易日数量：  {len(close)}")
    print(f"ETF数量：     {len(close.columns)}")
    print()

    all_metrics = []

    for top_n in TOP_N_LIST:
        result, daily_weights = run_backtest(
            close=close,
            top_n=top_n,
        )

        metrics = calculate_metrics(
            result=result,
            daily_weights=daily_weights,
            top_n=top_n,
        )

        all_metrics.append(metrics)
        print_metrics(metrics)

        result.to_csv(
            OUTPUT_DIR / f"backtest_top{top_n}.csv",
            encoding="utf-8-sig",
        )

        daily_weights.to_csv(
            OUTPUT_DIR / f"weights_top{top_n}.csv",
            encoding="utf-8-sig",
        )

    comparison = pd.DataFrame(all_metrics)

    comparison.to_csv(
        OUTPUT_DIR / "multi_etf_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    display_columns = [
        "top_n",
        "final_value",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "cash_ratio",
        "average_holding_count",
        "rebalance_days",
        "total_cost",
    ]

    print()
    print("=" * 100)
    print("v1.3 多ETF持仓对比")
    print("=" * 100)

    print(
        comparison[display_columns].to_string(
            index=False,
            formatters={
                "final_value": "{:.4f}".format,
                "annual_return": "{:.2%}".format,
                "annual_volatility": "{:.2%}".format,
                "sharpe": "{:.2f}".format,
                "max_drawdown": "{:.2%}".format,
                "cash_ratio": "{:.2%}".format,
                "average_holding_count": "{:.2f}".format,
                "total_cost": "{:.2%}".format,
            },
        )
    )

    print()
    print(f"结果已经保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()