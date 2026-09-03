from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 参数配置
# ============================================================

DATA_FILE = Path("data/etf_close_clean.csv")
OUTPUT_DIR = Path("results/v1_3_multi_etf_dynamic")

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
    """
    读取ETF收盘价宽表。

    要求：
    - 第一列为日期；
    - 其余列为ETF收盘价；
    - 日期升序排列；
    - 不允许日期重复。
    """

    if not file_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{file_path}")

    close = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True,
    )

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    close = close.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if MARKET_FILTER_ETF not in close.columns:
        raise ValueError(
            f"数据中缺少市场过滤ETF：{MARKET_FILTER_ETF}"
        )

    if close.index.has_duplicates:
        duplicate_dates = close.index[
            close.index.duplicated()
        ].unique()

        raise ValueError(
            f"日期索引存在重复值：{duplicate_dates.tolist()}"
        )

    if close.empty:
        raise ValueError("收盘价数据为空。")

    return close


# ============================================================
# 获取每月最后一个交易日
# ============================================================

def get_month_end_dates(
    close: pd.DataFrame,
) -> pd.DatetimeIndex:
    """
    返回每个月在当前数据中的最后一个交易日。
    """

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
    1. 使用最近10个交易日动量；
    2. hs300收盘价必须高于MA200；
    3. ETF动量必须严格大于4%；
    4. 选择动量排名前top_n的ETF；
    5. 在实际入选ETF之间动态等权；
    6. 没有符合条件的ETF时保持空仓。

    动态等权示例：

    Top4模式：
    - 4只合格：每只25%
    - 3只合格：每只33.33%
    - 2只合格：每只50%
    - 1只合格：该只100%
    - 0只合格：全部现金
    """

    if top_n <= 0:
        raise ValueError("top_n必须大于0。")

    momentum = close.pct_change(
        periods=MOMENTUM_DAYS,
        fill_method=None,
    )

    hs300_ma = (
        close[MARKET_FILTER_ETF]
        .rolling(window=MA_DAYS)
        .mean()
    )

    month_end_dates = get_month_end_dates(close)

    target_weights = pd.DataFrame(
        0.0,
        index=month_end_dates,
        columns=close.columns,
    )

    for signal_date in month_end_dates:
        hs300_price = close.at[
            signal_date,
            MARKET_FILTER_ETF,
        ]

        hs300_ma_value = hs300_ma.at[signal_date]

        # MA200尚未形成
        if pd.isna(hs300_ma_value):
            continue

        # hs300价格缺失
        if pd.isna(hs300_price):
            continue

        # 市场趋势过滤：hs300必须严格高于MA200
        if hs300_price <= hs300_ma_value:
            continue

        current_momentum = (
            momentum.loc[signal_date]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

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

        if selected_count == 0:
            continue

        # 动态等权：在实际入选ETF之间分配100%仓位
        equal_weight = 1.0 / selected_count

        target_weights.loc[
            signal_date,
            selected,
        ] = equal_weight

    return target_weights


# ============================================================
# 将月末信号转换成每日实际持仓
# ============================================================

def build_daily_weights(
    close: pd.DataFrame,
    monthly_target_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    月末产生信号，下一个交易日生效。

    例如：
    - 1月最后一个交易日收盘后生成信号；
    - 2月第一个交易日使用新持仓；
    - 持仓一直延续到下一个信号生效日。
    """

    daily_weights = pd.DataFrame(
        np.nan,
        index=close.index,
        columns=close.columns,
    )

    for signal_date, target_weight in (
        monthly_target_weights.iterrows()
    ):
        signal_position = close.index.get_indexer(
            [signal_date]
        )[0]

        if signal_position == -1:
            continue

        next_position = signal_position + 1

        # 最后一个交易日产生的信号没有下一交易日
        if next_position >= len(close.index):
            continue

        effective_date = close.index[next_position]

        daily_weights.loc[
            effective_date
        ] = target_weight.values

    # 信号生效后持续持有
    daily_weights = daily_weights.ffill()

    # 第一个有效信号出现之前全部为空仓
    daily_weights = daily_weights.fillna(0.0)

    # 浮点误差保护
    daily_weights = daily_weights.clip(
        lower=0.0,
        upper=1.0,
    )

    return daily_weights


# ============================================================
# 计算每日买卖换手和交易成本
# ============================================================

def calculate_daily_costs(
    daily_weights: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    根据目标权重变化计算交易成本。

    权重增加：
        视为买入，收取买入成本。

    权重减少：
        视为卖出，收取卖出成本。
    """

    previous_weights = (
        daily_weights
        .shift(1)
        .fillna(0.0)
    )

    weight_change = (
        daily_weights
        - previous_weights
    )

    buy_turnover = (
        weight_change
        .clip(lower=0.0)
        .sum(axis=1)
    )

    sell_turnover = (
        -weight_change
        .clip(upper=0.0)
        .sum(axis=1)
    )

    daily_cost = (
        buy_turnover * BUY_COST
        + sell_turnover * SELL_COST
    )

    return (
        daily_cost,
        buy_turnover,
        sell_turnover,
    )


# ============================================================
# 运行单组回测
# ============================================================

def run_backtest(
    close: pd.DataFrame,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    运行指定Top N组合回测。

    返回：
    1. 每日回测结果；
    2. 每日持仓权重；
    3. 月末信号权重。
    """

    monthly_weights = generate_monthly_target_weights(
        close=close,
        top_n=top_n,
    )

    daily_weights = build_daily_weights(
        close=close,
        monthly_target_weights=monthly_weights,
    )

    asset_returns = (
        close
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # 当日持仓承担当日完整收益
    gross_return = (
        daily_weights
        * asset_returns
    ).sum(axis=1)

    (
        daily_cost,
        buy_turnover,
        sell_turnover,
    ) = calculate_daily_costs(daily_weights)

    net_return = gross_return - daily_cost

    net_value = (
        1.0 + net_return
    ).cumprod()

    invested_weight = daily_weights.sum(axis=1)

    cash_weight = (
        1.0 - invested_weight
    ).clip(lower=0.0, upper=1.0)

    holding_count = (
        daily_weights > 1e-12
    ).sum(axis=1)

    result = pd.DataFrame({
        "gross_return": gross_return,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "cost": daily_cost,
        "net_return": net_return,
        "net_value": net_value,
        "invested_weight": invested_weight,
        "cash_weight": cash_weight,
        "holding_count": holding_count,
    })

    return (
        result,
        daily_weights,
        monthly_weights,
    )


# ============================================================
# 绩效统计
# ============================================================

def calculate_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    top_n: int,
) -> dict:
    """
    计算策略绩效指标。
    """

    net_return = result["net_return"]
    net_value = result["net_value"]

    final_value = net_value.iloc[-1]
    cumulative_return = final_value - 1.0

    # 使用自然日口径计算年化收益，
    # 与之前v1.1结果保持一致
    calendar_days = (
        result.index[-1]
        - result.index[0]
    ).days

    years = calendar_days / 365.25

    if years > 0 and final_value > 0:
        annual_return = (
            final_value ** (1.0 / years)
            - 1.0
        )
    else:
        annual_return = np.nan

    annual_volatility = (
        net_return.std(ddof=1)
        * np.sqrt(252)
    )

    if (
        pd.notna(annual_volatility)
        and annual_volatility > 0
    ):
        sharpe = (
            annual_return
            / annual_volatility
        )
    else:
        sharpe = np.nan

    rolling_max = net_value.cummax()

    drawdown = (
        net_value
        / rolling_max
        - 1.0
    )

    max_drawdown = drawdown.min()

    max_drawdown_end = drawdown.idxmin()

    if pd.notna(max_drawdown_end):
        peak_series = net_value.loc[
            :max_drawdown_end
        ]

        max_drawdown_start = peak_series.idxmax()
    else:
        max_drawdown_start = pd.NaT

    cash_ratio = (
        result["cash_weight"]
        .clip(lower=0.0, upper=1.0)
        .mean()
    )

    average_invested_weight = (
        result["invested_weight"]
        .clip(lower=0.0, upper=1.0)
        .mean()
    )

    holding_count = result["holding_count"]

    average_holding_count = holding_count.mean()

    maximum_holding_count = int(
        holding_count.max()
    )

    invested_days = holding_count > 0

    if invested_days.any():
        average_holding_count_when_invested = (
            holding_count[invested_days]
            .mean()
        )
    else:
        average_holding_count_when_invested = 0.0

    invested_day_ratio = invested_days.mean()

    weight_change = (
        daily_weights
        .diff()
        .abs()
        .sum(axis=1)
    )

    rebalance_days = int(
        (weight_change > 1e-12).sum()
    )

    total_buy_turnover = (
        result["buy_turnover"].sum()
    )

    total_sell_turnover = (
        result["sell_turnover"].sum()
    )

    total_turnover = (
        total_buy_turnover
        + total_sell_turnover
    )

    total_cost = result["cost"].sum()

    # 统计真正产生持仓信号的月数
    monthly_invested_weight = (
        monthly_weights.sum(axis=1)
    )

    invested_signal_count = int(
        (
            monthly_invested_weight > 1e-12
        ).sum()
    )

    cash_signal_count = int(
        (
            monthly_invested_weight <= 1e-12
        ).sum()
    )

    monthly_holding_count = (
        monthly_weights > 1e-12
    ).sum(axis=1)

    active_months = monthly_holding_count > 0

    if active_months.any():
        average_selected_count_when_active = (
            monthly_holding_count[active_months]
            .mean()
        )
    else:
        average_selected_count_when_active = 0.0

    return {
        "top_n": top_n,
        "final_value": final_value,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_start": max_drawdown_start,
        "max_drawdown_end": max_drawdown_end,
        "cash_ratio": cash_ratio,
        "average_invested_weight": average_invested_weight,
        "invested_day_ratio": invested_day_ratio,
        "average_holding_count": average_holding_count,
        "average_holding_count_when_invested":
            average_holding_count_when_invested,
        "maximum_holding_count": maximum_holding_count,
        "average_selected_count_when_active":
            average_selected_count_when_active,
        "rebalance_days": rebalance_days,
        "invested_signal_count": invested_signal_count,
        "cash_signal_count": cash_signal_count,
        "total_buy_turnover": total_buy_turnover,
        "total_sell_turnover": total_sell_turnover,
        "total_turnover": total_turnover,
        "total_cost": total_cost,
    }


# ============================================================
# 打印单组结果
# ============================================================

def print_metrics(metrics: dict) -> None:
    print("=" * 65)
    print(
        f"Top {metrics['top_n']} "
        f"动态等权多ETF持仓"
    )
    print("=" * 65)

    print(
        f"最终净值：           "
        f"{metrics['final_value']:.4f}"
    )

    print(
        f"累计收益：           "
        f"{metrics['cumulative_return']:.2%}"
    )

    print(
        f"年化收益：           "
        f"{metrics['annual_return']:.2%}"
    )

    print(
        f"年化波动：           "
        f"{metrics['annual_volatility']:.2%}"
    )

    print(
        f"夏普比率：           "
        f"{metrics['sharpe']:.2f}"
    )

    print(
        f"最大回撤：           "
        f"{metrics['max_drawdown']:.2%}"
    )

    print(
        f"最大回撤开始：       "
        f"{metrics['max_drawdown_start'].date()}"
    )

    print(
        f"最大回撤结束：       "
        f"{metrics['max_drawdown_end'].date()}"
    )

    print(
        f"平均现金比例：       "
        f"{metrics['cash_ratio']:.2%}"
    )

    print(
        f"平均投入仓位：       "
        f"{metrics['average_invested_weight']:.2%}"
    )

    print(
        f"实际持仓天数比例：   "
        f"{metrics['invested_day_ratio']:.2%}"
    )

    print(
        f"全样本平均持仓数量： "
        f"{metrics['average_holding_count']:.2f}"
    )

    print(
        f"持仓期间平均数量：   "
        f"{metrics['average_holding_count_when_invested']:.2f}"
    )

    print(
        f"活跃月份平均入选数： "
        f"{metrics['average_selected_count_when_active']:.2f}"
    )

    print(
        f"最大持仓数量：       "
        f"{metrics['maximum_holding_count']}"
    )

    print(
        f"调仓交易日数：       "
        f"{metrics['rebalance_days']}"
    )

    print(
        f"有持仓信号月份：     "
        f"{metrics['invested_signal_count']}"
    )

    print(
        f"空仓信号月份：       "
        f"{metrics['cash_signal_count']}"
    )

    print(
        f"累计买入换手：       "
        f"{metrics['total_buy_turnover']:.2f}"
    )

    print(
        f"累计卖出换手：       "
        f"{metrics['total_sell_turnover']:.2f}"
    )

    print(
        f"累计双边换手：       "
        f"{metrics['total_turnover']:.2f}"
    )

    print(
        f"累计成本率：         "
        f"{metrics['total_cost']:.2%}"
    )


# ============================================================
# 保存持仓明细
# ============================================================

def build_holding_details(
    daily_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    将每日权重转换成较容易查看的持仓明细。
    """

    details = pd.DataFrame(
        index=daily_weights.index
    )

    details["holding_count"] = (
        daily_weights > 1e-12
    ).sum(axis=1)

    details["invested_weight"] = (
        daily_weights.sum(axis=1)
    )

    details["cash_weight"] = (
        1.0 - details["invested_weight"]
    ).clip(lower=0.0, upper=1.0)

    def get_holding_names(
        row: pd.Series,
    ) -> str:
        holdings = row[row > 1e-12]

        if holdings.empty:
            return "cash"

        return ",".join(holdings.index.tolist())

    def get_holding_weights(
        row: pd.Series,
    ) -> str:
        holdings = row[row > 1e-12]

        if holdings.empty:
            return ""

        return ",".join(
            f"{name}:{weight:.4f}"
            for name, weight in holdings.items()
        )

    details["holdings"] = daily_weights.apply(
        get_holding_names,
        axis=1,
    )

    details["holding_weights"] = daily_weights.apply(
        get_holding_weights,
        axis=1,
    )

    return details


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    close = load_close_data(DATA_FILE)

    print(f"数据文件：    {DATA_FILE}")
    print(
        f"数据开始日期："
        f"{close.index.min().date()}"
    )
    print(
        f"数据结束日期："
        f"{close.index.max().date()}"
    )
    print(f"交易日数量：  {len(close)}")
    print(f"ETF数量：     {len(close.columns)}")
    print(
        f"ETF列表：     "
        f"{', '.join(close.columns)}"
    )
    print()
    print(
        f"动量窗口：    {MOMENTUM_DAYS}日"
    )
    print(
        f"动量门槛：    "
        f"{MOMENTUM_THRESHOLD:.2%}"
    )
    print(
        f"市场过滤：    "
        f"{MARKET_FILTER_ETF} MA{MA_DAYS}"
    )
    print(
        f"买入成本：    {BUY_COST:.2%}"
    )
    print(
        f"卖出成本：    {SELL_COST:.2%}"
    )
    print(
        "仓位方式：    动态等权"
    )
    print()

    all_metrics = []

    for top_n in TOP_N_LIST:
        (
            result,
            daily_weights,
            monthly_weights,
        ) = run_backtest(
            close=close,
            top_n=top_n,
        )

        metrics = calculate_metrics(
            result=result,
            daily_weights=daily_weights,
            monthly_weights=monthly_weights,
            top_n=top_n,
        )

        all_metrics.append(metrics)

        print_metrics(metrics)
        print()

        result.to_csv(
            OUTPUT_DIR
            / f"backtest_dynamic_top{top_n}.csv",
            encoding="utf-8-sig",
        )

        daily_weights.to_csv(
            OUTPUT_DIR
            / f"weights_dynamic_top{top_n}.csv",
            encoding="utf-8-sig",
        )

        monthly_weights.to_csv(
            OUTPUT_DIR
            / f"monthly_signal_dynamic_top{top_n}.csv",
            encoding="utf-8-sig",
        )

        holding_details = build_holding_details(
            daily_weights
        )

        holding_details.to_csv(
            OUTPUT_DIR
            / f"holding_details_dynamic_top{top_n}.csv",
            encoding="utf-8-sig",
        )

    comparison = pd.DataFrame(all_metrics)

    comparison.to_csv(
        OUTPUT_DIR
        / "multi_etf_dynamic_comparison.csv",
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
        "average_holding_count_when_invested",
        "average_selected_count_when_active",
        "rebalance_days",
        "total_cost",
    ]

    print("=" * 125)
    print("v1.3 动态等权多ETF持仓对比")
    print("=" * 125)

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "final_value":
                    "{:.4f}".format,
                "annual_return":
                    "{:.2%}".format,
                "annual_volatility":
                    "{:.2%}".format,
                "sharpe":
                    "{:.2f}".format,
                "max_drawdown":
                    "{:.2%}".format,
                "cash_ratio":
                    "{:.2%}".format,
                "average_holding_count_when_invested":
                    "{:.2f}".format,
                "average_selected_count_when_active":
                    "{:.2f}".format,
                "total_cost":
                    "{:.2%}".format,
            },
        )
    )

    print()
    print(
        f"结果已经保存到：{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()