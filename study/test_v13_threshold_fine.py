from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 参数配置
# ============================================================

DATA_FILE = Path("data/etf_close_clean.csv")
OUTPUT_DIR = Path("results/v1_3_threshold_fine_test")

MOMENTUM_DAYS = 10
MA_DAYS = 200

BUY_COST = 0.001
SELL_COST = 0.001

MARKET_FILTER_ETF = "hs300"

# 这次只测 Top2、Top3、Top4
TOP_N_LIST = [2, 3, 4]

# 3.00% 到 5.00%，每 0.25% 一档
THRESHOLD_PERCENT_LIST = [
    3.00,
    3.25,
    3.50,
    3.75,
    4.00,
    4.25,
    4.50,
    4.75,
    5.00,
]

THRESHOLD_LIST = [
    value / 100.0
    for value in THRESHOLD_PERCENT_LIST
]


# ============================================================
# 读取数据
# ============================================================

def load_close_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{file_path}"
        )

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

    if close.empty:
        raise ValueError("收盘价数据为空。")

    if close.index.has_duplicates:
        raise ValueError("日期索引存在重复值。")

    if MARKET_FILTER_ETF not in close.columns:
        raise ValueError(
            f"数据中缺少市场过滤ETF："
            f"{MARKET_FILTER_ETF}"
        )

    return close


# ============================================================
# 获取月末交易日
# ============================================================

def get_month_end_dates(
    close: pd.DataFrame,
) -> pd.DatetimeIndex:
    month_period = close.index.to_period("M")

    month_end_dates = (
        close.groupby(month_period)
        .apply(lambda df: df.index[-1])
    )

    return pd.DatetimeIndex(
        month_end_dates.values
    )


# ============================================================
# 生成月末目标权重
# ============================================================

def generate_monthly_target_weights(
    close: pd.DataFrame,
    top_n: int,
    momentum_threshold: float,
) -> pd.DataFrame:
    """
    月末生成动态等权目标持仓。

    规则：
    1. 月末计算10日动量；
    2. hs300必须高于MA200；
    3. ETF动量必须严格大于指定门槛；
    4. 选择动量排名前Top N；
    5. 在实际入选ETF之间动态等权；
    6. 没有合格ETF则空仓。
    """

    momentum = close.pct_change(
        periods=MOMENTUM_DAYS,
        fill_method=None,
    )

    market_ma = (
        close[MARKET_FILTER_ETF]
        .rolling(MA_DAYS)
        .mean()
    )

    month_end_dates = get_month_end_dates(
        close
    )

    monthly_weights = pd.DataFrame(
        0.0,
        index=month_end_dates,
        columns=close.columns,
    )

    for signal_date in month_end_dates:
        market_price = close.at[
            signal_date,
            MARKET_FILTER_ETF,
        ]

        market_ma_value = market_ma.at[
            signal_date
        ]

        if pd.isna(market_price):
            continue

        if pd.isna(market_ma_value):
            continue

        if market_price <= market_ma_value:
            continue

        current_momentum = (
            momentum.loc[signal_date]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        qualified = current_momentum[
            current_momentum > momentum_threshold
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

        equal_weight = 1.0 / selected_count

        monthly_weights.loc[
            signal_date,
            selected,
        ] = equal_weight

    return monthly_weights


# ============================================================
# 月末信号转换为每日权重
# ============================================================

def build_daily_weights(
    close: pd.DataFrame,
    monthly_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    月末信号从下一个交易日开始生效。
    """

    daily_weights = pd.DataFrame(
        np.nan,
        index=close.index,
        columns=close.columns,
    )

    for signal_date, target_weight in (
        monthly_weights.iterrows()
    ):
        signal_location = close.index.get_indexer(
            [signal_date]
        )[0]

        if signal_location < 0:
            continue

        next_location = signal_location + 1

        if next_location >= len(close.index):
            continue

        effective_date = close.index[
            next_location
        ]

        daily_weights.loc[
            effective_date
        ] = target_weight.values

    daily_weights = (
        daily_weights
        .ffill()
        .fillna(0.0)
    )

    return daily_weights


# ============================================================
# 交易成本
# ============================================================

def calculate_costs(
    daily_weights: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
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

    cost = (
        buy_turnover * BUY_COST
        + sell_turnover * SELL_COST
    )

    return (
        cost,
        buy_turnover,
        sell_turnover,
    )


# ============================================================
# 回测
# ============================================================

def run_backtest(
    close: pd.DataFrame,
    top_n: int,
    momentum_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_weights = generate_monthly_target_weights(
        close=close,
        top_n=top_n,
        momentum_threshold=momentum_threshold,
    )

    daily_weights = build_daily_weights(
        close=close,
        monthly_weights=monthly_weights,
    )

    asset_returns = (
        close
        .pct_change(fill_method=None)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    gross_return = (
        daily_weights
        * asset_returns
    ).sum(axis=1)

    (
        cost,
        buy_turnover,
        sell_turnover,
    ) = calculate_costs(
        daily_weights
    )

    net_return = gross_return - cost

    net_value = (
        1.0 + net_return
    ).cumprod()

    invested_weight = (
        daily_weights.sum(axis=1)
    )

    cash_weight = (
        1.0 - invested_weight
    )

    holding_count = (
        daily_weights > 1e-12
    ).sum(axis=1)

    result = pd.DataFrame({
        "gross_return": gross_return,
        "cost": cost,
        "net_return": net_return,
        "net_value": net_value,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
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
# 绩效指标
# ============================================================

def calculate_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    top_n: int,
    momentum_threshold: float,
) -> dict:
    net_return = result["net_return"]
    net_value = result["net_value"]

    final_value = net_value.iloc[-1]
    cumulative_return = final_value - 1.0

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

    rolling_peak = net_value.cummax()

    drawdown = (
        net_value
        / rolling_peak
        - 1.0
    )

    max_drawdown = drawdown.min()

    if max_drawdown < 0:
        calmar = (
            annual_return
            / abs(max_drawdown)
        )
    else:
        calmar = np.nan

    max_drawdown_end = drawdown.idxmin()

    if pd.notna(max_drawdown_end):
        max_drawdown_start = (
            net_value.loc[:max_drawdown_end]
            .idxmax()
        )
    else:
        max_drawdown_start = pd.NaT

    invested_weight = (
        result["invested_weight"]
        .clip(lower=0.0, upper=1.0)
    )

    cash_weight = (
        result["cash_weight"]
        .clip(lower=0.0, upper=1.0)
    )

    cash_ratio = cash_weight.mean()

    average_invested_weight = (
        invested_weight.mean()
    )

    invested_day_ratio = (
        invested_weight > 1e-12
    ).mean()

    holding_count = result["holding_count"]

    average_holding_count = (
        holding_count.mean()
    )

    invested_days = holding_count > 0

    if invested_days.any():
        average_holding_count_when_invested = (
            holding_count[invested_days]
            .mean()
        )
    else:
        average_holding_count_when_invested = 0.0

    maximum_holding_count = int(
        holding_count.max()
    )

    monthly_invested_weight = (
        monthly_weights.sum(axis=1)
    )

    active_signal_mask = (
        monthly_invested_weight > 1e-12
    )

    active_signal_count = int(
        active_signal_mask.sum()
    )

    cash_signal_count = int(
        (~active_signal_mask).sum()
    )

    monthly_holding_count = (
        monthly_weights > 1e-12
    ).sum(axis=1)

    if active_signal_mask.any():
        average_selected_count_when_active = (
            monthly_holding_count[
                active_signal_mask
            ].mean()
        )
    else:
        average_selected_count_when_active = 0.0

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

    total_cost = (
        result["cost"].sum()
    )

    return {
        "top_n":
            top_n,
        "momentum_threshold":
            momentum_threshold,
        "threshold_percent":
            momentum_threshold * 100.0,
        "final_value":
            final_value,
        "cumulative_return":
            cumulative_return,
        "annual_return":
            annual_return,
        "annual_volatility":
            annual_volatility,
        "sharpe":
            sharpe,
        "max_drawdown":
            max_drawdown,
        "calmar":
            calmar,
        "max_drawdown_start":
            max_drawdown_start,
        "max_drawdown_end":
            max_drawdown_end,
        "cash_ratio":
            cash_ratio,
        "average_invested_weight":
            average_invested_weight,
        "invested_day_ratio":
            invested_day_ratio,
        "average_holding_count":
            average_holding_count,
        "average_holding_count_when_invested":
            average_holding_count_when_invested,
        "maximum_holding_count":
            maximum_holding_count,
        "average_selected_count_when_active":
            average_selected_count_when_active,
        "active_signal_count":
            active_signal_count,
        "cash_signal_count":
            cash_signal_count,
        "rebalance_days":
            rebalance_days,
        "total_buy_turnover":
            total_buy_turnover,
        "total_sell_turnover":
            total_sell_turnover,
        "total_turnover":
            total_turnover,
        "total_cost":
            total_cost,
    }


# ============================================================
# 参数矩阵
# ============================================================

def build_pivot_table(
    all_results: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    pivot = all_results.pivot(
        index="top_n",
        columns="threshold_percent",
        values=value_column,
    )

    pivot = pivot.sort_index()
    pivot = pivot.sort_index(axis=1)

    pivot.index.name = "top_n"
    pivot.columns.name = "threshold_percent"

    return pivot


# ============================================================
# 打印矩阵
# ============================================================

def print_percent_pivot(
    title: str,
    pivot: pd.DataFrame,
) -> None:
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)

    display = pivot.copy()

    display.columns = [
        f"{column:.2f}%"
        for column in display.columns
    ]

    print(
        display.to_string(
            formatters={
                column: "{:.2%}".format
                for column in display.columns
            }
        )
    )


def print_number_pivot(
    title: str,
    pivot: pd.DataFrame,
    decimal_places: int = 2,
) -> None:
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)

    display = pivot.copy()

    display.columns = [
        f"{column:.2f}%"
        for column in display.columns
    ]

    format_string = (
        "{:."
        + str(decimal_places)
        + "f}"
    )

    print(
        display.to_string(
            formatters={
                column: format_string.format
                for column in display.columns
            }
        )
    )


# ============================================================
# 打印单组结果
# ============================================================

def print_single_result(
    metrics: dict,
) -> None:
    print(
        f"Top{metrics['top_n']} | "
        f"门槛 {metrics['threshold_percent']:.2f}% | "
        f"净值 {metrics['final_value']:.4f} | "
        f"年化 {metrics['annual_return']:.2%} | "
        f"波动 {metrics['annual_volatility']:.2%} | "
        f"夏普 {metrics['sharpe']:.2f} | "
        f"回撤 {metrics['max_drawdown']:.2%} | "
        f"Calmar {metrics['calmar']:.2f} | "
        f"现金 {metrics['cash_ratio']:.2%}"
    )


# ============================================================
# 排名打印
# ============================================================

def print_rankings(
    all_results: pd.DataFrame,
) -> None:
    columns = [
        "top_n",
        "threshold_percent",
        "final_value",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "cash_ratio",
        "active_signal_count",
        "average_selected_count_when_active",
        "total_cost",
    ]

    formatters = {
        "threshold_percent":
            "{:.2f}%".format,
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
        "calmar":
            "{:.2f}".format,
        "cash_ratio":
            "{:.2%}".format,
        "average_selected_count_when_active":
            "{:.2f}".format,
        "total_cost":
            "{:.2%}".format,
    }

    print()
    print("=" * 135)
    print("夏普最高的10组细分参数")
    print("=" * 135)

    top_sharpe = (
        all_results
        .sort_values(
            ["sharpe", "annual_return"],
            ascending=[False, False],
        )
        .head(10)
    )

    print(
        top_sharpe[columns].to_string(
            index=False,
            formatters=formatters,
        )
    )

    print()
    print("=" * 135)
    print("年化收益最高的10组细分参数")
    print("=" * 135)

    top_return = (
        all_results
        .sort_values(
            ["annual_return", "sharpe"],
            ascending=[False, False],
        )
        .head(10)
    )

    print(
        top_return[columns].to_string(
            index=False,
            formatters=formatters,
        )
    )

    print()
    print("=" * 135)
    print("Calmar最高的10组细分参数")
    print("=" * 135)

    top_calmar = (
        all_results
        .sort_values(
            ["calmar", "annual_return"],
            ascending=[False, False],
        )
        .head(10)
    )

    print(
        top_calmar[columns].to_string(
            index=False,
            formatters=formatters,
        )
    )


# ============================================================
# 3.75% ~ 4.25% 核心区域
# ============================================================

def print_core_area(
    all_results: pd.DataFrame,
) -> None:
    core = all_results[
        all_results["threshold_percent"].isin(
            [3.75, 4.00, 4.25]
        )
    ].copy()

    core = core.sort_values(
        [
            "threshold_percent",
            "top_n",
        ]
    )

    columns = [
        "top_n",
        "threshold_percent",
        "final_value",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "cash_ratio",
        "active_signal_count",
        "average_selected_count_when_active",
    ]

    print()
    print("=" * 135)
    print("核心区域：3.75% / 4.00% / 4.25%")
    print("=" * 135)

    print(
        core[columns].to_string(
            index=False,
            formatters={
                "threshold_percent":
                    "{:.2f}%".format,
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
                "calmar":
                    "{:.2f}".format,
                "cash_ratio":
                    "{:.2%}".format,
                "average_selected_count_when_active":
                    "{:.2f}".format,
            },
        )
    )


# ============================================================
# 局部稳定性
# ============================================================

def calculate_local_stability(
    all_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    对每个参数点，计算其局部邻域平均表现。

    邻域定义：
    - Top N 相差不超过1；
    - 门槛相差不超过0.25个百分点。
    """

    records = []

    for _, row in all_results.iterrows():
        top_n = int(row["top_n"])
        threshold_percent = float(
            row["threshold_percent"]
        )

        neighbor = all_results[
            (
                all_results["top_n"]
                .sub(top_n)
                .abs()
                <= 1
            )
            & (
                all_results[
                    "threshold_percent"
                ]
                .sub(threshold_percent)
                .abs()
                <= 0.25 + 1e-12
            )
        ]

        records.append({
            "top_n":
                top_n,
            "threshold_percent":
                threshold_percent,
            "neighbor_count":
                len(neighbor),
            "local_average_annual_return":
                neighbor[
                    "annual_return"
                ].mean(),
            "local_average_sharpe":
                neighbor[
                    "sharpe"
                ].mean(),
            "local_average_max_drawdown":
                neighbor[
                    "max_drawdown"
                ].mean(),
            "local_average_calmar":
                neighbor[
                    "calmar"
                ].mean(),
            "local_minimum_sharpe":
                neighbor[
                    "sharpe"
                ].min(),
            "local_worst_max_drawdown":
                neighbor[
                    "max_drawdown"
                ].min(),
        })

    return pd.DataFrame(records)


def print_local_stability(
    stability: pd.DataFrame,
) -> None:
    display = (
        stability
        .sort_values(
            [
                "local_average_sharpe",
                "local_average_annual_return",
            ],
            ascending=[False, False],
        )
        .head(10)
    )

    print()
    print("=" * 135)
    print("局部邻域平均夏普最高的10组")
    print("=" * 135)

    print(
        display.to_string(
            index=False,
            formatters={
                "threshold_percent":
                    "{:.2f}%".format,
                "local_average_annual_return":
                    "{:.2%}".format,
                "local_average_sharpe":
                    "{:.2f}".format,
                "local_average_max_drawdown":
                    "{:.2%}".format,
                "local_average_calmar":
                    "{:.2f}".format,
                "local_minimum_sharpe":
                    "{:.2f}".format,
                "local_worst_max_drawdown":
                    "{:.2%}".format,
            },
        )
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    close = load_close_data(DATA_FILE)

    print("=" * 90)
    print("v1.3 细分动量门槛测试")
    print("=" * 90)

    print(f"数据文件：      {DATA_FILE}")
    print(
        f"数据日期：      "
        f"{close.index.min().date()} "
        f"至 {close.index.max().date()}"
    )
    print(f"交易日数量：    {len(close)}")
    print(f"ETF数量：       {len(close.columns)}")
    print(f"动量窗口：      {MOMENTUM_DAYS}日")
    print(
        f"市场过滤：      "
        f"{MARKET_FILTER_ETF} MA{MA_DAYS}"
    )
    print(f"买入成本：      {BUY_COST:.2%}")
    print(f"卖出成本：      {SELL_COST:.2%}")
    print("仓位方式：      动态等权")
    print(f"Top N范围：     {TOP_N_LIST}")
    print(
        "门槛范围：      "
        + ", ".join(
            f"{value:.2f}%"
            for value in THRESHOLD_PERCENT_LIST
        )
    )
    print()

    all_metrics = []

    total_combinations = (
        len(TOP_N_LIST)
        * len(THRESHOLD_LIST)
    )

    completed = 0

    for top_n in TOP_N_LIST:
        for threshold in THRESHOLD_LIST:
            completed += 1

            (
                result,
                daily_weights,
                monthly_weights,
            ) = run_backtest(
                close=close,
                top_n=top_n,
                momentum_threshold=threshold,
            )

            metrics = calculate_metrics(
                result=result,
                daily_weights=daily_weights,
                monthly_weights=monthly_weights,
                top_n=top_n,
                momentum_threshold=threshold,
            )

            all_metrics.append(metrics)

            print(
                f"[{completed:02d}/"
                f"{total_combinations:02d}] ",
                end="",
            )

            print_single_result(metrics)

    all_results = pd.DataFrame(all_metrics)

    all_results = all_results.sort_values(
        [
            "top_n",
            "threshold_percent",
        ]
    ).reset_index(drop=True)

    all_results.to_csv(
        OUTPUT_DIR / "fine_threshold_all_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 生成矩阵
    # --------------------------------------------------------

    final_value_pivot = build_pivot_table(
        all_results,
        "final_value",
    )

    annual_return_pivot = build_pivot_table(
        all_results,
        "annual_return",
    )

    volatility_pivot = build_pivot_table(
        all_results,
        "annual_volatility",
    )

    sharpe_pivot = build_pivot_table(
        all_results,
        "sharpe",
    )

    drawdown_pivot = build_pivot_table(
        all_results,
        "max_drawdown",
    )

    calmar_pivot = build_pivot_table(
        all_results,
        "calmar",
    )

    cash_ratio_pivot = build_pivot_table(
        all_results,
        "cash_ratio",
    )

    active_signal_pivot = build_pivot_table(
        all_results,
        "active_signal_count",
    )

    selected_count_pivot = build_pivot_table(
        all_results,
        "average_selected_count_when_active",
    )

    pivot_tables = {
        "pivot_final_value.csv":
            final_value_pivot,
        "pivot_annual_return.csv":
            annual_return_pivot,
        "pivot_annual_volatility.csv":
            volatility_pivot,
        "pivot_sharpe.csv":
            sharpe_pivot,
        "pivot_max_drawdown.csv":
            drawdown_pivot,
        "pivot_calmar.csv":
            calmar_pivot,
        "pivot_cash_ratio.csv":
            cash_ratio_pivot,
        "pivot_active_signal_count.csv":
            active_signal_pivot,
        "pivot_average_selected_count.csv":
            selected_count_pivot,
    }

    for filename, pivot in pivot_tables.items():
        pivot.to_csv(
            OUTPUT_DIR / filename,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # 局部稳定性
    # --------------------------------------------------------

    stability = calculate_local_stability(
        all_results
    )

    stability.to_csv(
        OUTPUT_DIR / "fine_local_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_local = (
        stability
        .sort_values(
            [
                "local_average_sharpe",
                "local_average_annual_return",
            ],
            ascending=[False, False],
        )
        .head(10)
    )

    best_local.to_csv(
        OUTPUT_DIR / "fine_best_local_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 保存排名
    # --------------------------------------------------------

    all_results.sort_values(
        ["sharpe", "annual_return"],
        ascending=[False, False],
    ).to_csv(
        OUTPUT_DIR / "ranking_by_sharpe.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_results.sort_values(
        ["annual_return", "sharpe"],
        ascending=[False, False],
    ).to_csv(
        OUTPUT_DIR / "ranking_by_annual_return.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_results.sort_values(
        ["calmar", "annual_return"],
        ascending=[False, False],
    ).to_csv(
        OUTPUT_DIR / "ranking_by_calmar.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 打印矩阵
    # --------------------------------------------------------

    print_number_pivot(
        "最终净值矩阵",
        final_value_pivot,
        decimal_places=4,
    )

    print_percent_pivot(
        "年化收益矩阵",
        annual_return_pivot,
    )

    print_percent_pivot(
        "年化波动矩阵",
        volatility_pivot,
    )

    print_number_pivot(
        "夏普矩阵",
        sharpe_pivot,
        decimal_places=2,
    )

    print_percent_pivot(
        "最大回撤矩阵",
        drawdown_pivot,
    )

    print_number_pivot(
        "Calmar矩阵",
        calmar_pivot,
        decimal_places=2,
    )

    print_percent_pivot(
        "平均现金比例矩阵",
        cash_ratio_pivot,
    )

    print_number_pivot(
        "活跃信号月份数量矩阵",
        active_signal_pivot,
        decimal_places=0,
    )

    print_number_pivot(
        "活跃月份平均入选ETF数量矩阵",
        selected_count_pivot,
        decimal_places=2,
    )

    print_core_area(all_results)

    print_rankings(all_results)

    print_local_stability(stability)

    print()
    print("=" * 90)
    print("当前候选检查：Top3 + 4.00%")
    print("=" * 90)

    candidate = all_results[
        (all_results["top_n"] == 3)
        & (
            np.isclose(
                all_results[
                    "threshold_percent"
                ],
                4.00,
            )
        )
    ]

    if candidate.empty:
        print("未找到 Top3 + 4.00% 结果。")
    else:
        print(
            candidate[
                [
                    "top_n",
                    "threshold_percent",
                    "final_value",
                    "annual_return",
                    "annual_volatility",
                    "sharpe",
                    "max_drawdown",
                    "calmar",
                    "cash_ratio",
                    "active_signal_count",
                    "average_selected_count_when_active",
                ]
            ].to_string(
                index=False,
                formatters={
                    "threshold_percent":
                        "{:.2f}%".format,
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
                    "calmar":
                        "{:.2f}".format,
                    "cash_ratio":
                        "{:.2%}".format,
                    "average_selected_count_when_active":
                        "{:.2f}".format,
                },
            )
        )

    print()
    print(
        f"测试结果已保存到："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()