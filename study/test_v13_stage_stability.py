from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 参数配置
# ============================================================

DATA_FILE = Path("data/etf_close_clean.csv")
OUTPUT_DIR = Path("results/v1_3_stage_stability")

MOMENTUM_DAYS = 10
MA_DAYS = 200

BUY_COST = 0.001
SELL_COST = 0.001

MARKET_FILTER_ETF = "hs300"


# ============================================================
# 候选版本
# ============================================================

CANDIDATES = [
    {
        "name": "Top1_4.00",
        "top_n": 1,
        "momentum_threshold": 0.04,
        "description": "原激进基准：Top1 + 4.00%",
    },
    {
        "name": "Top3_4.00",
        "top_n": 3,
        "momentum_threshold": 0.04,
        "description": "当前主候选：Top3 + 4.00%",
    },
    {
        "name": "Top4_4.00",
        "top_n": 4,
        "momentum_threshold": 0.04,
        "description": "收益/回撤兼顾备选：Top4 + 4.00%",
    },
    {
        "name": "Top4_3.75",
        "top_n": 4,
        "momentum_threshold": 0.0375,
        "description": "稳健防守备选：Top4 + 3.75%",
    },
]


# ============================================================
# 分阶段区间
# ============================================================

STAGES = [
    {
        "stage": "2020_2022",
        "start": "2020-01-02",
        "end": "2022-12-31",
        "description": "阶段1：2020-2022",
    },
    {
        "stage": "2023_2024",
        "start": "2023-01-01",
        "end": "2024-12-31",
        "description": "阶段2：2023-2024",
    },
    {
        "stage": "2025_2026",
        "start": "2025-01-01",
        "end": "2026-05-29",
        "description": "阶段3：2025-2026",
    },
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
# 月末交易日
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
    momentum = close.pct_change(
        periods=MOMENTUM_DAYS,
        fill_method=None,
    )

    market_ma = (
        close[MARKET_FILTER_ETF]
        .rolling(MA_DAYS)
        .mean()
    )

    month_end_dates = get_month_end_dates(close)

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
# 月末信号转换为每日持仓
# ============================================================

def build_daily_weights(
    close: pd.DataFrame,
    monthly_weights: pd.DataFrame,
) -> pd.DataFrame:
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
# 计算任意区间绩效
# ============================================================

def calculate_period_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    period_start: str,
    period_end: str,
    candidate_name: str,
    top_n: int,
    momentum_threshold: float,
    stage_name: str,
    stage_description: str,
) -> dict:
    start = pd.to_datetime(period_start)
    end = pd.to_datetime(period_end)

    period_result = result.loc[
        (result.index >= start)
        & (result.index <= end)
    ].copy()

    period_weights = daily_weights.loc[
        period_result.index
    ].copy()

    period_monthly_weights = monthly_weights.loc[
        (monthly_weights.index >= start)
        & (monthly_weights.index <= end)
    ].copy()

    if period_result.empty:
        return {
            "candidate": candidate_name,
            "top_n": top_n,
            "momentum_threshold": momentum_threshold,
            "threshold_percent": momentum_threshold * 100.0,
            "stage": stage_name,
            "stage_description": stage_description,
            "start_date": start,
            "end_date": end,
            "trading_days": 0,
            "final_value": np.nan,
            "period_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
            "cash_ratio": np.nan,
            "invested_day_ratio": np.nan,
            "average_holding_count_when_invested": np.nan,
            "active_signal_count": 0,
            "total_cost": np.nan,
        }

    net_return = period_result["net_return"]

    period_value = (
        1.0 + net_return
    ).cumprod()

    final_value = period_value.iloc[-1]
    period_return = final_value - 1.0

    calendar_days = (
        period_result.index[-1]
        - period_result.index[0]
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

    rolling_peak = period_value.cummax()

    drawdown = (
        period_value
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

    invested_weight = (
        period_result["invested_weight"]
        .clip(lower=0.0, upper=1.0)
    )

    cash_weight = (
        period_result["cash_weight"]
        .clip(lower=0.0, upper=1.0)
    )

    cash_ratio = cash_weight.mean()

    invested_day_ratio = (
        invested_weight > 1e-12
    ).mean()

    holding_count = (
        period_result["holding_count"]
    )

    invested_days = holding_count > 0

    if invested_days.any():
        average_holding_count_when_invested = (
            holding_count[invested_days]
            .mean()
        )
    else:
        average_holding_count_when_invested = 0.0

    period_monthly_invested = (
        period_monthly_weights.sum(axis=1)
    )

    active_signal_count = int(
        (
            period_monthly_invested > 1e-12
        ).sum()
    )

    total_cost = (
        period_result["cost"].sum()
    )

    return {
        "candidate": candidate_name,
        "top_n": top_n,
        "momentum_threshold": momentum_threshold,
        "threshold_percent": momentum_threshold * 100.0,
        "stage": stage_name,
        "stage_description": stage_description,
        "start_date": period_result.index[0],
        "end_date": period_result.index[-1],
        "trading_days": len(period_result),
        "final_value": final_value,
        "period_return": period_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "cash_ratio": cash_ratio,
        "invested_day_ratio": invested_day_ratio,
        "average_holding_count_when_invested":
            average_holding_count_when_invested,
        "active_signal_count": active_signal_count,
        "total_cost": total_cost,
    }


# ============================================================
# 全样本绩效
# ============================================================

def calculate_full_sample_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    candidate_name: str,
    top_n: int,
    momentum_threshold: float,
) -> dict:
    return calculate_period_metrics(
        result=result,
        daily_weights=daily_weights,
        monthly_weights=monthly_weights,
        period_start=str(result.index[0].date()),
        period_end=str(result.index[-1].date()),
        candidate_name=candidate_name,
        top_n=top_n,
        momentum_threshold=momentum_threshold,
        stage_name="full_sample",
        stage_description="全样本",
    )


# ============================================================
# 建立透视表
# ============================================================

def build_stage_pivot(
    stage_results: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    pivot = stage_results.pivot(
        index="stage",
        columns="candidate",
        values=value_column,
    )

    return pivot


# ============================================================
# 打印表格
# ============================================================

def print_stage_table(
    title: str,
    table: pd.DataFrame,
    percent: bool = False,
    decimals: int = 2,
) -> None:
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)

    if percent:
        formatters = {
            column: "{:.2%}".format
            for column in table.columns
        }
    else:
        format_string = (
            "{:."
            + str(decimals)
            + "f}"
        )

        formatters = {
            column: format_string.format
            for column in table.columns
        }

    print(
        table.to_string(
            formatters=formatters
        )
    )


def print_ranking_by_stage(
    stage_results: pd.DataFrame,
) -> None:
    columns = [
        "stage",
        "candidate",
        "final_value",
        "period_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "cash_ratio",
        "invested_day_ratio",
        "active_signal_count",
        "average_holding_count_when_invested",
    ]

    formatters = {
        "final_value":
            "{:.4f}".format,
        "period_return":
            "{:.2%}".format,
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
        "invested_day_ratio":
            "{:.2%}".format,
        "average_holding_count_when_invested":
            "{:.2f}".format,
    }

    for stage in stage_results["stage"].unique():
        one_stage = stage_results[
            stage_results["stage"] == stage
        ].copy()

        one_stage = one_stage.sort_values(
            ["sharpe", "annual_return"],
            ascending=[False, False],
        )

        print()
        print("=" * 130)
        print(f"{stage}：按夏普排序")
        print("=" * 130)

        print(
            one_stage[columns].to_string(
                index=False,
                formatters=formatters,
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
    print("v1.3 候选版本分阶段稳定性验证")
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
    print()

    print("候选版本：")
    for candidate in CANDIDATES:
        print(
            f"- {candidate['name']}: "
            f"{candidate['description']}"
        )

    print()
    print("阶段划分：")
    for stage in STAGES:
        print(
            f"- {stage['stage']}: "
            f"{stage['start']} 至 {stage['end']}"
        )

    print()

    all_stage_metrics = []
    all_full_metrics = []

    for candidate in CANDIDATES:
        name = candidate["name"]
        top_n = candidate["top_n"]
        threshold = candidate["momentum_threshold"]

        (
            result,
            daily_weights,
            monthly_weights,
        ) = run_backtest(
            close=close,
            top_n=top_n,
            momentum_threshold=threshold,
        )

        full_metrics = calculate_full_sample_metrics(
            result=result,
            daily_weights=daily_weights,
            monthly_weights=monthly_weights,
            candidate_name=name,
            top_n=top_n,
            momentum_threshold=threshold,
        )

        all_full_metrics.append(full_metrics)

        result.to_csv(
            OUTPUT_DIR / f"{name}_daily_result.csv",
            encoding="utf-8-sig",
        )

        daily_weights.to_csv(
            OUTPUT_DIR / f"{name}_daily_weights.csv",
            encoding="utf-8-sig",
        )

        monthly_weights.to_csv(
            OUTPUT_DIR / f"{name}_monthly_weights.csv",
            encoding="utf-8-sig",
        )

        for stage in STAGES:
            stage_metrics = calculate_period_metrics(
                result=result,
                daily_weights=daily_weights,
                monthly_weights=monthly_weights,
                period_start=stage["start"],
                period_end=stage["end"],
                candidate_name=name,
                top_n=top_n,
                momentum_threshold=threshold,
                stage_name=stage["stage"],
                stage_description=stage["description"],
            )

            all_stage_metrics.append(
                stage_metrics
            )

    stage_results = pd.DataFrame(
        all_stage_metrics
    )

    full_results = pd.DataFrame(
        all_full_metrics
    )

    stage_results.to_csv(
        OUTPUT_DIR / "stage_stability_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_results.to_csv(
        OUTPUT_DIR / "full_sample_candidate_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 透视表
    # --------------------------------------------------------

    pivot_period_return = build_stage_pivot(
        stage_results,
        "period_return",
    )

    pivot_annual_return = build_stage_pivot(
        stage_results,
        "annual_return",
    )

    pivot_volatility = build_stage_pivot(
        stage_results,
        "annual_volatility",
    )

    pivot_sharpe = build_stage_pivot(
        stage_results,
        "sharpe",
    )

    pivot_drawdown = build_stage_pivot(
        stage_results,
        "max_drawdown",
    )

    pivot_calmar = build_stage_pivot(
        stage_results,
        "calmar",
    )

    pivot_cash = build_stage_pivot(
        stage_results,
        "cash_ratio",
    )

    pivot_active_signal = build_stage_pivot(
        stage_results,
        "active_signal_count",
    )

    pivot_holding_count = build_stage_pivot(
        stage_results,
        "average_holding_count_when_invested",
    )

    pivot_tables = {
        "pivot_stage_period_return.csv":
            pivot_period_return,
        "pivot_stage_annual_return.csv":
            pivot_annual_return,
        "pivot_stage_volatility.csv":
            pivot_volatility,
        "pivot_stage_sharpe.csv":
            pivot_sharpe,
        "pivot_stage_max_drawdown.csv":
            pivot_drawdown,
        "pivot_stage_calmar.csv":
            pivot_calmar,
        "pivot_stage_cash_ratio.csv":
            pivot_cash,
        "pivot_stage_active_signal_count.csv":
            pivot_active_signal,
        "pivot_stage_holding_count.csv":
            pivot_holding_count,
    }

    for filename, table in pivot_tables.items():
        table.to_csv(
            OUTPUT_DIR / filename,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # 全样本打印
    # --------------------------------------------------------

    print()
    print("=" * 130)
    print("全样本候选版本结果")
    print("=" * 130)

    full_display_columns = [
        "candidate",
        "final_value",
        "period_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "cash_ratio",
        "active_signal_count",
        "average_holding_count_when_invested",
        "total_cost",
    ]

    full_sorted = full_results.sort_values(
        ["sharpe", "annual_return"],
        ascending=[False, False],
    )

    print(
        full_sorted[
            full_display_columns
        ].to_string(
            index=False,
            formatters={
                "final_value":
                    "{:.4f}".format,
                "period_return":
                    "{:.2%}".format,
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
                "average_holding_count_when_invested":
                    "{:.2f}".format,
                "total_cost":
                    "{:.2%}".format,
            },
        )
    )

    # --------------------------------------------------------
    # 分阶段打印
    # --------------------------------------------------------

    print_stage_table(
        "分阶段累计收益",
        pivot_period_return,
        percent=True,
    )

    print_stage_table(
        "分阶段年化收益",
        pivot_annual_return,
        percent=True,
    )

    print_stage_table(
        "分阶段年化波动",
        pivot_volatility,
        percent=True,
    )

    print_stage_table(
        "分阶段夏普",
        pivot_sharpe,
        percent=False,
        decimals=2,
    )

    print_stage_table(
        "分阶段最大回撤",
        pivot_drawdown,
        percent=True,
    )

    print_stage_table(
        "分阶段Calmar",
        pivot_calmar,
        percent=False,
        decimals=2,
    )

    print_stage_table(
        "分阶段现金比例",
        pivot_cash,
        percent=True,
    )

    print_stage_table(
        "分阶段活跃信号月份数",
        pivot_active_signal,
        percent=False,
        decimals=0,
    )

    print_stage_table(
        "分阶段持仓期间平均ETF数量",
        pivot_holding_count,
        percent=False,
        decimals=2,
    )

    print_ranking_by_stage(
        stage_results
    )

    print()
    print(
        f"分阶段稳定性测试结果已保存到："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()