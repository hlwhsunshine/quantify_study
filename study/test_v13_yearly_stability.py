from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 参数配置
# ============================================================

DATA_FILE = Path("data/etf_close_clean.csv")
OUTPUT_DIR = Path("results/v1_3_yearly_stability")

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
# 数据读取
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
# 计算某一年绩效
# ============================================================

def calculate_year_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    year: int,
    candidate_name: str,
    top_n: int,
    momentum_threshold: float,
) -> dict:
    year_mask = result.index.year == year

    year_result = result.loc[year_mask].copy()
    year_weights = daily_weights.loc[year_result.index].copy()

    year_monthly_weights = monthly_weights.loc[
        monthly_weights.index.year == year
    ].copy()

    if year_result.empty:
        return {
            "year": year,
            "candidate": candidate_name,
            "top_n": top_n,
            "momentum_threshold": momentum_threshold,
            "threshold_percent": momentum_threshold * 100.0,
            "trading_days": 0,
            "final_value": np.nan,
            "year_return": np.nan,
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

    net_return = year_result["net_return"]

    year_value = (
        1.0 + net_return
    ).cumprod()

    final_value = year_value.iloc[-1]
    year_return = final_value - 1.0

    calendar_days = (
        year_result.index[-1]
        - year_result.index[0]
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
        and annual_volatility > 1e-12
    ):
        sharpe = annual_return / annual_volatility
    else:
        sharpe = np.nan

    rolling_peak = year_value.cummax()

    drawdown = (
        year_value
        / rolling_peak
        - 1.0
    )

    max_drawdown = drawdown.min()

    if max_drawdown < -1e-12:
        calmar = (
            annual_return
            / abs(max_drawdown)
        )
    else:
        calmar = np.nan

    invested_weight = (
        year_result["invested_weight"]
        .clip(lower=0.0, upper=1.0)
    )

    cash_weight = (
        year_result["cash_weight"]
        .clip(lower=0.0, upper=1.0)
    )

    cash_ratio = cash_weight.mean()

    invested_day_ratio = (
        invested_weight > 1e-12
    ).mean()

    holding_count = year_result["holding_count"]
    invested_days = holding_count > 0

    if invested_days.any():
        average_holding_count_when_invested = (
            holding_count[invested_days]
            .mean()
        )
    else:
        average_holding_count_when_invested = 0.0

    monthly_invested = (
        year_monthly_weights.sum(axis=1)
    )

    active_signal_count = int(
        (monthly_invested > 1e-12).sum()
    )

    total_cost = year_result["cost"].sum()

    return {
        "year": year,
        "candidate": candidate_name,
        "top_n": top_n,
        "momentum_threshold": momentum_threshold,
        "threshold_percent": momentum_threshold * 100.0,
        "trading_days": len(year_result),
        "final_value": final_value,
        "year_return": year_return,
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

def calculate_full_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
    monthly_weights: pd.DataFrame,
    candidate_name: str,
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

    annual_return = (
        final_value ** (1.0 / years)
        - 1.0
    )

    annual_volatility = (
        net_return.std(ddof=1)
        * np.sqrt(252)
    )

    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 1e-12
        else np.nan
    )

    rolling_peak = net_value.cummax()

    drawdown = (
        net_value
        / rolling_peak
        - 1.0
    )

    max_drawdown = drawdown.min()

    calmar = (
        annual_return / abs(max_drawdown)
        if max_drawdown < -1e-12
        else np.nan
    )

    monthly_invested = (
        monthly_weights.sum(axis=1)
    )

    active_signal_count = int(
        (monthly_invested > 1e-12).sum()
    )

    holding_count = result["holding_count"]
    invested_days = holding_count > 0

    if invested_days.any():
        average_holding_count_when_invested = (
            holding_count[invested_days]
            .mean()
        )
    else:
        average_holding_count_when_invested = 0.0

    return {
        "candidate": candidate_name,
        "top_n": top_n,
        "momentum_threshold": momentum_threshold,
        "threshold_percent": momentum_threshold * 100.0,
        "final_value": final_value,
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "cash_ratio": result["cash_weight"].mean(),
        "active_signal_count": active_signal_count,
        "average_holding_count_when_invested":
            average_holding_count_when_invested,
        "total_cost": result["cost"].sum(),
    }


# ============================================================
# 年度稳定性汇总
# ============================================================

def build_yearly_summary(
    yearly_results: pd.DataFrame,
) -> pd.DataFrame:
    records = []

    for candidate, group in yearly_results.groupby("candidate"):
        returns = group["year_return"].dropna()

        positive_years = int(
            (returns > 1e-12).sum()
        )

        negative_years = int(
            (returns < -1e-12).sum()
        )

        flat_years = int(
            (
                returns.abs() <= 1e-12
            ).sum()
        )

        best_row = group.loc[
            group["year_return"].idxmax()
        ]

        worst_row = group.loc[
            group["year_return"].idxmin()
        ]

        records.append({
            "candidate": candidate,
            "year_count": len(returns),
            "positive_years": positive_years,
            "negative_years": negative_years,
            "flat_years": flat_years,
            "average_year_return":
                returns.mean(),
            "median_year_return":
                returns.median(),
            "year_return_std":
                returns.std(ddof=1),
            "best_year":
                int(best_row["year"]),
            "best_year_return":
                best_row["year_return"],
            "worst_year":
                int(worst_row["year"]),
            "worst_year_return":
                worst_row["year_return"],
            "average_max_drawdown":
                group["max_drawdown"].mean(),
            "worst_year_drawdown":
                group["max_drawdown"].min(),
            "average_sharpe":
                group["sharpe"].mean(),
            "average_cash_ratio":
                group["cash_ratio"].mean(),
            "average_active_signal_count":
                group["active_signal_count"].mean(),
        })

    summary = pd.DataFrame(records)

    return summary


# ============================================================
# 透视表
# ============================================================

def build_year_pivot(
    yearly_results: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    pivot = yearly_results.pivot(
        index="year",
        columns="candidate",
        values=value_column,
    )

    return pivot


# ============================================================
# 打印工具
# ============================================================

def print_table(
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


def print_full_results(
    full_results: pd.DataFrame,
) -> None:
    columns = [
        "candidate",
        "final_value",
        "cumulative_return",
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

    display = full_results.sort_values(
        ["sharpe", "annual_return"],
        ascending=[False, False],
    )

    print()
    print("=" * 130)
    print("全样本候选版本结果")
    print("=" * 130)

    print(
        display[columns].to_string(
            index=False,
            formatters={
                "final_value":
                    "{:.4f}".format,
                "cumulative_return":
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


def print_yearly_summary(
    summary: pd.DataFrame,
) -> None:
    columns = [
        "candidate",
        "positive_years",
        "negative_years",
        "flat_years",
        "average_year_return",
        "median_year_return",
        "year_return_std",
        "best_year",
        "best_year_return",
        "worst_year",
        "worst_year_return",
        "average_max_drawdown",
        "worst_year_drawdown",
        "average_sharpe",
    ]

    display = summary.sort_values(
        [
            "average_year_return",
            "worst_year_return",
        ],
        ascending=[False, False],
    )

    print()
    print("=" * 150)
    print("年度稳定性汇总")
    print("=" * 150)

    print(
        display[columns].to_string(
            index=False,
            formatters={
                "average_year_return":
                    "{:.2%}".format,
                "median_year_return":
                    "{:.2%}".format,
                "year_return_std":
                    "{:.2%}".format,
                "best_year_return":
                    "{:.2%}".format,
                "worst_year_return":
                    "{:.2%}".format,
                "average_max_drawdown":
                    "{:.2%}".format,
                "worst_year_drawdown":
                    "{:.2%}".format,
                "average_sharpe":
                    "{:.2f}".format,
            },
        )
    )


def print_yearly_ranking(
    yearly_results: pd.DataFrame,
) -> None:
    columns = [
        "year",
        "candidate",
        "year_return",
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
        "year_return":
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

    for year in sorted(yearly_results["year"].unique()):
        one_year = yearly_results[
            yearly_results["year"] == year
        ].copy()

        one_year = one_year.sort_values(
            ["year_return", "sharpe"],
            ascending=[False, False],
        )

        print()
        print("=" * 130)
        print(f"{year} 年：按年度收益排序")
        print("=" * 130)

        print(
            one_year[columns].to_string(
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

    years = sorted(close.index.year.unique())

    print("=" * 90)
    print("v1.3 候选版本逐年稳定性验证")
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
    print(
        "测试年份：      "
        + ", ".join(str(year) for year in years)
    )
    print()

    print("候选版本：")
    for candidate in CANDIDATES:
        print(
            f"- {candidate['name']}: "
            f"{candidate['description']}"
        )

    all_yearly_metrics = []
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

        full_metrics = calculate_full_metrics(
            result=result,
            daily_weights=daily_weights,
            monthly_weights=monthly_weights,
            candidate_name=name,
            top_n=top_n,
            momentum_threshold=threshold,
        )

        all_full_metrics.append(full_metrics)

        for year in years:
            yearly_metrics = calculate_year_metrics(
                result=result,
                daily_weights=daily_weights,
                monthly_weights=monthly_weights,
                year=year,
                candidate_name=name,
                top_n=top_n,
                momentum_threshold=threshold,
            )

            all_yearly_metrics.append(
                yearly_metrics
            )

    yearly_results = pd.DataFrame(
        all_yearly_metrics
    )

    full_results = pd.DataFrame(
        all_full_metrics
    )

    yearly_summary = build_yearly_summary(
        yearly_results
    )

    yearly_results.to_csv(
        OUTPUT_DIR / "yearly_stability_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    full_results.to_csv(
        OUTPUT_DIR / "full_sample_candidate_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    yearly_summary.to_csv(
        OUTPUT_DIR / "yearly_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 透视表
    # --------------------------------------------------------

    pivot_year_return = build_year_pivot(
        yearly_results,
        "year_return",
    )

    pivot_year_volatility = build_year_pivot(
        yearly_results,
        "annual_volatility",
    )

    pivot_year_sharpe = build_year_pivot(
        yearly_results,
        "sharpe",
    )

    pivot_year_drawdown = build_year_pivot(
        yearly_results,
        "max_drawdown",
    )

    pivot_year_calmar = build_year_pivot(
        yearly_results,
        "calmar",
    )

    pivot_year_cash = build_year_pivot(
        yearly_results,
        "cash_ratio",
    )

    pivot_year_active_signals = build_year_pivot(
        yearly_results,
        "active_signal_count",
    )

    pivot_year_holding_count = build_year_pivot(
        yearly_results,
        "average_holding_count_when_invested",
    )

    pivot_tables = {
        "pivot_year_return.csv":
            pivot_year_return,
        "pivot_year_volatility.csv":
            pivot_year_volatility,
        "pivot_year_sharpe.csv":
            pivot_year_sharpe,
        "pivot_year_max_drawdown.csv":
            pivot_year_drawdown,
        "pivot_year_calmar.csv":
            pivot_year_calmar,
        "pivot_year_cash_ratio.csv":
            pivot_year_cash,
        "pivot_year_active_signal_count.csv":
            pivot_year_active_signals,
        "pivot_year_holding_count.csv":
            pivot_year_holding_count,
    }

    for filename, table in pivot_tables.items():
        table.to_csv(
            OUTPUT_DIR / filename,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------

    print_full_results(full_results)

    print_table(
        "逐年收益",
        pivot_year_return,
        percent=True,
    )

    print_table(
        "逐年波动",
        pivot_year_volatility,
        percent=True,
    )

    print_table(
        "逐年夏普",
        pivot_year_sharpe,
        percent=False,
        decimals=2,
    )

    print_table(
        "逐年最大回撤",
        pivot_year_drawdown,
        percent=True,
    )

    print_table(
        "逐年Calmar",
        pivot_year_calmar,
        percent=False,
        decimals=2,
    )

    print_table(
        "逐年现金比例",
        pivot_year_cash,
        percent=True,
    )

    print_table(
        "逐年活跃信号月份数",
        pivot_year_active_signals,
        percent=False,
        decimals=0,
    )

    print_table(
        "逐年持仓期间平均ETF数量",
        pivot_year_holding_count,
        percent=False,
        decimals=2,
    )

    print_yearly_summary(
        yearly_summary
    )

    print_yearly_ranking(
        yearly_results
    )

    print()
    print(
        f"逐年稳定性测试结果已保存到："
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()