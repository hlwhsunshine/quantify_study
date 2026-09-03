from pathlib import Path

import numpy as np
import pandas as pd

from multi_etf_backtest_v13_dynamic import (
    DATA_FILE,
    OUTPUT_DIR as BACKTEST_OUTPUT_DIR,
    load_close_data,
    run_backtest,
)


# ============================================================
# 参数
# ============================================================

TOP1 = 1
TOP3 = 3

ANALYSIS_OUTPUT_DIR = Path(
    "results/v1_3_top3_analysis"
)


# ============================================================
# 年度绩效
# ============================================================

def calculate_yearly_metrics(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    按自然年计算：

    - 年度收益
    - 年化波动
    - 年内最大回撤
    - 平均投入仓位
    - 持仓天数比例
    - 平均持仓数量
    - 交易成本
    """

    records = []

    years = sorted(result.index.year.unique())

    for year in years:
        year_mask = result.index.year == year

        year_result = result.loc[year_mask]
        year_weights = daily_weights.loc[year_mask]

        if year_result.empty:
            continue

        annual_return = (
            1.0 + year_result["net_return"]
        ).prod() - 1.0

        annual_volatility = (
            year_result["net_return"].std(ddof=1)
            * np.sqrt(252)
        )

        year_net_value = (
            1.0 + year_result["net_return"]
        ).cumprod()

        year_drawdown = (
            year_net_value
            / year_net_value.cummax()
            - 1.0
        )

        max_drawdown = year_drawdown.min()

        average_invested_weight = (
            year_result["invested_weight"].mean()
        )

        invested_day_ratio = (
            year_result["invested_weight"] > 1e-12
        ).mean()

        holding_count = (
            year_weights > 1e-12
        ).sum(axis=1)

        average_holding_count = holding_count.mean()

        invested_days = holding_count > 0

        if invested_days.any():
            average_count_when_invested = (
                holding_count[invested_days].mean()
            )
        else:
            average_count_when_invested = 0.0

        total_cost = year_result["cost"].sum()

        records.append({
            "year": year,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "max_drawdown": max_drawdown,
            "average_invested_weight":
                average_invested_weight,
            "invested_day_ratio":
                invested_day_ratio,
            "average_holding_count":
                average_holding_count,
            "average_count_when_invested":
                average_count_when_invested,
            "total_cost": total_cost,
        })

    return pd.DataFrame(records)


# ============================================================
# 最大回撤详情
# ============================================================

def calculate_drawdown_details(
    result: pd.DataFrame,
) -> dict:
    """
    返回整个样本最大回撤详情。
    """

    net_value = result["net_value"]

    rolling_peak = net_value.cummax()

    drawdown = (
        net_value
        / rolling_peak
        - 1.0
    )

    trough_date = drawdown.idxmin()

    peak_date = (
        net_value.loc[:trough_date]
        .idxmax()
    )

    peak_value = net_value.loc[peak_date]
    trough_value = net_value.loc[trough_date]

    max_drawdown = drawdown.loc[trough_date]

    recovery_candidates = net_value.loc[
        trough_date:
    ]

    recovery_candidates = recovery_candidates[
        recovery_candidates >= peak_value
    ]

    if recovery_candidates.empty:
        recovery_date = pd.NaT
        recovery_days = np.nan
    else:
        recovery_date = recovery_candidates.index[0]

        recovery_days = (
            recovery_date - trough_date
        ).days

    peak_to_trough_days = (
        trough_date - peak_date
    ).days

    return {
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "peak_value": peak_value,
        "trough_value": trough_value,
        "max_drawdown": max_drawdown,
        "peak_to_trough_days":
            peak_to_trough_days,
        "recovery_days": recovery_days,
    }


# ============================================================
# ETF收益贡献
# ============================================================

def calculate_etf_contribution(
    close: pd.DataFrame,
    daily_weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算每只ETF每天对组合毛收益的贡献。

    每日贡献：
        当日权重 × 当日ETF收益

    注意：
    - 贡献为毛收益贡献；
    - 交易成本单独统计；
    - 各ETF累计贡献是日贡献之和，
      不是复合收益率。
    """

    asset_returns = (
        close
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    daily_contribution = (
        daily_weights * asset_returns
    )

    total_contribution = (
        daily_contribution.sum()
        .sort_values(ascending=False)
        .rename("gross_return_contribution")
        .to_frame()
    )

    total_contribution[
        "holding_days"
    ] = (
        daily_weights > 1e-12
    ).sum()

    total_contribution[
        "average_weight_all_days"
    ] = daily_weights.mean()

    active_weight = daily_weights.where(
        daily_weights > 1e-12
    )

    total_contribution[
        "average_weight_when_held"
    ] = active_weight.mean()

    total_contribution[
        "positive_contribution_days"
    ] = (
        daily_contribution > 0
    ).sum()

    total_contribution[
        "negative_contribution_days"
    ] = (
        daily_contribution < 0
    ).sum()

    return (
        daily_contribution,
        total_contribution,
    )


# ============================================================
# 年度ETF收益贡献
# ============================================================

def calculate_yearly_etf_contribution(
    daily_contribution: pd.DataFrame,
) -> pd.DataFrame:
    """
    返回年度ETF毛收益贡献宽表。
    """

    yearly = daily_contribution.groupby(
        daily_contribution.index.year
    ).sum()

    yearly.index.name = "year"

    return yearly


# ============================================================
# 持仓周期明细
# ============================================================

def build_holding_periods(
    result: pd.DataFrame,
    daily_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    将每日持仓整理成连续持仓周期。

    每次权重发生变化，视为新的持仓周期。
    """

    weight_change = (
        daily_weights
        .diff()
        .abs()
        .sum(axis=1)
    )

    if len(weight_change) > 0:
        weight_change.iloc[0] = (
            daily_weights.iloc[0]
            .abs()
            .sum()
        )

    change_dates = daily_weights.index[
        weight_change > 1e-12
    ].tolist()

    if not change_dates:
        return pd.DataFrame()

    records = []

    for i, start_date in enumerate(change_dates):
        if i + 1 < len(change_dates):
            next_change_date = change_dates[i + 1]

            next_position = result.index.get_loc(
                next_change_date
            )

            end_position = next_position - 1
            end_date = result.index[end_position]
        else:
            end_date = result.index[-1]

        weights = daily_weights.loc[start_date]

        active_weights = weights[
            weights > 1e-12
        ]

        holdings = (
            ",".join(active_weights.index)
            if not active_weights.empty
            else "cash"
        )

        weight_text = (
            ",".join(
                f"{etf}:{weight:.2%}"
                for etf, weight
                in active_weights.items()
            )
            if not active_weights.empty
            else ""
        )

        period_result = result.loc[
            start_date:end_date
        ]

        period_return = (
            1.0 + period_result["net_return"]
        ).prod() - 1.0

        period_gross_return = (
            1.0 + period_result["gross_return"]
        ).prod() - 1.0

        period_cost = period_result["cost"].sum()

        period_net_value = (
            1.0 + period_result["net_return"]
        ).cumprod()

        period_drawdown = (
            period_net_value
            / period_net_value.cummax()
            - 1.0
        )

        max_drawdown = period_drawdown.min()

        records.append({
            "start_date": start_date,
            "end_date": end_date,
            "calendar_days":
                (end_date - start_date).days + 1,
            "trading_days": len(period_result),
            "holding_count":
                len(active_weights),
            "holdings": holdings,
            "weights": weight_text,
            "gross_return":
                period_gross_return,
            "net_return":
                period_return,
            "max_drawdown":
                max_drawdown,
            "cost":
                period_cost,
        })

    periods = pd.DataFrame(records)

    return periods


# ============================================================
# 月度收益
# ============================================================

def calculate_monthly_returns(
    result: pd.DataFrame,
    name: str,
) -> pd.Series:
    """
    计算每月复合净收益。
    """

    monthly_return = (
        result["net_return"]
        .groupby(
            result.index.to_period("M")
        )
        .apply(
            lambda x: (
                1.0 + x
            ).prod() - 1.0
        )
    )

    monthly_return.index = (
        monthly_return.index.astype(str)
    )

    monthly_return.name = name

    return monthly_return


# ============================================================
# Top1和Top3对比
# ============================================================

def build_top1_top3_comparison(
    top1_result: pd.DataFrame,
    top3_result: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成月度和年度Top1/Top3对比。
    """

    top1_monthly = calculate_monthly_returns(
        top1_result,
        "top1_return",
    )

    top3_monthly = calculate_monthly_returns(
        top3_result,
        "top3_return",
    )

    monthly_comparison = pd.concat(
        [top1_monthly, top3_monthly],
        axis=1,
    ).fillna(0.0)

    monthly_comparison[
        "top3_minus_top1"
    ] = (
        monthly_comparison["top3_return"]
        - monthly_comparison["top1_return"]
    )

    monthly_comparison.index.name = "month"

    top1_yearly = (
        top1_result["net_return"]
        .groupby(top1_result.index.year)
        .apply(
            lambda x: (
                1.0 + x
            ).prod() - 1.0
        )
        .rename("top1_return")
    )

    top3_yearly = (
        top3_result["net_return"]
        .groupby(top3_result.index.year)
        .apply(
            lambda x: (
                1.0 + x
            ).prod() - 1.0
        )
        .rename("top3_return")
    )

    yearly_comparison = pd.concat(
        [top1_yearly, top3_yearly],
        axis=1,
    ).fillna(0.0)

    yearly_comparison[
        "top3_minus_top1"
    ] = (
        yearly_comparison["top3_return"]
        - yearly_comparison["top1_return"]
    )

    yearly_comparison.index.name = "year"

    return (
        monthly_comparison,
        yearly_comparison,
    )


# ============================================================
# 持仓数量分布
# ============================================================

def calculate_holding_count_distribution(
    daily_weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    统计0只、1只、2只、3只ETF分别占多少天。
    """

    holding_count = (
        daily_weights > 1e-12
    ).sum(axis=1)

    distribution = (
        holding_count.value_counts()
        .sort_index()
        .rename("trading_days")
        .to_frame()
    )

    distribution[
        "day_ratio"
    ] = (
        distribution["trading_days"]
        / len(holding_count)
    )

    distribution.index.name = "holding_count"

    return distribution


# ============================================================
# 打印函数
# ============================================================

def print_yearly_comparison(
    yearly_comparison: pd.DataFrame,
) -> None:
    print("=" * 75)
    print("Top1 与 Top3 年度收益对比")
    print("=" * 75)

    print(
        yearly_comparison.to_string(
            formatters={
                "top1_return":
                    "{:.2%}".format,
                "top3_return":
                    "{:.2%}".format,
                "top3_minus_top1":
                    "{:+.2%}".format,
            }
        )
    )


def print_drawdown_details(
    details: dict,
) -> None:
    print()
    print("=" * 75)
    print("Top3 最大回撤详情")
    print("=" * 75)

    print(
        f"回撤起点：       "
        f"{details['peak_date'].date()}"
    )

    print(
        f"回撤低点：       "
        f"{details['trough_date'].date()}"
    )

    if pd.isna(details["recovery_date"]):
        recovery_text = "截至样本结束尚未恢复"
    else:
        recovery_text = (
            details["recovery_date"].date()
        )

    print(
        f"恢复日期：       "
        f"{recovery_text}"
    )

    print(
        f"高点净值：       "
        f"{details['peak_value']:.4f}"
    )

    print(
        f"低点净值：       "
        f"{details['trough_value']:.4f}"
    )

    print(
        f"最大回撤：       "
        f"{details['max_drawdown']:.2%}"
    )

    print(
        f"高点到低点天数： "
        f"{details['peak_to_trough_days']}"
    )

    if pd.notna(details["recovery_days"]):
        print(
            f"低点到恢复天数： "
            f"{int(details['recovery_days'])}"
        )


def print_top_contributions(
    contribution: pd.DataFrame,
) -> None:
    print()
    print("=" * 90)
    print("Top3 各ETF毛收益贡献")
    print("=" * 90)

    display = contribution.copy()

    print(
        display.to_string(
            formatters={
                "gross_return_contribution":
                    "{:+.2%}".format,
                "average_weight_all_days":
                    "{:.2%}".format,
                "average_weight_when_held":
                    "{:.2%}".format,
            }
        )
    )


def print_best_worst_periods(
    holding_periods: pd.DataFrame,
) -> None:
    active_periods = holding_periods[
        holding_periods["holding_count"] > 0
    ]

    if active_periods.empty:
        return

    best = (
        active_periods
        .sort_values(
            "net_return",
            ascending=False,
        )
        .head(10)
    )

    worst = (
        active_periods
        .sort_values(
            "net_return",
            ascending=True,
        )
        .head(10)
    )

    columns = [
        "start_date",
        "end_date",
        "holding_count",
        "holdings",
        "weights",
        "net_return",
        "max_drawdown",
        "cost",
    ]

    print()
    print("=" * 110)
    print("Top3 收益最好的10个持仓周期")
    print("=" * 110)

    print(
        best[columns].to_string(
            index=False,
            formatters={
                "net_return":
                    "{:+.2%}".format,
                "max_drawdown":
                    "{:.2%}".format,
                "cost":
                    "{:.2%}".format,
            }
        )
    )

    print()
    print("=" * 110)
    print("Top3 收益最差的10个持仓周期")
    print("=" * 110)

    print(
        worst[columns].to_string(
            index=False,
            formatters={
                "net_return":
                    "{:+.2%}".format,
                "max_drawdown":
                    "{:.2%}".format,
                "cost":
                    "{:.2%}".format,
            }
        )
    )


# ============================================================
# 主程序
# ============================================================

def main() -> None:
    ANALYSIS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    close = load_close_data(DATA_FILE)

    (
        top1_result,
        top1_weights,
        top1_monthly_weights,
    ) = run_backtest(
        close=close,
        top_n=TOP1,
    )

    (
        top3_result,
        top3_weights,
        top3_monthly_weights,
    ) = run_backtest(
        close=close,
        top_n=TOP3,
    )

    # --------------------------------------------------------
    # 年度指标
    # --------------------------------------------------------

    top1_yearly_metrics = (
        calculate_yearly_metrics(
            top1_result,
            top1_weights,
        )
    )

    top3_yearly_metrics = (
        calculate_yearly_metrics(
            top3_result,
            top3_weights,
        )
    )

    top1_yearly_metrics.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top1_yearly_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top3_yearly_metrics.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_yearly_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 最大回撤
    # --------------------------------------------------------

    drawdown_details = (
        calculate_drawdown_details(
            top3_result
        )
    )

    pd.DataFrame(
        [drawdown_details]
    ).to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_max_drawdown_details.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 保存最大回撤区间每日数据
    drawdown_start = drawdown_details[
        "peak_date"
    ]

    if pd.isna(
        drawdown_details["recovery_date"]
    ):
        drawdown_end = top3_result.index[-1]
    else:
        drawdown_end = drawdown_details[
            "recovery_date"
        ]

    drawdown_period = top3_result.loc[
        drawdown_start:drawdown_end
    ].copy()

    drawdown_period[
        "drawdown"
    ] = (
        drawdown_period["net_value"]
        / drawdown_period[
            "net_value"
        ].cummax()
        - 1.0
    )

    drawdown_period.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_max_drawdown_daily.csv",
        encoding="utf-8-sig",
    )

    top3_weights.loc[
        drawdown_start:drawdown_end
    ].to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_max_drawdown_weights.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # ETF收益贡献
    # --------------------------------------------------------

    (
        daily_contribution,
        total_contribution,
    ) = calculate_etf_contribution(
        close,
        top3_weights,
    )

    yearly_contribution = (
        calculate_yearly_etf_contribution(
            daily_contribution
        )
    )

    daily_contribution.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_daily_etf_contribution.csv",
        encoding="utf-8-sig",
    )

    total_contribution.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_total_etf_contribution.csv",
        encoding="utf-8-sig",
    )

    yearly_contribution.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_yearly_etf_contribution.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 持仓周期
    # --------------------------------------------------------

    holding_periods = build_holding_periods(
        top3_result,
        top3_weights,
    )

    holding_periods.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_holding_periods.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Top1 vs Top3
    # --------------------------------------------------------

    (
        monthly_comparison,
        yearly_comparison,
    ) = build_top1_top3_comparison(
        top1_result,
        top3_result,
    )

    monthly_comparison.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top1_top3_monthly_comparison.csv",
        encoding="utf-8-sig",
    )

    yearly_comparison.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top1_top3_yearly_comparison.csv",
        encoding="utf-8-sig",
    )

    # Top3改善最大的月份
    best_improvement_months = (
        monthly_comparison
        .sort_values(
            "top3_minus_top1",
            ascending=False,
        )
        .head(15)
    )

    worst_improvement_months = (
        monthly_comparison
        .sort_values(
            "top3_minus_top1",
            ascending=True,
        )
        .head(15)
    )

    best_improvement_months.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_best_improvement_months.csv",
        encoding="utf-8-sig",
    )

    worst_improvement_months.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_worst_improvement_months.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 持仓数量分布
    # --------------------------------------------------------

    holding_distribution = (
        calculate_holding_count_distribution(
            top3_weights
        )
    )

    holding_distribution.to_csv(
        ANALYSIS_OUTPUT_DIR
        / "top3_holding_count_distribution.csv",
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 打印结果
    # --------------------------------------------------------

    print("数据文件：", DATA_FILE)
    print(
        "原回测结果目录：",
        BACKTEST_OUTPUT_DIR,
    )
    print(
        "分析结果目录：",
        ANALYSIS_OUTPUT_DIR,
    )
    print()

    print_yearly_comparison(
        yearly_comparison
    )

    print_drawdown_details(
        drawdown_details
    )

    print_top_contributions(
        total_contribution
    )

    print_best_worst_periods(
        holding_periods
    )

    print()
    print("=" * 75)
    print("Top3持仓数量分布")
    print("=" * 75)

    print(
        holding_distribution.to_string(
            formatters={
                "day_ratio":
                    "{:.2%}".format,
            }
        )
    )

    print()
    print("=" * 75)
    print("Top3相对Top1改善最大的10个月")
    print("=" * 75)

    print(
        best_improvement_months
        .head(10)
        .to_string(
            formatters={
                "top1_return":
                    "{:+.2%}".format,
                "top3_return":
                    "{:+.2%}".format,
                "top3_minus_top1":
                    "{:+.2%}".format,
            }
        )
    )

    print()
    print("=" * 75)
    print("Top3相对Top1表现落后的10个月")
    print("=" * 75)

    print(
        worst_improvement_months
        .head(10)
        .to_string(
            formatters={
                "top1_return":
                    "{:+.2%}".format,
                "top3_return":
                    "{:+.2%}".format,
                "top3_minus_top1":
                    "{:+.2%}".format,
            }
        )
    )

    print()
    print(
        f"分析结果已保存到："
        f"{ANALYSIS_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()