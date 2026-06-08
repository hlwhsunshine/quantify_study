#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二阶段：获取 ETF 日线数据，并计算基础指标

当前目标：
1. 获取沪深300ETF 510300 的日线行情
2. 保留必要行情字段
3. 计算每日涨跌幅、MA20、MA60
4. 保存为 CSV，后面做回测使用
"""

import akshare as ak
import pandas as pd


# =========================
# 配置区
# =========================

# 沪深300ETF，代码 510300，上海市场，所以前缀是 sh
SYMBOL = "sh510300"

START_DATE = "2020-01-01"
END_DATE = "2026-05-31"

OUTPUT_CSV = "data/10300_etf_daily.csv"


# =========================
# 获取数据
# =========================

def get_data():
    """
    使用 ak.fund_etf_hist_sina 获取 ETF 日线数据。

    注意：
    - 这个接口返回的是新浪数据
    - 当前学习阶段可以使用
    - 后面严肃回测时，我们再考虑前复权数据源
    """
    df = ak.fund_etf_hist_sina(symbol=SYMBOL)

    return df


# =========================
# 清洗数据
# =========================

def clean_data(df):
    """
    清洗数据，统一格式。

    必要字段说明：
    - date: 日期
    - open: 开盘价
    - high: 最高价
    - low: 最低价
    - close: 收盘价，后面计算收益率、均线、动量主要靠它
    - volume: 成交量
    - amount: 成交额
    - outstanding_share: 流通股本/流通份额，接口返回则保留
    - turnover: 换手率，接口返回则保留
    """

    if df is None or df.empty:
        raise ValueError("没有获取到数据，请检查代码或日期范围")

    # 这些是后续策略必须要用的核心字段
    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]

    # 这些字段不是必须，但如果接口返回了，就保留
    optional_columns = [
        "outstanding_share",
        "turnover",
    ]

    # 检查必要字段是否缺失
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"数据缺少必要字段：{missing_columns}")

    # 最终保留字段 = 必要字段 + 实际存在的可选字段
    keep_columns = required_columns + [
        col for col in optional_columns
        if col in df.columns
    ]

    df = df[keep_columns].copy()

    # 日期转成 datetime，方便后面按月处理
    df["date"] = pd.to_datetime(df["date"])

    # 新浪 ETF 接口会返回该 ETF 全部历史数据，这里按配置的日期区间过滤
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    # 数字字段转成数值类型，避免后面计算出错
    number_columns = [
        col for col in keep_columns
        if col != "date"
    ]

    for col in number_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除关键价格字段为空的数据
    df = df.dropna(subset=["open", "high", "low", "close"])

    # 按日期从早到晚排序
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 计算指标
# =========================

def add_indicators(df):
    """
    添加基础指标。

    daily_return:
        每日涨跌幅 = 今天收盘价 / 昨天收盘价 - 1

    ma20:
        20 日均线

    ma60:
        60 日均线，后面月度动量策略也会用 60 日收益
    """

    df = df.copy()

    # 每日涨跌幅
    df["daily_return"] = df["close"].pct_change()

    # 20 日均线
    df["ma20"] = df["close"].rolling(window=20).mean()

    # 60 日均线
    df["ma60"] = df["close"].rolling(window=60).mean()

    # 60 日动量：当前收盘价相比 60 个交易日前涨了多少
    # 后面 ETF 月度动量轮动策略会重点用这个字段
    df["momentum_60"] = df["close"] / df["close"].shift(60) - 1

    return df


# =========================
# 主程序
# =========================

def main():
    # 1. 获取原始数据
    df = get_data()

    # 2. 清洗数据
    df = clean_data(df)

    # 3. 计算指标
    df = add_indicators(df)

    # 4. 打印检查结果
    print("=" * 60)
    print(f"代码：{SYMBOL}")
    print(f"数据日期：{df['date'].min().date()} 到 {df['date'].max().date()}")
    print(f"数据行数：{len(df)}")
    print("=" * 60)

    print("\n前 5 行：")
    print(df.head())

    print("\n后 5 行：")
    print(df.tail())

    print("\n字段：")
    print(df.columns)

    # 5. 保存为 CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n已保存到：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
