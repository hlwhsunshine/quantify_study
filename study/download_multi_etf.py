#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二阶段：批量获取多个 ETF 的历史行情数据

当前目标：
1. 使用 ak.fund_etf_hist_sina 获取多个 ETF 日线数据
2. 统一字段格式
3. 计算 daily_return、ma20、ma60、momentum_60
4. 每只 ETF 单独保存为 CSV
"""

import os
import time

import akshare as ak
import pandas as pd


# =========================
# 配置区
# =========================

START_DATE = "2020-01-01"
END_DATE = "2026-05-31"

DATA_DIR = "data"

# ETF 标的池
# 注意：新浪 ETF 接口需要 sh / sz 前缀
ETF_LIST = {
    "hs300": "sh510300",      # 沪深300ETF
    "zz500": "sh510500",      # 中证500ETF
    "cyb": "sz159915",        # 创业板ETF
    "kc50": "sh588000",       # 科创50ETF
    "securities": "sh512880", # 证券ETF
    "consumer": "sz159928",   # 消费ETF
    "medicine": "sh512010",   # 医药ETF
    "new_energy": "sh516160", # 新能源ETF
    "dividend": "sh510880",   # 红利ETF
}


# =========================
# 获取单只 ETF 数据
# =========================

def get_etf_data(symbol):
    """
    获取单只 ETF 历史行情。

    这里使用你本地能跑通的接口：
    ak.fund_etf_hist_sina(symbol=SYMBOL)

    返回字段通常包括：
    date, open, high, low, close, volume, amount
    """
    df = ak.fund_etf_hist_sina(symbol=symbol)

    return df


# =========================
# 清洗数据
# =========================

def clean_data(df):
    """
    清洗单只 ETF 数据。

    保留必要字段：
    - date: 日期
    - open: 开盘价
    - high: 最高价
    - low: 最低价
    - close: 收盘价
    - volume: 成交量
    - amount: 成交额
    """

    if df is None or df.empty:
        raise ValueError("没有获取到数据")

    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"数据缺少必要字段：{missing_columns}")

    df = df[required_columns].copy()

    # 转换日期格式
    df["date"] = pd.to_datetime(df["date"])

    # 过滤日期范围
    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)

    df = df[
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ]

    # 转换数字字段
    number_columns = ["open", "high", "low", "close", "volume", "amount"]

    for col in number_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除关键价格缺失的数据
    df = df.dropna(subset=["open", "high", "low", "close"])

    # 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    return df


# =========================
# 计算指标
# =========================

def add_indicators(df):
    """
    添加基础指标。

    daily_return:
        每日涨跌幅

    ma20:
        20 日均线

    ma60:
        60 日均线

    momentum_60:
        60 日动量，后面月度轮动会用
    """

    df = df.copy()

    df["daily_return"] = df["close"].pct_change()

    df["ma20"] = df["close"].rolling(window=20).mean()

    df["ma60"] = df["close"].rolling(window=60).mean()

    df["momentum_60"] = df["close"] / df["close"].shift(60) - 1

    return df


# =========================
# 保存 CSV
# =========================

def save_csv(df, name, symbol):
    """
    保存单只 ETF 数据到 data 目录。
    """

    os.makedirs(DATA_DIR, exist_ok=True)

    file_path = os.path.join(DATA_DIR, f"{name}_{symbol}.csv")

    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    return file_path


# =========================
# 主程序
# =========================

def main():
    print("=" * 60)
    print("开始批量下载 ETF 数据")
    print("=" * 60)

    success_count = 0
    failed_list = []

    for name, symbol in ETF_LIST.items():
        print(f"\n正在处理：{name} - {symbol}")

        try:
            # 1. 获取数据
            df = get_etf_data(symbol)

            # 2. 清洗数据
            df = clean_data(df)

            # 3. 计算指标
            df = add_indicators(df)

            # 4. 保存 CSV
            file_path = save_csv(df, name, symbol)

            print(f"成功：{name} - {symbol}")
            print(f"日期：{df['date'].min().date()} 到 {df['date'].max().date()}")
            print(f"行数：{len(df)}")
            print(f"保存：{file_path}")

            success_count += 1

            # 稍微暂停一下，避免请求太快
            time.sleep(0.5)

        except Exception as exc:
            print(f"失败：{name} - {symbol}")
            print(f"原因：{exc}")
            failed_list.append((name, symbol, str(exc)))

    print("\n" + "=" * 60)
    print("批量下载完成")
    print("=" * 60)
    print(f"成功数量：{success_count}")
    print(f"失败数量：{len(failed_list)}")

    if failed_list:
        print("\n失败列表：")
        for name, symbol, reason in failed_list:
            print(f"{name} - {symbol} - {reason}")


if __name__ == "__main__":
    main()