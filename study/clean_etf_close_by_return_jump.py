#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据异常单日涨跌幅，自动修正 ETF 收盘价序列断点。

输入：
data/etf_close.csv

输出：
data/etf_close_clean.csv
data/etf_price_adjustment_log.csv

逻辑：
1. 计算每只 ETF 的日收益率
2. 如果某天涨跌幅绝对值超过阈值，认为可能发生价格口径断点
3. 用 前一日价格 / 当日价格 作为调整因子
4. 将该 ETF 从当日开始到最后的价格整体乘以调整因子
5. 这样可以消除异常跳变，使价格序列连续

注意：
这不是官方复权数据，只是为了学习回测做的价格连续化处理。
正式实盘研究时，最好换成可靠的复权数据源。
"""

import os
import pandas as pd


DATA_DIR = "data"

INPUT_FILE = os.path.join(DATA_DIR, "etf_close.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "etf_close_clean.csv")
LOG_FILE = os.path.join(DATA_DIR, "etf_price_adjustment_log.csv")

# 超过 50% 的单日跳变才自动修
# 这样不会误修 2024-09-30 创业板、科创50 那种真实大波动
JUMP_THRESHOLD = 0.50


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    clean_df = df.copy()

    etf_cols = [col for col in df.columns if col != "date"]

    logs = []

    print("===== ETF 收盘价异常断点修正 =====")
    print(f"输入文件：{INPUT_FILE}")
    print(f"输出文件：{OUTPUT_FILE}")
    print(f"跳变阈值：±{JUMP_THRESHOLD:.0%}")
    print()

    for col in etf_cols:
        print(f"检查：{col}")

        # 注意：这里每修正一次后，都基于修正后的价格重新计算收益率
        while True:
            price = clean_df[col]
            daily_return = price.pct_change(fill_method=None)

            abnormal_mask = (
                (daily_return > JUMP_THRESHOLD) |
                (daily_return < -JUMP_THRESHOLD)
            )

            abnormal_indices = clean_df.index[abnormal_mask].tolist()

            if len(abnormal_indices) == 0:
                break

            # 每次只处理最早的一个异常点
            idx = abnormal_indices[0]

            prev_idx = idx - 1

            if prev_idx < 0:
                break

            date = clean_df.loc[idx, "date"]
            prev_date = clean_df.loc[prev_idx, "date"]

            prev_price = clean_df.loc[prev_idx, col]
            current_price = clean_df.loc[idx, col]
            original_return = daily_return.loc[idx]

            if pd.isna(prev_price) or pd.isna(current_price) or current_price == 0:
                print(f"  跳过异常点：{date.date()}，价格为空或为0")
                break

            # 调整因子：让当日价格接近前一日价格
            adjust_factor = prev_price / current_price

            # 从异常日开始，后面所有价格整体乘以调整因子
            clean_df.loc[idx:, col] = clean_df.loc[idx:, col] * adjust_factor

            new_price = clean_df.loc[idx, col]
            new_return = clean_df[col].pct_change(fill_method=None).loc[idx]

            logs.append({
                "etf": col,
                "date": date.date(),
                "prev_date": prev_date.date(),
                "prev_price": prev_price,
                "current_price_before": current_price,
                "original_return": original_return,
                "adjust_factor": adjust_factor,
                "current_price_after": new_price,
                "new_return": new_return,
            })

            print(
                f"  修正断点：{date.date()} | "
                f"前价：{prev_price:.6f} | "
                f"原当日价：{current_price:.6f} | "
                f"原涨跌幅：{original_return:.2%} | "
                f"调整因子：{adjust_factor:.6f} | "
                f"修正后涨跌幅：{new_return:.2%}"
            )

    clean_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

    print()
    print("===== 修正完成 =====")
    print(f"修正记录数量：{len(log_df)}")
    print(f"清洗后数据已保存到：{OUTPUT_FILE}")
    print(f"修正日志已保存到：{LOG_FILE}")


if __name__ == "__main__":
    main()