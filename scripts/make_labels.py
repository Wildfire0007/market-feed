import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def realized_pnl_label(df, entry_col, exit_col, side_col=None, qty_col=None, fees_cols=()):
    entry = pd.to_numeric(df[entry_col], errors="coerce")
    exit_ = pd.to_numeric(df[exit_col], errors="coerce")

    qty = pd.to_numeric(df[qty_col], errors="coerce") if qty_col and qty_col in df else 1.0
    if isinstance(qty, (int, float)):
        qty = pd.Series(qty, index=df.index)

    if side_col and side_col in df:
        side = df[side_col].astype(str).str.lower().map(
            lambda s: -1 if s in {"short", "sell", "-1"} else 1
        )
    else:
        side = 1

    gross = np.where((side == 1) | (side == True), (exit_ - entry) * qty, (entry - exit_) * qty)

    fees = sum(
        pd.to_numeric(df[c], errors="coerce").fillna(0)
        for c in fees_cols
        if c in df
    )
    pnl = pd.Series(gross, index=df.index) - (fees if isinstance(fees, pd.Series) else 0.0)

    return (pnl > 0).astype(int)


def fixed_horizon_label(price, horizon=12, threshold=0.0):
    ret_h = price.shift(-horizon) / price - 1.0
    lab = (ret_h > threshold).astype(int)
    if horizon > 0:
        lab.iloc[-horizon:] = 0
    return lab


def triple_barrier_labels(price, events_idx, pt=0.01, sl=0.01, max_h=48):
    labels = pd.Series(0, index=events_idx, dtype=int)

    for i in events_idx:
        if i >= len(price):
            continue

        entry = price.iloc[i]
        up = entry * (1 + pt)
        dn = entry * (1 - sl)
        end = min(i + max_h, len(price) - 1)

        if i + 1 > end:
            ret = price.iloc[end] / entry - 1.0 if entry != 0 else 0.0
            labels.loc[i] = 1 if ret > 0 else 0
            continue

        win = price.iloc[i + 1 : end + 1]
        hit_up = win[win >= up]
        hit_dn = win[win <= dn]

        if len(hit_up) and (not len(hit_dn) or hit_up.index[0] < hit_dn.index[0]):
            labels.loc[i] = 1
        elif len(hit_dn) and (not len(hit_up) or hit_dn.index[0] < hit_up.index[0]):
            labels.loc[i] = 0
        else:
            ret = price.iloc[end] / entry - 1.0 if entry != 0 else 0.0
            labels.loc[i] = 1 if ret > 0 else 0

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--ohlc")
    parser.add_argument("--timestamp", default="timestamp")
    parser.add_argument("--price_col", default="close")
    parser.add_argument("--method", choices=["realized", "fixed", "tbm"], required=True)

    parser.add_argument("--entry_col")
    parser.add_argument("--exit_col")
    parser.add_argument("--side_col")
    parser.add_argument("--qty_col")
    parser.add_argument("--fees_cols", nargs="*", default=[])

    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--pt", type=float, default=0.01)
    parser.add_argument("--sl", type=float, default=0.01)
    parser.add_argument("--trigger_col", default="precision_trigger_fire")

    args = parser.parse_args()

    feat = pd.read_csv(args.features)

    if args.method in {"fixed", "tbm"}:
        if not args.ohlc:
            raise ValueError("fixed/tbm methods require --ohlc path")

        px = pd.read_csv(args.ohlc)

        for data in (feat, px):
            if args.timestamp in data and not pd.api.types.is_datetime64_any_dtype(data[args.timestamp]):
                data[args.timestamp] = pd.to_datetime(data[args.timestamp])

        df = pd.merge_asof(
            feat.sort_values(args.timestamp),
            px[[args.timestamp, args.price_col]].sort_values(args.timestamp),
            on=args.timestamp,
            direction="nearest",
        )
        price = pd.to_numeric(df[args.price_col], errors="coerce").fillna(method="ffill")
    else:
        df = feat.copy()
        price = None

    if args.method == "realized":
        if not args.entry_col or not args.exit_col:
            raise ValueError("realized method requires --entry_col and --exit_col")
        lab = realized_pnl_label(
            df,
            args.entry_col,
            args.exit_col,
            side_col=args.side_col,
            qty_col=args.qty_col,
            fees_cols=args.fees_cols,
        )
    elif args.method == "fixed":
        if price is None:
            raise ValueError("price series missing for fixed method")
        lab = fixed_horizon_label(price, horizon=args.horizon, threshold=args.threshold)
    else:
        if price is None:
            raise ValueError("price series missing for tbm method")
        if args.trigger_col in df:
            events = df.index[df[args.trigger_col].fillna(0).astype(int) == 1]
        else:
            events = df.index
        lab = triple_barrier_labels(price, events, pt=args.pt, sl=args.sl, max_h=args.horizon)
        lab = lab.reindex(df.index).fillna(0).astype(int)

    df["label"] = lab.values

    out_path = Path(args.features)
    out_file = out_path.with_name(f"{out_path.stem}_labeled.csv")
    df.to_csv(out_file, index=False)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()
