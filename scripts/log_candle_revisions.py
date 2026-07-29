"""Compare working-copy candle files against the last committed (HEAD) versions
and append revision events to public/debug/revision_telemetry.jsonl.
Runs in the notify job after the public refresh, before the commit step.
Measurement-only: always exits 0."""
import json
import os
import subprocess
from datetime import datetime, timezone

ASSETS = ("GOLD_CFD", "XAGUSD", "USOIL")
SERIES = ("klines_1m", "klines_5m")


def _closes(payload):
    out = {}
    for v in (payload or {}).get("values") or []:
        try:
            out[str(v.get("datetime"))] = float(v.get("close"))
        except (TypeError, ValueError):
            continue
    return out


def main() -> int:
    events = 0
    for asset in ASSETS:
        for series in SERIES:
            rel = f"public/{asset}/{series}.json"
            try:
                with open(rel, encoding="utf-8") as handle:
                    new = json.load(handle)
            except Exception:
                continue
            proc = subprocess.run(["git", "show", f"HEAD:{rel}"], capture_output=True, text=True)
            if proc.returncode != 0:
                continue
            try:
                old = json.loads(proc.stdout)
            except Exception:
                continue
            o, n = _closes(old), _closes(new)
            if not o:
                continue
            newest_old = max(o)
            deltas = []
            for ts, oc in o.items():
                if ts == newest_old:
                    continue
                nc = n.get(ts)
                if nc is not None and abs(nc - oc) > 1e-9:
                    deltas.append((abs(nc - oc), ts, oc, nc))
            if not deltas:
                continue
            deltas.sort(reverse=True)
            top = deltas[0]
            row = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "asset": asset,
                "series": series,
                "overlap": sum(1 for ts in o if ts != newest_old and ts in n),
                "revised_count": len(deltas),
                "max_abs_delta": round(top[0], 6),
                "max_delta_ts": top[1],
                "old_close": round(top[2], 6),
                "new_close": round(top[3], 6),
            }
            os.makedirs("public/debug", exist_ok=True)
            with open("public/debug/revision_telemetry.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            events += 1
    print(f"revision events logged: {events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
