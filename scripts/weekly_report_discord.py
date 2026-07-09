#!/usr/bin/env python3
"""Heti ACTIONABLE mérési riport Discordra."""
from __future__ import annotations
import csv, json, os, sys, logging
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
try: import requests
except Exception: requests=None
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config import analysis_settings as settings
from scripts.label_trades import _wilson_interval
from scripts.trade_ledger import rows_between, stats
from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
LOGGER=logging.getLogger(__name__)
PUBLIC=Path(os.getenv("NOTIFY_PUBLIC_DIR","public")); STATE=PUBLIC/"monitoring"/"weekly_report_state.json"
DAYS={"MON":0,"TUE":1,"WED":2,"THU":3,"FRI":4,"SAT":5,"SUN":6}
def load(p):
    try: return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    except Exception: return {}
def urls():
    raw=os.getenv('DISCORD_WEBHOOK_URL_ACTIONABLE') or os.getenv('DISCORD_WEBHOOK_URL','')
    return [u.strip() for u in raw.replace('\n',',').split(',') if u.strip()]
def _parse_utc(v):
    try:
        ts=datetime.fromisoformat(str(v).replace('Z','+00:00'))
        return (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
    except Exception: return None
def resolve_log_ts(row):
    return _parse_utc((row or {}).get('ts_utc') or (row or {}).get('bud_ts'))
def _boundary(now, raw):
    parts=str(raw or 'SUN 20:30').upper().split(); day=DAYS.get(parts[0],6); hh,mm=[int(x) for x in parts[1].split(':')[:2]]
    week_start=datetime.combine(now.date()-timedelta(days=now.weekday()), time(0,0,tzinfo=timezone.utc))
    return week_start+timedelta(days=day,hours=hh,minutes=mm)
def due_window(now, cfg):
    boundary=_boundary(now, cfg.get('boundary','SUN 20:30'))
    if now < boundary:
        if now.weekday() == boundary.weekday():
            return None
        boundary -= timedelta(days=7)
    iso=f"{boundary.isocalendar().year}-W{boundary.isocalendar().week:02d}"
    return iso, boundary-timedelta(days=6,hours=20,minutes=30), boundary
def _iter_jsonl(path):
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try: yield json.loads(line)
            except Exception: continue
def _in(ts,start,end): return ts is not None and start <= ts <= end
def _float(v):
    try: return float(v or 0)
    except Exception: return 0.0
def _precision_line(title,hits,n):
    if n:
        lo,hi=_wilson_interval(hits,n); return f"{title}: {hits/n*100:.0f}% — 95% Wilson CI: [{lo*100:.0f}%, {hi*100:.0f}%] (N={n})"
    return f"{title}: nincs értékelhető minta (N=0)"        
def _journal_rows(path,start,end):
    if not path.exists(): return []
    with path.open(newline='',encoding='utf-8') as fh:
        return [r for r in csv.DictReader(fh) if _in(_parse_utc(r.get('analysis_timestamp')),start,end)]
def build_embed(now=None):
    now=(now or datetime.now(timezone.utc)).astimezone(timezone.utc); cfg=getattr(settings,'WEEKLY_REPORT',{}) or load(ROOT/'config'/'analysis_settings.json').get('weekly_report',{})
    win=due_window(now,cfg)
    if not cfg.get('enabled',True) or not win: return None
    iso,start,end=win; journal=PUBLIC/'journal'/'trade_journal.csv'; rows=_journal_rows(journal,start,end); ledger_rows=rows_between(PUBLIC/'journal'/'trade_ledger.csv',start,end)
    closed=[r for r in ledger_rows if str(r.get('outcome') or '').strip().lower()!='expired']
    assets=Counter(str(r.get('asset') or 'N/A').upper() for r in closed); hits=sum(1 for r in closed if str(r.get('outcome') or '').strip().lower() in {'tp1_closed','take_profit_2_hit'})
    asset_txt=', '.join(f'{a}: {n}' for a,n in assets.items()) or 'nincs adat'; n=len(closed)
    line1=f"Címkézett ügyletek: {n} (eszközönként: {asset_txt})" + (" — nincs adat" if not n else "")
    line2=_precision_line('Precision (élő)',hits,n)
    shadow=[r for r in rows if str(r.get('mode') or '').strip().lower()=='suppressed_momentum' and str(r.get('validation_outcome') or '').strip().lower() in {'tp_hit','tp1_closed','stopped'}]
    shadow_hits=sum(1 for r in shadow if str(r.get('validation_outcome') or '').strip().lower() in {'tp_hit','tp1_closed'})
    line2b=_precision_line('Momentum (árnyék)',shadow_hits,len(shadow))    
    ledger_stats=stats(ledger_rows); line3=f"Realizált heti PnL: ${ledger_stats['pnl']:.2f} (vesztes ügyletek: {ledger_stats['losses']})"
    lifecycle=load(PUBLIC/'_position_lifecycle_state.json').get('positions') or {}; expired=sum(1 for p in lifecycle.values() if isinstance(p,dict) and str(p.get('close_reason')).lower()=='expired' and _in(_parse_utc(p.get('closed_at_utc')),start,end))
    delivered=sum(1 for r in _iter_jsonl(PUBLIC/'monitoring'/'webhook_delivery.jsonl') if r.get('script')=='notify_discord' and r.get('ok') and _in(_parse_utc(r.get('ts_utc')),start,end))
    line4=f"Jelzés-jelöltek / kiküldött ENTRY / lejárt: {len(rows)}/{delivered}/{expired}"
    reasons=Counter()
    for path in (PUBLIC/'debug'/'entry_gates').glob('entry_gates_*.jsonl') if (PUBLIC/'debug'/'entry_gates').exists() else []:
        for r in _iter_jsonl(path):
            if _in(resolve_log_ts(r),start,end): reasons.update(r.get('reasons') or ([r.get('reason')] if r.get('reason') else []))
    line5='Kapu-vétók top 5: '+(', '.join(f'{k}: {v}' for k,v in reasons.most_common(5)) or 'nincs adat')
    feas=[r for r in _iter_jsonl(PUBLIC/'debug'/'entry_gate_gap_log.jsonl') if r.get('gate')=='profit_target_feasibility' and _in(_parse_utc(r.get('ts_utc')),start,end)]
    passed=sum(1 for r in feas if r.get('result')=='pass'); line6=f"Feasibility pass-arány: {(passed/len(feas)*100):.0f}% ({len(feas)} kiértékelés)" if feas else "Feasibility pass-arány: nincs adat (0 kiértékelés)"
    desc='\n'.join([line1,line2,line2b,line3,line4,line5,line6])    
    return {'title':f'📊 Heti mérési riport – {start.date().isoformat()}–{end.date().isoformat()}','description':desc,'footer':{'text':'(a heti határ utáni első futáson küldve)'},'color':0x2ECC71,'_iso_week':iso}
def main():
    embed=build_embed();
    if not embed: return 0
    iso=embed.pop('_iso_week'); st=load(STATE)
    if st.get('last_reported_iso_week')==iso: return 0
    sent=False
    for u in urls():
        if requests:
            try: sent=_webhook_log_response(LOGGER,'weekly_report_discord','actionable',requests.post(u,json={'embeds':[embed]},timeout=8)) or sent
            except Exception as exc: _webhook_log_exception(LOGGER,'weekly_report_discord','actionable',exc)
    if sent:
        STATE.parent.mkdir(parents=True,exist_ok=True); STATE.write_text(json.dumps({'last_reported_iso_week':iso,'sent_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z')},indent=2),encoding='utf-8')
    return 0
if __name__=='__main__':
    if any(a in {'-h','--help'} for a in sys.argv[1:]): print('usage: weekly_report_discord.py [--help]'); raise SystemExit(0)
    raise SystemExit(main())
