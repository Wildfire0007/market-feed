#!/usr/bin/env python3
"""Send one UTC-day actionable digest for manual trading."""
from __future__ import annotations
import csv,json,os,sys,logging
from datetime import datetime, timezone
from pathlib import Path
try: import requests
except Exception: requests=None
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from config import analysis_settings as settings
from scripts.trade_ledger import rows_between, stats
from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
LOGGER=logging.getLogger(__name__)
PUBLIC=Path(os.getenv("NOTIFY_PUBLIC_DIR","public")); STATE=PUBLIC/"monitoring"/"daily_digest_state.json"
def load(p):
    try: return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    except Exception: return {}
def urls():
    raw=os.getenv('DISCORD_WEBHOOK_URL_ACTIONABLE') or os.getenv('DISCORD_WEBHOOK_URL','')
    return [u.strip() for u in raw.replace('\n',',').split(',') if u.strip()]
def _parse_utc(v):
    try: return datetime.fromisoformat(str(v).replace('Z','+00:00')).astimezone(timezone.utc)
    except Exception: return None
def _same_utc_day(v, day):
    ts=_parse_utc(v); return ts is not None and ts.date().isoformat()==day
def _journal_today(journal: Path, day: str):
    sent=0
    if journal.exists():
        with journal.open(newline='',encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                if _same_utc_day(r.get('analysis_timestamp'), day): sent += 1
    return sent,{}

def _ledger_today(ledger: Path, day: str):
    start=datetime.fromisoformat(day+"T00:00:00+00:00"); end=datetime.fromisoformat(day+"T23:59:59+00:00")
    rows=rows_between(ledger,start,end); outcomes={}
    for r in rows:
        outcome=str(r.get("outcome") or "").strip().lower()
        if outcome: outcomes[outcome]=outcomes.get(outcome,0)+1
    return rows,outcomes
def _entry_cards_today(path: Path, day: str):
    count=0
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try: row=json.loads(line)
            except Exception: continue
            script=str(row.get('script') or '')
            ok=bool(row.get('ok'))
            if script=='notify_discord' and ok and _same_utc_day(row.get('ts_utc'), day): count += 1
    return count                    
def main():
    cfg=getattr(settings,'POSITION_LIFECYCLE',{}) or {}; now=datetime.now(timezone.utc); day=now.date().isoformat()
    hh,mm=[int(x) for x in str(cfg.get('daily_digest_utc','20:30')).split(':')[:2]]
    if (now.hour,now.minute)<(hh,mm): return 0
    st=load(STATE)
    if st.get('last_digest_utc_day')==day: return 0
    lifecycle=load(PUBLIC/'_position_lifecycle_state.json').get('positions') or {}
    active=[f"{a}: {p.get('status')} @ {p.get('entry')}" for a,p in lifecycle.items() if isinstance(p,dict) and str(p.get('status')).lower() in {'open','pending'}]
    expired=sum(1 for p in lifecycle.values() if isinstance(p,dict) and str(p.get('close_reason')).lower()=='expired' and _same_utc_day(p.get('closed_at_utc'), day))
    journal=PUBLIC/'journal'/'trade_journal.csv'
    sent_count,_=_journal_today(journal, day)
    ledger_rows,outcomes=_ledger_today(PUBLIC/'journal'/'trade_ledger.csv', day)    
    entry_cards=_entry_cards_today(PUBLIC/'monitoring'/'webhook_delivery.jsonl', day)    
    ledger_stats=stats(ledger_rows)
    desc=(f"Mai jelzés-jelöltek: {sent_count} / {expired} lejárt\n"
          f"Kiküldött ENTRY kártyák: {entry_cards}\n"
          f"Realizált napi PnL: ${ledger_stats['pnl']:.2f} (vesztes ügyletek: {ledger_stats['losses']})\n\n"
          f"Aktív/pending pozíciók:\n"+("\n".join(active) or 'nincs')+f"\n\nKimenetek: {outcomes or 'N/A'}")
    embed={'title':f'📋 Napi actionable összefoglaló – {day} UTC','description':desc,'color':0x3498DB}
    sent=False    
    for u in urls():
        if requests:
            try:
                resp=requests.post(u,json={'embeds':[embed]},timeout=8)
                sent=_webhook_log_response(LOGGER,'daily_actionable_digest','actionable',resp) or sent
            except Exception as exc:
                _webhook_log_exception(LOGGER,'daily_actionable_digest','actionable',exc)
    if sent:
        STATE.parent.mkdir(parents=True,exist_ok=True); STATE.write_text(json.dumps({'last_digest_utc_day':day,'sent_at_utc':now.isoformat().replace('+00:00','Z')},indent=2),encoding='utf-8')        
    return 0
if __name__=='__main__':
    if any(arg in {'-h', '--help'} for arg in sys.argv[1:]):
        print('usage: daily_actionable_digest.py [--help]')
        raise SystemExit(0)
    raise SystemExit(main())
