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
from scripts.webhook_delivery import log_exception as _webhook_log_exception, log_response as _webhook_log_response
LOGGER=logging.getLogger(__name__)
PUBLIC=Path(os.getenv("NOTIFY_PUBLIC_DIR","public")); STATE=PUBLIC/"monitoring"/"daily_digest_state.json"
def load(p):
    try: return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    except Exception: return {}
def urls():
    raw=os.getenv('DISCORD_WEBHOOK_URL_ACTIONABLE') or os.getenv('DISCORD_WEBHOOK_URL','')
    return [u.strip() for u in raw.replace('\n',',').split(',') if u.strip()]
def main():
    cfg=getattr(settings,'POSITION_LIFECYCLE',{}) or {}; now=datetime.now(timezone.utc); day=now.date().isoformat()
    hh,mm=[int(x) for x in str(cfg.get('daily_digest_utc','20:30')).split(':')[:2]]
    if (now.hour,now.minute)<(hh,mm): return 0
    st=load(STATE)
    if st.get('last_digest_utc_day')==day: return 0
    lifecycle=load(PUBLIC/'_position_lifecycle_state.json').get('positions') or {}
    active=[f"{a}: {p.get('status')} @ {p.get('entry')}" for a,p in lifecycle.items() if isinstance(p,dict) and str(p.get('status')).lower() in {'open','pending'}]
    outcomes={}; journal=Path('reports/trade_journal_labeled.csv')
    if journal.exists():
        for r in csv.DictReader(journal.open(encoding='utf-8')): outcomes[r.get('validation_outcome','')]=outcomes.get(r.get('validation_outcome',''),0)+1
    embed={'title':f'📋 Napi actionable összefoglaló – {day} UTC','description':f"Aktív/pending pozíciók:\n"+("\n".join(active) or 'nincs')+f"\n\nKimenetek: {outcomes or 'N/A'}",'color':0x3498DB}
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
if __name__=='__main__': raise SystemExit(main())
