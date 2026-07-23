export default {
  async scheduled(event, env, ctx) {
    // Éjszakai őrjárat-cron: CSAK heartbeat-ellenőrzés, pipeline-indítás nélkül
    const nightCron = "*/15 22-23,0-4 * * *";
    if (event.cron === nightCron) {
      ctx.waitUntil(checkHeartbeat(env));
    } else {
      // Napközbeni/esti triggerek: dispatch + őrzés együtt
      ctx.waitUntil(Promise.all([triggerGitHubAction(env, "cron"), checkHeartbeat(env)]));
    }
  },

  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname === "/ping") {
      const res = await triggerGitHubAction(env, "ping_endpoint");
      return new Response(JSON.stringify(res, null, 2), { headers: { "content-type": "application/json" } });
    }
    if (url.pathname === "/heartbeat") {
      const res = await checkHeartbeat(env, true);
      return new Response(JSON.stringify(res, null, 2), { headers: { "content-type": "application/json" } });
    }
    return new Response("Worker fut. Végpontok: /ping (pipeline-indítás), /heartbeat (őr-teszt).");
  }
};

async function triggerGitHubAction(env, source) {
  const ref = env.REF_BRANCH || "main";
  const url = `https://api.github.com/repos/${env.REPO_OWNER}/${env.REPO_NAME}/actions/workflows/${env.WORKFLOW_FILE}/dispatches`;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "cf-worker-td-trigger",
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ ref, inputs: { force_public_sync: "true" } })
    });
    return { ok: resp.ok, status: resp.status, source };
  } catch (e) { return { ok: false, error: e.message, source }; }
}

async function checkHeartbeat(env, forceReport = false) {
  const tickMin = isQuietHoursUTC() ? 15 : 2;             // az adott ablak cron-sűrűsége
  const threshold = isQuietHoursUTC()
    ? Number(env.QUIET_MAX_AGE_MIN || 150)                 // éjjel/hétvégén magasabb küszöb
    : Number(env.MAX_AGE_MIN || 30);
  const hbUrl = `https://raw.githubusercontent.com/${env.REPO_OWNER}/${env.REPO_NAME}/main/public/system_heartbeat.json?ts=${Date.now()}`;
  let ageMin, lastUtc;
  try {
    const r = await fetch(hbUrl, { headers: { "Cache-Control": "no-cache" } });
    if (!r.ok) return { ok: false, note: `heartbeat fetch HTTP ${r.status}` };
    const hb = await r.json();
    lastUtc = hb.last_update_utc;
    ageMin = (Date.now() - Date.parse(lastUtc)) / 60000;
  } catch (e) { return { ok: false, error: e.message }; }

  if (ageMin < threshold && !forceReport) return { ok: true, ageMin: ageMin.toFixed(1) };

  // Állapotmentes sáv-dedup: az első [T, T+tick) sávban és óránkénti
  // eszkalációs sávokban [T+60k, T+60k+tick) küldünk — epizódonként ~1 üzenet/óra.
  const over = ageMin - threshold;
  const inBand = (over % 60) < tickMin;
  if (ageMin >= threshold && !inBand && !forceReport) return { ok: true, suppressed: true, ageMin: ageMin.toFixed(1) };

  const quiet = isQuietHoursUTC();
  const webhook = quiet ? env.DISCORD_WEBHOOK_URL_DIAGNOSTIC : env.DISCORD_WEBHOOK_URL_ACTIONABLE;
  const embed = {
    title: ageMin >= threshold ? "🛰️ CF-őr: TD pipeline heartbeat elavult" : "🛰️ CF-őr teszt",
    description: `Kor: ${ageMin.toFixed(1)} min (küszöb: ${threshold} min)\nUtolsó heartbeat: ${lastUtc}\nCsendes időszak: ${quietLabel()}`,
    color: 15158332
  };
  try {
    const resp = await fetch((webhook || "").trim(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embeds: [embed] })
    });
    return { ok: resp.ok, status: resp.status, ageMin: ageMin.toFixed(1) };
  } catch (e) { return { ok: false, error: e.message }; }
}

function isQuietHoursUTC(now = new Date()) {
  const day = now.getUTCDay();
  if (day === 0 || day === 6) return true;   // szombat/vasárnap = csendes
  const hour = now.getUTCHours();
  return hour >= 22 || hour < 5;
}

function quietLabel(now = new Date()) {
  const day = now.getUTCDay();
  if (day === 0 || day === 6) return "hétvége (diagnosztikai csatorna)";
  const hour = now.getUTCHours();
  if (hour >= 22 || hour < 5) return "éjszaka (diagnosztikai csatorna)";
  return "nem";
}
