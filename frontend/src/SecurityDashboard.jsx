import React, { useEffect, useState, useRef } from "react";

// Works on both local and cloud:
// - Locally: uses http://127.0.0.1:5000
// - On cloud (Vercel/Netlify/etc): set REACT_APP_API_URL in your frontend env
// - If no env var, auto-detects based on current hostname
const getApiBase = () => {
  if (process.env.REACT_APP_API_URL) return process.env.REACT_APP_API_URL;
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://127.0.0.1:5000';
  }
  // On cloud: backend URL must be set via REACT_APP_API_URL env var
  // Fallback: same origin (if frontend and backend are on same domain)
  return window.location.origin;
};
const API_BASE = getApiBase();
const STORAGE_KEY   = "sentinel_history";   // { "YYYY-MM-DD": [alert, ...] }
const DISMISSED_KEY = "sentinel_dismissed"; // { "YYYY-MM-DD": [id, ...] }

// ── date helpers ───────────────────────────────────────────────────────────
const todayKey = () => new Date().toISOString().slice(0, 10); // "2025-06-15"
const alertDay = (ts) => {
  const d = new Date(ts);
  return isNaN(d) ? todayKey() : d.toISOString().slice(0, 10);
};
const formatTime = (ts) => {
  if (!ts) return "—";
  const d = new Date(ts);
  return isNaN(d) ? ts : d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};
const formatDate = (ts) => {
  if (!ts) return "";
  const d = new Date(ts);
  return isNaN(d) ? "" : d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
};
const prettyDay = (key) => {
  const d = new Date(key + "T00:00:00");
  const today = new Date(); today.setHours(0,0,0,0);
  const yest  = new Date(today); yest.setDate(yest.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return "Today";
  if (d.toDateString() === yest.toDateString())  return "Yesterday";
  return d.toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
};

// ── localStorage helpers ───────────────────────────────────────────────────
const loadHistory = () => {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch { return {}; }
};
const saveHistory = (h) => {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(h)); } catch {}
};
const loadDismissed = () => {
  try { return JSON.parse(localStorage.getItem(DISMISSED_KEY) || "{}"); }
  catch { return {}; }
};
const saveDismissed = (d) => {
  try { localStorage.setItem(DISMISSED_KEY, JSON.stringify(d)); } catch {}
};

// ── icons ──────────────────────────────────────────────────────────────────
const Icon = ({ d, size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d={d} />
  </svg>
);
const ShieldIcon   = ({ size }) => <Icon size={size} d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />;
const AlertIcon    = ({ size }) => <Icon size={size} d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01" />;
const CameraIcon   = ({ size }) => <Icon size={size} d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2zM12 17a4 4 0 100-8 4 4 0 000 8z" />;
const ClockIcon    = ({ size }) => <Icon size={size} d="M12 22a10 10 0 110-20 10 10 0 010 20zM12 6v6l4 2" />;
const EyeIcon      = ({ size }) => <Icon size={size} d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 9a3 3 0 100 6 3 3 0 000-6z" />;
const RefreshIcon  = ({ size }) => <Icon size={size} d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />;
const XIcon        = ({ size }) => <Icon size={size} d="M18 6L6 18M6 6l12 12" />;
const CheckIcon    = ({ size }) => <Icon size={size} d="M20 6L9 17l-5-5" />;
const HistoryIcon  = ({ size }) => <Icon size={size} d="M12 8v4l3 3M3.05 11a9 9 0 1 0 .5-3M3 4v4h4" />;
const ChevronIcon  = ({ size, open }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
    style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform .3s" }}>
    <path d="M6 9l6 6 6-6" />
  </svg>
);
const TrashIcon    = ({ size }) => <Icon size={size} d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6" />;

// ── Pulse dot ──────────────────────────────────────────────────────────────
const PulseDot = ({ color = "#ef4444" }) => (
  <span style={{ position: "relative", display: "inline-block", width: 10, height: 10 }}>
    <span style={{ position: "absolute", inset: 0, borderRadius: "50%", background: color, animation: "ping 1.4s cubic-bezier(0,0,.2,1) infinite", opacity: 0.6 }} />
    <span style={{ position: "absolute", inset: 1, borderRadius: "50%", background: color }} />
  </span>
);

// ── Alert Card (today) ─────────────────────────────────────────────────────
const AlertCard = ({ alert, onView, onDismiss, index }) => {
  const isNew = index === 0;
  return (
    <div className={`alert-card${isNew ? " alert-card--new" : ""}`} style={{ animationDelay: `${index * 60}ms` }}>
      <div className="card-badge"><AlertIcon size={11} />UNKNOWN FACE</div>
      <div className="card-img-wrap" onClick={() => onView(alert)}>
        <img src={alert.image?.startsWith("http") || alert.image?.startsWith("data:") ? alert.image : `${API_BASE}/${alert.image}`} alt="Intruder" className="card-img"
          onError={(e) => { e.target.src = "https://placehold.co/300x200/1a1a1a/444?text=No+Image"; }} />
        <div className="card-img-overlay"><EyeIcon size={22} /><span>View Full</span></div>
        {isNew && <span className="card-new-tag">LIVE</span>}
      </div>
      <div className="card-meta">
        <div className="card-meta-row"><CameraIcon size={13} /><span>Camera {alert.camera_id ?? "CAM-01"}</span></div>
        <div className="card-meta-row"><ClockIcon size={13} /><span>{formatTime(alert.timestamp)}</span></div>
        <div className="card-meta-date">{formatDate(alert.timestamp)}</div>
      </div>
      <div className="card-status">
        <span className="status-dot-wrap"><PulseDot /></span>
        <span className="status-text">{alert.status ?? "THREAT DETECTED"}</span>
      </div>
      <div className="card-actions">
        <button className="btn btn-dismiss" onClick={() => onDismiss(alert.id)}><XIcon size={14} /> Dismiss</button>
        <button className="btn btn-acknowledge" onClick={() => onView(alert)}><CheckIcon size={14} /> Inspect</button>
      </div>
    </div>
  );
};

// ── History Card (past days, compact) ─────────────────────────────────────
const HistoryCard = ({ alert, onView, dismissed, onDismiss, onRestore }) => (
  <div className={`alert-card history-card${dismissed ? " history-card--dismissed" : ""}`}>
    <div className="card-img-wrap" style={{ aspectRatio: "16/9" }} onClick={() => !dismissed && onView(alert)}>
      <img src={alert.image?.startsWith("http") || alert.image?.startsWith("data:") ? alert.image : `${API_BASE}/${alert.image}`} alt="Historical alert" className="card-img"
        onError={(e) => { e.target.src = "https://placehold.co/300x200/1a1a1a/444?text=No+Image"; }}
        style={{ filter: dismissed ? "grayscale(80%) brightness(0.5)" : "grayscale(40%)" }} />
      {!dismissed && <div className="card-img-overlay"><EyeIcon size={18} /><span>View</span></div>}
      {dismissed  && <div className="dismissed-overlay">DISMISSED</div>}
    </div>
    <div className="card-meta" style={{ padding: "8px 10px 4px" }}>
      <div className="card-meta-row" style={{ fontSize: 10 }}><ClockIcon size={11} />{formatTime(alert.timestamp)}</div>
      <div className="card-meta-row" style={{ fontSize: 10 }}><CameraIcon size={11} />Cam {alert.camera_id ?? "01"}</div>
    </div>
    <div className="card-actions" style={{ borderTop: "1px solid var(--border)" }}>
      {dismissed
        ? <button className="btn" style={{ color: "var(--gold)", fontSize: 9, gridColumn: "1/-1" }} onClick={onRestore}>↩ Restore</button>
        : <>
            <button className="btn btn-dismiss" style={{ fontSize: 9 }} onClick={onDismiss}><XIcon size={11} /> Hide</button>
            <button className="btn btn-acknowledge" style={{ fontSize: 9 }} onClick={() => onView(alert)}><EyeIcon size={11} /> View</button>
          </>
      }
    </div>
  </div>
);

// ── History Day Group ──────────────────────────────────────────────────────
const HistoryGroup = ({ dayKey, alerts, dismissedIds, onView, onDismiss, onRestore, onClearDay }) => {
  const [open, setOpen] = useState(true);
  const hiddenCount = alerts.filter(a => dismissedIds.has(a.id)).length;

  return (
    <div className="history-group">
      <div className="history-group-header" onClick={() => setOpen(o => !o)}>
        <div className="hg-left">
          <ChevronIcon size={15} open={open} />
          <span className="hg-date">{prettyDay(dayKey)}</span>
          <span className="hg-count">{alerts.length} alert{alerts.length !== 1 ? "s" : ""}</span>
          {hiddenCount > 0 && <span className="hg-hidden">{hiddenCount} hidden</span>}
        </div>
        <button className="hg-clear" title="Clear this day from history"
          onClick={(e) => { e.stopPropagation(); onClearDay(dayKey); }}>
          <TrashIcon size={13} />
        </button>
      </div>

      {open && (
        <div className="history-grid">
          {alerts.map(alert => (
            <HistoryCard
              key={alert.id}
              alert={alert}
              dismissed={dismissedIds.has(alert.id)}
              onView={onView}
              onDismiss={() => onDismiss(dayKey, alert.id)}
              onRestore={() => onRestore(dayKey, alert.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ── Lightbox ───────────────────────────────────────────────────────────────
const Lightbox = ({ alert, onClose }) => {
  useEffect(() => {
    const h = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);

  if (!alert) return null;
  return (
    <div className="lightbox-backdrop" onClick={onClose}>
      <div className="lightbox" onClick={(e) => e.stopPropagation()}>
        <button className="lightbox-close" onClick={onClose}><XIcon size={18} /></button>
        <img src={alert.image?.startsWith("http") ? alert.image : `${API_BASE}/${alert.image}`} alt="Captured threat" className="lightbox-img"
          onError={(e) => { e.target.src = "https://placehold.co/720x400/1a1a1a/444?text=Image+Not+Found"; }} />
        <div className="lightbox-info">
          <div className="lightbox-label">Alert ID</div><div className="lightbox-val">#{alert.id}</div>
          <div className="lightbox-label">Status</div><div className="lightbox-val" style={{ color: "#ef4444" }}>{alert.status}</div>
          <div className="lightbox-label">Timestamp</div><div className="lightbox-val">{formatDate(alert.timestamp)} · {formatTime(alert.timestamp)}</div>
          <div className="lightbox-label">Camera</div><div className="lightbox-val">Camera {alert.camera_id ?? "CAM-01"}</div>
        </div>
      </div>
    </div>
  );
};

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  const today = todayKey();

  const [alerts, setAlerts]       = useState([]);
  const [lastFetch, setLastFetch] = useState(null);
  const [isLive, setIsLive]       = useState(true);
  const [selected, setSelected]   = useState(null);
  const [tab, setTab]             = useState("today");
  const [newBurst, setNewBurst]   = useState(false);
  const prevCount = useRef(0);

  // per-day dismissed: { "YYYY-MM-DD": Set<id> }
  const [dismissed, setDismissed] = useState(() => {
    const raw = loadDismissed();
    return Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, new Set(v)]));
  });

  // history archive: { "YYYY-MM-DD": [alert, ...] } — never includes today
  const [history, setHistory] = useState(() => {
    const h = loadHistory();
    delete h[today];
    return h;
  });

  // persist to localStorage
  useEffect(() => {
    const raw = Object.fromEntries(Object.entries(dismissed).map(([k, v]) => [k, [...v]]));
    saveDismissed(raw);
  }, [dismissed]);

  useEffect(() => { saveHistory(history); }, [history]);

  // ── fetch + daily-archive logic ────────────────────────────────────────
  const fetchAlerts = async () => {
    try {
      const res  = await fetch(`${API_BASE}/alerts`);
      const data = await res.json();

      // Any alert NOT from today → move to history archive
      const pastAlerts = data.filter(a => alertDay(a.timestamp) !== today);
      if (pastAlerts.length > 0) {
        setHistory(prev => {
          const next = { ...prev };
          pastAlerts.forEach(a => {
            const dk = alertDay(a.timestamp);
            const existing = next[dk] ?? [];
            if (!existing.find(e => e.id === a.id)) next[dk] = [...existing, a];
          });
          return next;
        });
      }

      // Live feed = today only
      const todayAlerts = data.filter(a => alertDay(a.timestamp) === today);
      setAlerts(todayAlerts);
      setLastFetch(new Date());

      if (todayAlerts.length > prevCount.current) {
        setNewBurst(true);
        setTimeout(() => setNewBurst(false), 800);
      }
      prevCount.current = todayAlerts.length;
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => { fetchAlerts(); }, []);
  useEffect(() => {
    if (!isLive) return;
    const id = setInterval(fetchAlerts, 3000);
    return () => clearInterval(id);
  }, [isLive]);

  // ── today dismissed ────────────────────────────────────────────────────
  const todayDismissed = dismissed[today] ?? new Set();
  const dismissToday    = (id) => setDismissed(prev => ({ ...prev, [today]: new Set([...(prev[today] ?? new Set()), id]) }));
  const restoreAllToday = ()   => setDismissed(prev => ({ ...prev, [today]: new Set() }));

  // ── history dismissed ──────────────────────────────────────────────────
  const dismissHistory = (dk, id) =>
    setDismissed(prev => ({ ...prev, [dk]: new Set([...(prev[dk] ?? new Set()), id]) }));
  const restoreHistory = (dk, id) =>
    setDismissed(prev => { const s = new Set(prev[dk] ?? new Set()); s.delete(id); return { ...prev, [dk]: s }; });
  const clearHistoryDay = (dk) => {
    setHistory(prev => { const n = { ...prev }; delete n[dk]; return n; });
    setDismissed(prev => { const n = { ...prev }; delete n[dk]; return n; });
  };

  // ── computed ───────────────────────────────────────────────────────────
  const visibleToday = alerts.filter(a => !todayDismissed.has(a.id));
  const historyDays  = Object.keys(history).sort((a, b) => b.localeCompare(a));
  const totalHistory = Object.values(history).reduce((s, arr) => s + arr.length, 0);

  return (
    <>
      <style>{CSS}</style>

      {/* HEADER */}
      <header className="header">
        <div className="header-left">
          <div className={`logo-wrap${newBurst ? " logo-burst" : ""}`}>
            <ShieldIcon size={28} />
          </div>
          <div>
            <div className="header-title">SENTINEL</div>
            <div className="header-sub">Security Intelligence Dashboard</div>
          </div>
        </div>

        <div className="header-center">
          <div className="stat-pill">
            <PulseDot color="#ef4444" />
            <span className="stat-num">{visibleToday.length}</span>
            <span className="stat-label">Live Today</span>
          </div>
          <div className="stat-pill">
            <span className="stat-num" style={{ color: "#94a3b8" }}>{alerts.length}</span>
            <span className="stat-label">Total Today</span>
          </div>
          <div className="stat-pill">
            <span className="stat-num" style={{ color: "#a78bfa" }}>{totalHistory}</span>
            <span className="stat-label">In History</span>
          </div>
        </div>

        <div className="header-right">
          <div className="live-badge" style={{ opacity: isLive ? 1 : 0.4 }}>
            {isLive && <PulseDot color="#22c55e" />}{isLive ? "LIVE" : "PAUSED"}
          </div>
          <button className={`icon-btn${!isLive ? " icon-btn--active" : ""}`}
            onClick={() => setIsLive(v => !v)} title={isLive ? "Pause" : "Resume"}>
            {isLive ? "⏸" : "▶"}
          </button>
          <button className="icon-btn" onClick={fetchAlerts} title="Refresh now">
            <RefreshIcon size={15} />
          </button>
        </div>
      </header>

      {/* TABS */}
      <div className="toolbar">
        <div className="filter-tabs">
          <button className={`filter-tab${tab === "today" ? " filter-tab--active" : ""}`} onClick={() => setTab("today")}>
            Today's Alerts {alerts.length > 0 && <span className="tab-badge">{alerts.length}</span>}
          </button>
          <button className={`filter-tab${tab === "history" ? " filter-tab--active filter-tab--history" : ""}`} onClick={() => setTab("history")}>
            <HistoryIcon size={12} /> History{historyDays.length > 0 && <span className="tab-badge tab-badge--purple">{historyDays.length}d</span>}
          </button>
        </div>
        {lastFetch && (
          <div className="last-fetch"><ClockIcon size={12} /> Last sync {formatTime(lastFetch)}</div>
        )}
        {tab === "today" && todayDismissed.size > 0 && (
          <button className="clear-btn" onClick={restoreAllToday}>
            ↩ Restore {todayDismissed.size} dismissed
          </button>
        )}
      </div>

      {/* MAIN */}
      <main className="main">

        {/* ── TODAY ── */}
        {tab === "today" && (
          visibleToday.length === 0
            ? <div className="empty-state">
                <ShieldIcon size={52} />
                <div className="empty-title">All Clear</div>
                <div className="empty-sub">No active threats today. System is monitoring.</div>
                {todayDismissed.size > 0 && (
                  <button className="clear-btn" style={{ marginTop: 8 }} onClick={restoreAllToday}>
                    ↩ Restore {todayDismissed.size} dismissed
                  </button>
                )}
              </div>
            : <div className="grid">
                {visibleToday.map((alert, i) => (
                  <AlertCard key={alert.id} alert={alert} index={i}
                    onView={setSelected} onDismiss={dismissToday} />
                ))}
              </div>
        )}

        {/* ── HISTORY ── */}
        {tab === "history" && (
          historyDays.length === 0
            ? <div className="empty-state">
                <HistoryIcon size={52} />
                <div className="empty-title">No History Yet</div>
                <div className="empty-sub">Past days' alerts will appear here automatically at midnight.</div>
              </div>
            : <div className="history-section">
                <div className="history-banner">
                  <HistoryIcon size={15} />
                  Past alerts are archived here automatically each day. Today's feed resets at midnight.
                </div>
                {historyDays.map(dk => (
                  <HistoryGroup
                    key={dk}
                    dayKey={dk}
                    alerts={history[dk]}
                    dismissedIds={dismissed[dk] ?? new Set()}
                    onView={setSelected}
                    onDismiss={dismissHistory}
                    onRestore={restoreHistory}
                    onClearDay={clearHistoryDay}
                  />
                ))}
              </div>
        )}
      </main>

      {selected && <Lightbox alert={selected} onClose={() => setSelected(null)} />}

      <footer className="footer">
        SENTINEL · Daily reset at midnight · History kept until manually cleared · {new Date().getFullYear()}
      </footer>
    </>
  );
}

// ── Styles ─────────────────────────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg:        #09090b;
    --surface:   #111113;
    --surface2:  #18181b;
    --border:    rgba(255,255,255,0.07);
    --accent:    #ef4444;
    --accent2:   #f97316;
    --gold:      #f59e0b;
    --green:     #22c55e;
    --purple:    #a78bfa;
    --muted:     #52525b;
    --text:      #e4e4e7;
    --text-dim:  #71717a;
    --font-head: 'Rajdhani', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius:    10px;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font-mono); min-height: 100vh; }

  @keyframes ping       { 75%,100% { transform: scale(2); opacity: 0; } }
  @keyframes slideDown  { from { opacity:0; transform:translateY(-12px); } to { opacity:1; transform:translateY(0); } }
  @keyframes cardIn     { from { opacity:0; transform:translateY(18px) scale(.97); } to { opacity:1; transform:translateY(0) scale(1); } }
  @keyframes burstPulse { 0%,100% { filter:drop-shadow(0 0 0 #ef4444); } 50% { filter:drop-shadow(0 0 12px #ef4444); } }
  @keyframes fadeIn     { from { opacity:0; } to { opacity:1; } }

  .header {
    display:flex; align-items:center; justify-content:space-between; gap:16px;
    padding:14px 28px; background:var(--surface); border-bottom:1px solid var(--border);
    position:sticky; top:0; z-index:100; animation:slideDown .4s ease;
  }
  .header-left   { display:flex; align-items:center; gap:12px; }
  .header-center { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .header-right  { display:flex; align-items:center; gap:10px; }

  .logo-wrap {
    color:var(--accent); display:flex; align-items:center; justify-content:center;
    background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.25);
    border-radius:10px; width:48px; height:48px; transition:filter .3s;
  }
  .logo-burst { animation:burstPulse .8s ease; }

  .header-title { font-family:var(--font-head); font-size:22px; font-weight:700; letter-spacing:4px; }
  .header-sub   { font-size:10px; letter-spacing:2px; color:var(--text-dim); text-transform:uppercase; }

  .stat-pill {
    display:flex; align-items:center; gap:6px;
    background:var(--surface2); border:1px solid var(--border);
    border-radius:20px; padding:5px 14px; font-size:11px;
  }
  .stat-num   { font-size:17px; font-weight:600; }
  .stat-label { color:var(--text-dim); }

  .live-badge { display:flex; align-items:center; gap:6px; font-size:11px; letter-spacing:2px; color:var(--green); font-weight:600; transition:opacity .3s; }

  .icon-btn {
    background:var(--surface2); border:1px solid var(--border);
    border-radius:8px; color:var(--text-dim); width:34px; height:34px;
    display:flex; align-items:center; justify-content:center;
    cursor:pointer; font-size:14px; transition:all .2s;
  }
  .icon-btn:hover { border-color:var(--accent); color:var(--accent); }
  .icon-btn--active { border-color:var(--gold); color:var(--gold); }

  .toolbar {
    display:flex; align-items:center; gap:14px; flex-wrap:wrap;
    padding:10px 28px; background:var(--surface); border-bottom:1px solid var(--border);
  }
  .filter-tabs { display:flex; gap:4px; }
  .filter-tab {
    display:flex; align-items:center; gap:6px;
    background:transparent; border:1px solid transparent;
    border-radius:6px; color:var(--text-dim);
    padding:5px 14px; font-size:11px; letter-spacing:1px;
    cursor:pointer; text-transform:uppercase; font-family:var(--font-mono); transition:all .2s;
  }
  .filter-tab:hover { color:var(--text); border-color:var(--border); }
  .filter-tab--active { background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.35); color:var(--accent); }
  .filter-tab--history.filter-tab--active { background:rgba(167,139,250,.10); border-color:rgba(167,139,250,.35); color:var(--purple); }

  .tab-badge { background:rgba(239,68,68,.25); color:var(--accent); border-radius:10px; padding:1px 6px; font-size:9px; font-weight:700; }
  .tab-badge--purple { background:rgba(167,139,250,.2); color:var(--purple); }

  .last-fetch { margin-left:auto; display:flex; align-items:center; gap:5px; font-size:10px; color:var(--text-dim); }
  .clear-btn {
    background:transparent; border:1px solid var(--border); border-radius:6px; color:var(--text-dim);
    padding:4px 12px; font-size:10px; cursor:pointer; font-family:var(--font-mono); transition:all .2s;
  }
  .clear-btn:hover { color:var(--gold); border-color:var(--gold); }

  .main {
    padding:28px; min-height:calc(100vh - 160px);
    background-image:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(255,255,255,.012) 2px,rgba(255,255,255,.012) 4px);
  }

  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px,1fr)); gap:20px; }

  .alert-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:var(--radius); overflow:hidden; position:relative;
    animation:cardIn .35s ease both;
    transition:transform .25s, box-shadow .25s, border-color .25s;
  }
  .alert-card:hover { transform:translateY(-3px); box-shadow:0 12px 40px rgba(0,0,0,.6),0 0 0 1px rgba(239,68,68,.2); border-color:rgba(239,68,68,.25); }
  .alert-card--new  { border-color:rgba(239,68,68,.4); box-shadow:0 0 20px rgba(239,68,68,.12); }
  .alert-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,var(--accent),var(--accent2)); }

  .history-card::before { background:linear-gradient(90deg,var(--purple),#6366f1); }
  .history-card--dismissed { opacity:.55; }
  .history-card--dismissed::before { background:var(--muted); }

  .card-badge { display:flex; align-items:center; gap:5px; font-size:9px; letter-spacing:2px; font-weight:600; color:var(--accent); text-transform:uppercase; padding:10px 12px 6px; }

  .card-img-wrap { position:relative; cursor:pointer; background:#000; aspect-ratio:4/3; overflow:hidden; }
  .card-img { width:100%; height:100%; object-fit:cover; display:block; transition:transform .4s ease, filter .4s ease; filter:grayscale(20%); }
  .card-img-wrap:hover .card-img { transform:scale(1.04); filter:grayscale(0%); }

  .card-img-overlay {
    position:absolute; inset:0; background:rgba(0,0,0,.55);
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:6px; color:white; font-size:11px; letter-spacing:1px; opacity:0; transition:opacity .3s;
  }
  .card-img-wrap:hover .card-img-overlay { opacity:1; }

  .dismissed-overlay {
    position:absolute; inset:0; background:rgba(0,0,0,.55);
    display:flex; align-items:center; justify-content:center;
    color:var(--muted); font-size:10px; letter-spacing:3px; font-weight:700;
  }

  .card-new-tag {
    position:absolute; top:8px; right:8px; background:var(--accent); color:white;
    font-size:9px; letter-spacing:2px; font-weight:700; padding:2px 8px; border-radius:4px;
    animation:ping .9s ease infinite alternate;
  }

  .card-meta { padding:10px 12px 6px; display:flex; flex-direction:column; gap:4px; }
  .card-meta-row { display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text-dim); }
  .card-meta-date { font-size:10px; color:var(--muted); margin-top:2px; }

  .card-status { display:flex; align-items:center; gap:7px; padding:6px 12px; background:rgba(239,68,68,.07); border-top:1px solid rgba(239,68,68,.12); }
  .status-dot-wrap { display:flex; align-items:center; }
  .status-text { font-size:10px; letter-spacing:2px; font-weight:600; color:var(--accent); text-transform:uppercase; }

  .card-actions { display:grid; grid-template-columns:1fr 1fr; border-top:1px solid var(--border); }
  .btn { display:flex; align-items:center; justify-content:center; gap:5px; padding:8px 4px; font-size:10px; letter-spacing:1px; font-family:var(--font-mono); cursor:pointer; border:none; text-transform:uppercase; font-weight:500; transition:background .2s, color .2s; }
  .btn-dismiss    { background:transparent; color:var(--text-dim); border-right:1px solid var(--border); }
  .btn-dismiss:hover { background:rgba(255,255,255,.04); color:var(--text); }
  .btn-acknowledge { background:transparent; color:var(--accent); }
  .btn-acknowledge:hover { background:rgba(239,68,68,.1); }

  /* HISTORY */
  .history-section { display:flex; flex-direction:column; gap:20px; animation:fadeIn .3s ease; }

  .history-banner {
    display:flex; align-items:center; gap:10px;
    background:rgba(167,139,250,.08); border:1px solid rgba(167,139,250,.2);
    border-radius:8px; padding:10px 16px; font-size:11px; color:var(--purple); letter-spacing:.5px;
  }

  .history-group { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; }

  .history-group-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:12px 16px; cursor:pointer; border-bottom:1px solid var(--border); transition:background .2s;
  }
  .history-group-header:hover { background:var(--surface2); }

  .hg-left  { display:flex; align-items:center; gap:10px; }
  .hg-date  { font-family:var(--font-head); font-size:16px; font-weight:600; letter-spacing:1px; }
  .hg-count { font-size:10px; color:var(--text-dim); background:var(--surface2); border:1px solid var(--border); border-radius:10px; padding:2px 8px; }
  .hg-hidden { font-size:10px; color:var(--muted); }

  .hg-clear {
    background:transparent; border:1px solid transparent; border-radius:6px;
    color:var(--muted); padding:4px 8px; cursor:pointer; display:flex; align-items:center; transition:all .2s;
  }
  .hg-clear:hover { border-color:var(--accent); color:var(--accent); }

  .history-grid {
    display:grid; grid-template-columns:repeat(auto-fill, minmax(180px,1fr));
    gap:14px; padding:16px; animation:fadeIn .25s ease;
  }

  .empty-state {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:16px; min-height:60vh; color:var(--muted); animation:fadeIn .4s ease;
  }
  .empty-title { font-family:var(--font-head); font-size:28px; letter-spacing:3px; }
  .empty-sub   { font-size:12px; letter-spacing:1px; }

  .lightbox-backdrop {
    position:fixed; inset:0; z-index:200;
    background:rgba(0,0,0,.85); backdrop-filter:blur(6px);
    display:flex; align-items:center; justify-content:center; padding:20px;
    animation:slideDown .2s ease;
  }
  .lightbox {
    background:var(--surface); border:1px solid rgba(239,68,68,.3);
    border-radius:14px; overflow:hidden; max-width:720px; width:100%;
    box-shadow:0 40px 80px rgba(0,0,0,.7),0 0 0 1px rgba(239,68,68,.15);
    position:relative; display:flex; flex-direction:column;
  }
  .lightbox-close {
    position:absolute; top:12px; right:12px;
    background:rgba(0,0,0,.6); border:1px solid var(--border); border-radius:6px;
    color:var(--text-dim); width:32px; height:32px;
    display:flex; align-items:center; justify-content:center; cursor:pointer; z-index:10; transition:all .2s;
  }
  .lightbox-close:hover { border-color:var(--accent); color:var(--accent); }
  .lightbox-img { width:100%; max-height:420px; object-fit:cover; display:block; }
  .lightbox-info { display:grid; grid-template-columns:auto 1fr; gap:8px 20px; padding:18px 20px; }
  .lightbox-label { font-size:10px; letter-spacing:2px; color:var(--text-dim); text-transform:uppercase; display:flex; align-items:center; }
  .lightbox-val   { font-size:13px; color:var(--text); }

  .footer { text-align:center; padding:16px; font-size:10px; letter-spacing:2px; color:var(--muted); border-top:1px solid var(--border); }
`;