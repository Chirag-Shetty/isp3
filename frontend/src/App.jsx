import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity, AlertTriangle, ShieldCheck, ChevronDown, ChevronUp,
  Footprints, Armchair, CircleUserRound, ArrowDownUp, ArrowUpDown,
  TrendingDown, Radar, WifiOff, Users
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';

// ── Class config ───────────────────────────────────────────────────────────────
const CC = {
  'NO-FALL':                   { label:'No Fall',             icon:ShieldCheck,   color:'#34d399', glow:'rgba(52,211,153,0.18)',  border:'rgba(52,211,153,0.35)',  cat:'safe' },
  'FALL':                      { label:'Fall Detected',       icon:AlertTriangle, color:'#f87171', glow:'rgba(248,113,113,0.22)', border:'rgba(248,113,113,0.45)', cat:'alert' },
  Standing_walk:               { label:'Walking / Standing',  icon:Footprints,    color:'#22d3ee', glow:'rgba(34,211,238,0.18)',  border:'rgba(34,211,238,0.35)',  cat:'safe' },
  Sitting_chair:               { label:'Sitting on Chair',    icon:Armchair,      color:'#34d399', glow:'rgba(52,211,153,0.18)',  border:'rgba(52,211,153,0.35)',  cat:'safe' },
  sitting_floor:               { label:'Sitting on Floor',    icon:CircleUserRound,color:'#a78bfa',glow:'rgba(167,139,250,0.18)',border:'rgba(167,139,250,0.35)', cat:'safe' },
  Stand_Sit_chair_transition:  { label:'Stand ↔ Chair',       icon:ArrowDownUp,   color:'#fbbf24', glow:'rgba(251,191,36,0.18)',  border:'rgba(251,191,36,0.35)',  cat:'transition' },
  chair_floor_transition:      { label:'Chair ↔ Floor',       icon:ArrowUpDown,   color:'#fb923c', glow:'rgba(251,146,60,0.18)',  border:'rgba(251,146,60,0.35)',  cat:'transition' },
  stand_floor_transition:      { label:'Stand → Floor (Fall)',icon:TrendingDown,  color:'#f87171', glow:'rgba(248,113,113,0.22)', border:'rgba(248,113,113,0.45)', cat:'alert' },
};
const DEFAULT_CLASS_ORDER = ['Standing_walk','Sitting_chair','sitting_floor','Stand_Sit_chair_transition','chair_floor_transition','stand_floor_transition'];
const BINARY_CLASS_ORDER  = ['NO-FALL','FALL'];

const EC2_IP          = import.meta.env.VITE_EC2_IP  || '43.205.167.81';
const DEFAULT_WS_URL  = `ws://${EC2_IP}/ws`;
const DEFAULT_API_URL = `http://${EC2_IP}`;
const DEFAULT_DEVICE_ID = 'rpi-1';

function cfg(name) {
  return CC[name] || { label: name||'Unknown', icon:Activity, color:'#64748b', glow:'rgba(100,116,139,0.1)', border:'rgba(100,116,139,0.3)', cat:'safe' };
}

// ── Custom Tooltip ─────────────────────────────────────────────────────────────
function ChartTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:'#0a0f1e', border:'1px solid rgba(255,255,255,0.1)', borderRadius:10, padding:'8px 14px' }}>
      <div style={{ color:'#94a3b8', fontSize:'0.75rem' }}>{payload[0].payload.name}</div>
      <div style={{ color:'#f1f5f9', fontFamily:'JetBrains Mono, monospace', fontWeight:600 }}>{payload[0].value}%</div>
    </div>
  );
}

// ── Probability bar chart ──────────────────────────────────────────────────────
function ProbChart({ probs, classOrder }) {
  if (!probs || probs.length === 0) return null;
  const order = probs.length === 2 ? BINARY_CLASS_ORDER : (classOrder || DEFAULT_CLASS_ORDER);
  if (probs.length < order.length) return null;
  const data = order.map((cls, i) => ({
    name:  CC[cls]?.label || cls,
    value: +(probs[i] * 100).toFixed(1),
    color: CC[cls]?.color || '#64748b',
  }));
  return (
    <ResponsiveContainer width="100%" height={150}>
      <BarChart data={data} layout="vertical" margin={{ left:4, right:20, top:4, bottom:4 }}>
        <XAxis type="number" domain={[0,100]} tick={{ fill:'#475569', fontSize:11, fontFamily:'JetBrains Mono, monospace' }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="name" width={100} tick={{ fill:'#94a3b8', fontSize:11 }} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
        <Bar dataKey="value" radius={[0,6,6,0]} barSize={12}>
          {data.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.85} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── History row ────────────────────────────────────────────────────────────────
function HRow({ event, delay = 0, classOrder }) {
  const [open, setOpen] = useState(false);
  const c    = cfg(event.class_name);
  const Icon = c.icon;
  const ts   = event.timestamp || event.ts;
  const timeStr = ts
    ? new Date(ts).toLocaleTimeString([], { hour12:false, hour:'2-digit', minute:'2-digit', second:'2-digit' })
    : '--:--:--';

  return (
    <div
      className={`h-row ${c.cat==='alert'?'is-alert':''} ${c.cat==='transition'?'is-transition':''}`}
      style={{ borderLeftColor: c.color, animationDelay: `${delay}ms` }}
    >
      <div className="h-header" onClick={() => setOpen(!open)}>
        <div className="h-left">
          <div className="h-icon" style={{ color: c.color }}><Icon size={16} /></div>
          <span className="h-label" style={{ color: c.color }}>{c.label}</span>
          <span className="h-time">{timeStr}</span>
        </div>
        <div className="h-right">
          <span className="h-conf" style={{ color: c.color }}>{(event.confidence*100).toFixed(1)}%</span>
          {open ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
        </div>
      </div>
      {open && (
        <div className="h-details">
          <div className="h-detail-grid">
            <div className="h-detail-item"><span className="detail-lbl">Points</span>{event.n_points}</div>
            <div className="h-detail-item"><span className="detail-lbl">Z-Mean</span>{event.z_mean?.toFixed(3) ?? 'N/A'}</div>
            <div className="h-detail-item"><span className="detail-lbl">Height</span>{event.height_range?.toFixed(3) ?? 'N/A'}</div>
            <div className="h-detail-item"><span className="detail-lbl">X-Mean</span>{event.x_mean?.toFixed(3) ?? 'N/A'}</div>
          </div>
          <div>
            <div className="prob-section-title">Class Probabilities</div>
            <div className="prob-chart-wrap"><ProbChart probs={event.probs} classOrder={classOrder} /></div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Human Count Panel ─────────────────────────────────────────────────────────
function HumanCountPanel({ count, connected }) {
  const hasHumans = typeof count === 'number' && count > 0;
  const displayCount = typeof count === 'number' ? count : '—';

  return (
    <div className={`human-panel ${hasHumans ? 'active' : ''}`}>
      {/* Animated ring */}
      {hasHumans && <div className="human-ring" />}
      {hasHumans && <div className="human-ring ring2" />}

      <div className="human-panel-inner">
        <div className="human-icon-wrap">
          <Users size={28} className={hasHumans ? 'human-icon-active' : 'human-icon-idle'} />
        </div>
        <div className="human-label">Humans in Frame</div>
        <div className={`human-number ${hasHumans ? 'num-active' : 'num-idle'}`}>
          {displayCount}
        </div>
        <div className={`human-status-pill ${hasHumans ? 'pill-active' : 'pill-idle'}`}>
          {hasHumans ? `${count} DETECTED` : connected ? 'MONITORING' : 'OFFLINE'}
        </div>
      </div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [telemetry, setTelemetry] = useState([]);
  const [connected, setConnected]  = useState(false);
  const deviceId = import.meta.env.VITE_DEVICE_ID || DEFAULT_DEVICE_ID;

  useEffect(() => {
    const wsUrl   = import.meta.env.VITE_WS_URL  || DEFAULT_WS_URL;
    const apiBase = import.meta.env.VITE_API_URL  || DEFAULT_API_URL;

    if (apiBase) {
      fetch(`${apiBase}/history?device_id=${encodeURIComponent(deviceId)}&limit=200`)
        .then(r => (r.ok ? r.json() : []))
        .then(data => { if (Array.isArray(data) && data.length) setTelemetry(data); })
        .catch(() => {});
    }

    if (!wsUrl) return () => {};
    let ws, retryTimer;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); retryTimer = setTimeout(connect, 2000); };
      ws.onerror = () => { setConnected(false); ws.close(); };
      ws.onmessage = (evt) => {
        try {
          const row = JSON.parse(evt.data);
          setTelemetry(prev => {
            const next = [row, ...prev];
            if (next.length > 200) next.pop();
            return next;
          });
        } catch (e) { console.warn('WS parse error', e); }
      };
    };
    connect();
    return () => { if (retryTimer) clearTimeout(retryTimer); if (ws) ws.close(); };
  }, [deviceId]);

  const classOrder = useMemo(() => {
    const hasBinary = telemetry.some(e => e.class_name === 'FALL' || e.class_name === 'NO-FALL');
    const hasMulti  = telemetry.some(e => DEFAULT_CLASS_ORDER.includes(e.class_name));
    if (hasBinary && !hasMulti) return BINARY_CLASS_ORDER;
    return DEFAULT_CLASS_ORDER;
  }, [telemetry]);

  const latest     = telemetry[0];
  const history    = telemetry.slice(1);
  const latestCfg  = latest ? cfg(latest.class_name) : null;
  const LatestIcon = latestCfg?.icon ?? Activity;
  const latestTs   = latest?.timestamp || latest?.ts;
  const humanCount = typeof latest?.human_count === 'number' ? latest.human_count : null;
  const avgConf    = telemetry.length
    ? (telemetry.reduce((s,e) => s+(e.confidence||0), 0) / telemetry.length * 100).toFixed(1)
    : '—';

  return (
    <div className="app-root">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="brand-icon"><Radar size={26} /></div>
          <div>
            <div className="brand-title">RadarWatch</div>
            <div className="brand-sub">IWR6843 · Real-time Activity Monitor · AWS Rule-Based</div>
          </div>
        </div>
        <div className={`live-badge ${connected?'on':'off'}`}>
          {connected ? <span className="live-dot"/> : <WifiOff size={12}/>}
          {connected ? 'Live Stream' : 'Offline'}
        </div>
      </header>

      {/* ── Stats row — 3 cards, no fall count ── */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Events Logged</div>
          <div className="stat-value">{telemetry.length}</div>
          <div className="stat-sub">in session</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Current Status</div>
          <div className="stat-value" style={{ color: latestCfg?.color ?? '#64748b', fontSize:'1.1rem', paddingTop:4 }}>
            {latestCfg?.label ?? '—'}
          </div>
          <div className="stat-sub">latest inference</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Confidence</div>
          <div className="stat-value cyan">{avgConf}{telemetry.length?'%':''}</div>
          <div className="stat-sub">model score</div>
        </div>
      </div>

      {/* ── Main 2-column layout ── */}
      <div className="main-grid">

        {/* Left column */}
        <div className="left-col">
          {!latest ? (
            <div className="empty">
              <Radar size={72} className="empty-icon" />
              <h2>Waiting for radar stream…</h2>
              <p>Start your RPi pipeline and <code>aws_watcher.py</code></p>
            </div>
          ) : (
            <>
              {/* ── Current Activity ── */}
              <div
                className="activity-card glass"
                style={{
                  '--act-color':  latestCfg.color,
                  '--act-glow':   latestCfg.glow,
                  '--act-border': latestCfg.border,
                }}
              >
                <div className="section-hdr"><span>Current Activity</span></div>

                <div className="activity-top">
                  <div className={`act-icon-wrap ${latestCfg.cat==='alert'?'alert':''}`}>
                    <LatestIcon size={38} />
                  </div>
                  <div className="act-info">
                    <div className="act-name">{latestCfg.label}</div>
                    <div className="act-badge">
                      {latestCfg.cat==='alert' ? <AlertTriangle size={11}/> : <ShieldCheck size={11}/>}
                      {latest.class_name}
                    </div>
                    <div className="act-ts">
                      {latestTs ? new Date(latestTs).toLocaleString() : '--'}
                    </div>
                  </div>
                  <div className="act-conf-block">
                    <div className="conf-label">Confidence</div>
                    <div className="conf-num">{(latest.confidence*100).toFixed(1)}%</div>
                    <div className="conf-bar-wrap">
                      <div className="conf-bar" style={{ width:`${(latest.confidence*100).toFixed(1)}%` }}/>
                    </div>
                  </div>
                </div>

                {/* Probabilities */}
                <div className="prob-section">
                  <div className="prob-section-title">Class Probabilities</div>
                  <div className="prob-chart-wrap">
                    <ProbChart probs={latest.probs} classOrder={classOrder} />
                  </div>
                </div>

                {/* Metrics */}
                <div className="metrics-row">
                  {[
                    { label:'Points',       val: latest.n_points },
                    { label:'Z-Mean (m)',   val: latest.z_mean?.toFixed(3) ?? 'N/A' },
                    { label:'Height Range', val: latest.height_range?.toFixed(3) ?? 'N/A' },
                    { label:'X-Mean (m)',   val: latest.x_mean?.toFixed(3) ?? 'N/A' },
                  ].map(m => (
                    <div className="metric-box" key={m.label}>
                      <div className="metric-lbl">{m.label}</div>
                      <div className="metric-val">{m.val}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* ── History ── */}
              {history.length > 0 && (
                <div className="history-section">
                  <div className="section-hdr">
                    <span>Inference History</span>
                    <span className="section-pill">{history.length} rows</span>
                  </div>
                  <div className="history-list">
                    {history.map((ev, i) => (
                      <HRow key={ev.id ?? i} event={ev} delay={i*20} classOrder={classOrder} />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Right column — Human Count */}
        <div className="right-col">
          <HumanCountPanel count={humanCount} connected={connected} />

          {/* Quick stats below human panel */}
          {latest && (
            <div className="right-stats">
              <div className="rstat-card">
                <div className="rstat-label">Device</div>
                <div className="rstat-val mono">{latest.device_id ?? deviceId}</div>
              </div>
              <div className="rstat-card">
                <div className="rstat-label">Points Detected</div>
                <div className="rstat-val mono">{latest.n_points ?? '—'}</div>
              </div>
              <div className="rstat-card">
                <div className="rstat-label">Height Range</div>
                <div className="rstat-val mono">{latest.height_range?.toFixed(2) ?? '—'} m</div>
              </div>
              <div className="rstat-card">
                <div className="rstat-label">Z-Mean</div>
                <div className="rstat-val mono">{latest.z_mean?.toFixed(2) ?? '—'} m</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
