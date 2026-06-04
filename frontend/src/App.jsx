import React, { useState, useEffect, useMemo } from 'react';
import {
  Activity, AlertTriangle, ShieldCheck, ChevronDown, ChevronUp,
  Footprints, Armchair, CircleUserRound, ArrowDownUp, ArrowUpDown,
  TrendingDown, Radar, Wifi, WifiOff, BarChart3, Zap, Clock, Hash
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';

// ── Class config ──────────────────────────────────────────────────────────────
const CC = {
  'NO-FALL':                   { label:'No Fall',             icon:ShieldCheck,   color:'#34d399', glow:'rgba(52,211,153,0.18)',  border:'rgba(52,211,153,0.35)',  cat:'safe' },
  'FALL':                      { label:'Fall Detected',       icon:AlertTriangle, color:'#f87171', glow:'rgba(248,113,113,0.22)', border:'rgba(248,113,113,0.45)', cat:'alert' },
  Standing_walk:              { label:'Walking / Standing',    icon:Footprints,     color:'#22d3ee', glow:'rgba(34,211,238,0.18)',  border:'rgba(34,211,238,0.35)',  cat:'safe' },
  Sitting_chair:              { label:'Sitting on Chair',      icon:Armchair,       color:'#34d399', glow:'rgba(52,211,153,0.18)',  border:'rgba(52,211,153,0.35)',  cat:'safe' },
  sitting_floor:              { label:'Sitting on Floor',      icon:CircleUserRound,color:'#a78bfa', glow:'rgba(167,139,250,0.18)',border:'rgba(167,139,250,0.35)', cat:'safe' },
  Stand_Sit_chair_transition: { label:'Stand ↔ Chair',         icon:ArrowDownUp,    color:'#fbbf24', glow:'rgba(251,191,36,0.18)',  border:'rgba(251,191,36,0.35)',  cat:'transition' },
  chair_floor_transition:     { label:'Chair ↔ Floor',         icon:ArrowUpDown,    color:'#fb923c', glow:'rgba(251,146,60,0.18)',  border:'rgba(251,146,60,0.35)',  cat:'transition' },
  stand_floor_transition:     { label:'Stand → Floor (Fall)',  icon:TrendingDown,   color:'#f87171', glow:'rgba(248,113,113,0.22)', border:'rgba(248,113,113,0.45)', cat:'alert' },
};
const DEFAULT_CLASS_ORDER = ['Standing_walk','Sitting_chair','sitting_floor','Stand_Sit_chair_transition','chair_floor_transition','stand_floor_transition'];
const BINARY_CLASS_ORDER = ['NO-FALL','FALL'];
const EC2_IP = import.meta.env.VITE_EC2_IP || '43.205.167.81';
const DEFAULT_WS_URL  = `ws://${EC2_IP}/ws`;
const DEFAULT_API_URL = `http://${EC2_IP}`;
const DEFAULT_DEVICE_ID = 'rpi-1';

function cfg(name) {
  return CC[name] || { label: name||'Unknown', icon:Activity, color:'#64748b', glow:'rgba(100,116,139,0.1)', border:'rgba(100,116,139,0.3)', cat:'safe' };
}

// ── Custom Tooltip ────────────────────────────────────────────────────────────
function ChartTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:'#0a0f1e', border:'1px solid rgba(255,255,255,0.1)', borderRadius:10, padding:'8px 14px' }}>
      <div style={{ color:'#94a3b8', fontSize:'0.75rem' }}>{payload[0].payload.name}</div>
      <div style={{ color:'#f1f5f9', fontFamily:'JetBrains Mono, monospace', fontWeight:600 }}>{payload[0].value}%</div>
    </div>
  );
}

// ── Probability bar chart ─────────────────────────────────────────────────────
function ProbChart({ probs, classOrder }) {
  if (!probs || probs.length === 0) return null;
  const order = probs.length === 2 ? BINARY_CLASS_ORDER : (classOrder || DEFAULT_CLASS_ORDER);
  if (probs.length < order.length) return null;
  const data = order.map((cls, i) => ({
    name: CC[cls]?.label || cls,
    value: +(probs[i] * 100).toFixed(1),
    color: CC[cls]?.color || '#64748b',
  }));
  return (
    <ResponsiveContainer width="100%" height={170}>
      <BarChart data={data} layout="vertical" margin={{ left:4, right:20, top:4, bottom:4 }}>
        <XAxis type="number" domain={[0,100]} tick={{ fill:'#475569', fontSize:11, fontFamily:'JetBrains Mono, monospace' }} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="name" width={115} tick={{ fill:'#94a3b8', fontSize:11 }} axisLine={false} tickLine={false} />
        <Tooltip content={<ChartTip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
        <Bar dataKey="value" radius={[0,6,6,0]} barSize={13}>
          {data.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.85} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Distribution grid ─────────────────────────────────────────────────────────
function Distribution({ telemetry, classOrder }) {
  const order = classOrder?.length ? classOrder : DEFAULT_CLASS_ORDER;
  const dist = useMemo(() => {
    const counts = {};
    order.forEach(c => counts[c] = 0);
    telemetry.forEach(e => { if (e.class_name && counts[e.class_name] !== undefined) counts[e.class_name]++; });
    const total = telemetry.length || 1;
    return order.map(c => ({ cls: c, count: counts[c], pct: ((counts[c]/total)*100).toFixed(1) }));
  }, [telemetry, order]);

  return (
    <div className="dist-grid">
      {dist.map(({ cls, count, pct }) => {
        const c = cfg(cls);
        const Icon = c.icon;
        return (
          <div key={cls} className="dist-card" style={{ borderLeftColor: c.color }}>
            <div className="dist-icon" style={{ color: c.color }}>
              <Icon size={18} />
            </div>
            <div className="dist-info">
              <div className="dist-name">{c.label}</div>
              <div className="dist-count" style={{ color: c.color }}>
                {count} <span className="dist-pct">({pct}%)</span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── History row ───────────────────────────────────────────────────────────────
function HRow({ event, delay = 0, classOrder }) {
  const [open, setOpen] = useState(false);
  const c = cfg(event.class_name);
  const Icon = c.icon;
  const ts = event.timestamp || event.ts;
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
          <div className="h-icon" style={{ color: c.color }}><Icon size={18} /></div>
          <span className="h-label" style={{ color: c.color }}>{c.label}</span>
          <span className="h-time">{timeStr} · #{event.frame_count}</span>
        </div>
        <div className="h-right">
          <span className="h-conf" style={{ color: c.color }}>{(event.confidence*100).toFixed(1)}%</span>
          {open ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
        </div>
      </div>
      {open && (
        <div className="h-details">
          <div className="h-detail-grid">
            <div className="h-detail-item"><span className="stat-label" style={{display:'block'}}>Points</span>{event.n_points}</div>
            <div className="h-detail-item"><span className="stat-label" style={{display:'block'}}>Z-Mean</span>{event.z_mean?.toFixed(3) ?? 'N/A'}</div>
            <div className="h-detail-item"><span className="stat-label" style={{display:'block'}}>Height Range</span>{event.height_range?.toFixed(3) ?? 'N/A'}</div>
            <div className="h-detail-item"><span className="stat-label" style={{display:'block'}}>X-Mean</span>{event.x_mean?.toFixed(3) ?? 'N/A'}</div>
          </div>
          <div style={{ marginBottom:12 }}>
            <div className="prob-section-title">Class Probabilities</div>
            <div className="prob-chart-wrap"><ProbChart probs={event.probs} classOrder={classOrder} /></div>
          </div>
          {event.window_features && (
            <div>
              <div className="prob-section-title">Raw Window (40×20)</div>
              <pre className="raw-pre">{JSON.stringify(event.window_features, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [telemetry, setTelemetry] = useState([]);
  const [connected, setConnected] = useState(false);
  const deviceId = import.meta.env.VITE_DEVICE_ID || DEFAULT_DEVICE_ID;

  useEffect(() => {
    // Always point directly at EC2 — never try to derive from window.location
    // (frontend on S3/https would cause mixed-content blocks otherwise)
    const wsUrl  = import.meta.env.VITE_WS_URL  || DEFAULT_WS_URL;
    const apiBase = import.meta.env.VITE_API_URL || DEFAULT_API_URL;

    if (apiBase) {
      const historyUrl = `${apiBase}/history?device_id=${encodeURIComponent(deviceId)}&limit=200`;
      fetch(historyUrl)
        .then(res => (res.ok ? res.json() : []))
        .then(data => {
          if (Array.isArray(data) && data.length) setTelemetry(data);
        })
        .catch(() => {});
    }

    if (!wsUrl) return () => {};

    let ws;
    let retryTimer;

    const connect = () => {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retryTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => {
        setConnected(false);
        ws.close();
      };
      ws.onmessage = (evt) => {
        try {
          const row = JSON.parse(evt.data);
          setTelemetry(prev => {
            const next = [row, ...prev];
            if (next.length > 500) next.pop();
            return next;
          });
        } catch (err) {
          console.warn('WS parse error', err);
        }
      };
    };

    connect();

    return () => {
      if (retryTimer) clearTimeout(retryTimer);
      if (ws) ws.close();
    };
  }, [deviceId]);

  const classOrder = useMemo(() => {
    const hasBinary = telemetry.some(e => e.class_name === 'FALL' || e.class_name === 'NO-FALL');
    const hasMulti = telemetry.some(e => DEFAULT_CLASS_ORDER.includes(e.class_name));
    if (hasBinary && !hasMulti) return BINARY_CLASS_ORDER;
    return DEFAULT_CLASS_ORDER;
  }, [telemetry]);

  const latest  = telemetry[0];
  const history = telemetry.slice(1);
  const latestCfg = latest ? cfg(latest.class_name) : null;
  const LatestIcon = latestCfg?.icon ?? Activity;

  const totalFrames = latest?.frame_count ?? 0;
  const fallCount   = telemetry.filter(e => e.is_fall).length;
  const avgConf     = telemetry.length ? (telemetry.reduce((s,e) => s+(e.confidence||0), 0)/telemetry.length*100).toFixed(1) : '—';
  const latestTs = latest?.timestamp || latest?.ts;

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

      {/* ── Stats row ── */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Total Frames</div>
          <div className="stat-value cyan">{totalFrames.toLocaleString()}</div>
          <div className="stat-sub">processed</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Events Logged</div>
          <div className="stat-value">{telemetry.length}</div>
          <div className="stat-sub">in session</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Fall Alerts</div>
          <div className={`stat-value ${fallCount>0?'red':'green'}`}>{fallCount}</div>
          <div className="stat-sub">{fallCount===0?'all clear':'detected'}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Confidence</div>
          <div className="stat-value cyan">{avgConf}{telemetry.length?'%':''}</div>
          <div className="stat-sub">model score</div>
        </div>
      </div>

      {!latest ? (
        <div className="empty">
          <Radar size={72} className="empty-icon" />
          <h2>Waiting for radar stream…</h2>
          <p>Start your RPi pipeline or run simulate_sender.py</p>
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
            <div style={{ marginBottom:12 }}>
              <div className="section-hdr"><span>Current Activity</span></div>
            </div>

            <div className="activity-top">
              <div className={`act-icon-wrap ${latestCfg.cat==='alert'?'alert':''}`}>
                <LatestIcon size={40} />
              </div>
              <div className="act-info">
                <div className="act-name">{latestCfg.label}</div>
                <div className="act-badge">
                  {latestCfg.cat==='alert' ? <AlertTriangle size={11}/> : <ShieldCheck size={11}/>}
                  {latest.class_name}
                </div>
                <div className="act-ts">
                  {latestTs ? new Date(latestTs).toLocaleString() : '--'} · Frame #{latest.frame_count ?? '--'}
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
              <div className="prob-chart-wrap"><ProbChart probs={latest.probs} classOrder={classOrder} /></div>
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

          {/* ── Activity Distribution ── */}
          <div className="dist-section">
            <div className="section-hdr">
              <span>Session Breakdown</span>
              <span className="section-pill">{telemetry.length} events</span>
            </div>
            <Distribution telemetry={telemetry} classOrder={classOrder} />
          </div>

          {/* ── History ── */}
          {history.length > 0 && (
            <div className="history-section">
              <div className="section-hdr">
                <span>Inference History</span>
                <span className="section-pill">{history.length} rows</span>
              </div>
              <div className="history-list">
                {history.map((ev, i) => <HRow key={ev.id ?? i} event={ev} delay={i*25} classOrder={classOrder} />)}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
