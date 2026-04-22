import React, { useState, useEffect, useMemo } from 'react';
import { supabase } from './lib/supabase';
import {
  Activity, AlertTriangle, ShieldCheck, ChevronDown, ChevronUp,
  Footprints, Armchair, CircleUserRound, ArrowDownUp, ArrowUpDown,
  TrendingDown, Radar, Wifi, WifiOff, BarChart3
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts';

// ── Class configuration ──────────────────────────────────────────────────────
const CLASS_CONFIG = {
  Standing_walk: {
    label: 'Walking / Standing',
    icon: Footprints,
    color: '#06b6d4',
    bgGlow: 'rgba(6, 182, 212, 0.15)',
    borderColor: 'rgba(6, 182, 212, 0.6)',
    category: 'safe',
  },
  Sitting_chair: {
    label: 'Sitting (Chair)',
    icon: Armchair,
    color: '#10b981',
    bgGlow: 'rgba(16, 185, 129, 0.15)',
    borderColor: 'rgba(16, 185, 129, 0.6)',
    category: 'safe',
  },
  sitting_floor: {
    label: 'Sitting (Floor)',
    icon: CircleUserRound,
    color: '#8b5cf6',
    bgGlow: 'rgba(139, 92, 246, 0.15)',
    borderColor: 'rgba(139, 92, 246, 0.6)',
    category: 'safe',
  },
  Stand_Sit_chair_transition: {
    label: 'Stand ↔ Chair',
    icon: ArrowDownUp,
    color: '#f59e0b',
    bgGlow: 'rgba(245, 158, 11, 0.15)',
    borderColor: 'rgba(245, 158, 11, 0.6)',
    category: 'transition',
  },
  chair_floor_transition: {
    label: 'Chair ↔ Floor',
    icon: ArrowUpDown,
    color: '#f97316',
    bgGlow: 'rgba(249, 115, 22, 0.15)',
    borderColor: 'rgba(249, 115, 22, 0.6)',
    category: 'transition',
  },
  stand_floor_transition: {
    label: 'Stand ↔ Floor',
    icon: TrendingDown,
    color: '#ef4444',
    bgGlow: 'rgba(239, 68, 68, 0.15)',
    borderColor: 'rgba(239, 68, 68, 0.6)',
    category: 'alert',
  },
};

const CLASS_ORDER = [
  'Standing_walk', 'Sitting_chair', 'sitting_floor',
  'Stand_Sit_chair_transition', 'chair_floor_transition', 'stand_floor_transition'
];

function getClassConfig(className) {
  return CLASS_CONFIG[className] || {
    label: className || 'Unknown',
    icon: Activity,
    color: '#94a3b8',
    bgGlow: 'rgba(148, 163, 184, 0.15)',
    borderColor: 'rgba(148, 163, 184, 0.6)',
    category: 'safe',
  };
}

// ── Probability bar chart for a single event ─────────────────────────────────
function ProbabilityChart({ probs }) {
  if (!probs || probs.length < 6) return null;

  const data = CLASS_ORDER.map((cls, i) => ({
    name: CLASS_CONFIG[cls]?.label || cls,
    value: +(probs[i] * 100).toFixed(1),
    color: CLASS_CONFIG[cls]?.color || '#94a3b8',
  }));

  return (
    <div className="prob-chart-container">
      <ResponsiveContainer width="100%" height={160}>
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 16, top: 4, bottom: 4 }}>
          <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={110} tick={{ fill: '#cbd5e1', fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#f8fafc' }}
            formatter={(v) => [`${v}%`, 'Probability']}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={14}>
            {data.map((entry, idx) => (
              <Cell key={idx} fill={entry.color} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Activity distribution summary ────────────────────────────────────────────
function ActivityDistribution({ telemetry }) {
  const distribution = useMemo(() => {
    const counts = {};
    CLASS_ORDER.forEach(cls => { counts[cls] = 0; });
    telemetry.forEach(ev => {
      if (ev.class_name && counts[ev.class_name] !== undefined) {
        counts[ev.class_name]++;
      }
    });
    const total = telemetry.length || 1;
    return CLASS_ORDER.map(cls => ({
      className: cls,
      count: counts[cls],
      pct: ((counts[cls] / total) * 100).toFixed(1),
    }));
  }, [telemetry]);

  return (
    <div className="dist-grid">
      {distribution.map(({ className, count, pct }) => {
        const cfg = getClassConfig(className);
        const Icon = cfg.icon;
        return (
          <div key={className} className="dist-card" style={{ borderColor: cfg.borderColor }}>
            <div className="dist-card-icon" style={{ color: cfg.color }}>
              <Icon size={18} />
            </div>
            <div className="dist-card-info">
              <span className="dist-card-label">{cfg.label}</span>
              <span className="dist-card-value">{count} <span className="dist-pct">({pct}%)</span></span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Expandable history row ───────────────────────────────────────────────────
function EventRow({ event }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = getClassConfig(event.class_name);
  const Icon = cfg.icon;

  return (
    <div
      className={`history-row history-${cfg.category}`}
      style={{ borderLeftColor: cfg.color }}
    >
      <div className="history-header" onClick={() => setExpanded(!expanded)}>
        <div className="history-summary">
          <div className="history-icon" style={{ color: cfg.color }}><Icon size={20} /></div>
          <strong style={{ color: cfg.color }}>{cfg.label}</strong>
          <span className="history-time">
            {new Date(event.timestamp).toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            &nbsp;|&nbsp; Frame #{event.frame_count}
          </span>
        </div>
        <div className="history-actions">
          <span className="history-conf" style={{ color: cfg.color }}>
            {(event.confidence * 100).toFixed(1)}%
          </span>
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </div>

      {expanded && (
        <div className="history-details">
          <div className="detail-grid">
            <div><span className="label">Points</span>{event.n_points}</div>
            <div><span className="label">Z-Mean</span>{event.z_mean?.toFixed(3) || 'N/A'}</div>
            <div><span className="label">Height Range</span>{event.height_range?.toFixed(3) || 'N/A'}</div>
            <div><span className="label">X-Mean</span>{event.x_mean?.toFixed(3) || 'N/A'}</div>
          </div>
          <div style={{ marginTop: '16px' }}>
            <span className="label">Class Probabilities</span>
            <ProbabilityChart probs={event.probs} />
          </div>
          <div style={{ marginTop: '12px' }}>
            <span className="label">Raw Window Features (40×20)</span>
            <pre className="raw-data-dump">
              {JSON.stringify(event.window_features, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
function App() {
  const [telemetry, setTelemetry] = useState([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const fetchInitialData = async () => {
      const { data, error } = await supabase
        .from('radar_predictions')
        .select('*')
        .order('id', { ascending: false })
        .limit(50);

      if (data && data.length > 0) {
        setTelemetry(data);
      }
    };

    fetchInitialData();

    const channel = supabase
      .channel('schema-db-changes')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'radar_predictions' },
        (payload) => {
          setTelemetry((prev) => {
            const updated = [payload.new, ...prev];
            if (updated.length > 100) updated.pop();
            return updated;
          });
        }
      )
      .subscribe((status) => {
        setConnected(status === 'SUBSCRIBED');
      });

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  const latestEvent = telemetry[0];
  const history = telemetry.slice(1);
  const latestCfg = latestEvent ? getClassConfig(latestEvent.class_name) : null;
  const LatestIcon = latestCfg ? latestCfg.icon : Activity;

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <div className="header-logo">
            <Radar size={28} className="logo-icon" />
          </div>
          <div>
            <h1>Radar Activity Monitor</h1>
            <p className="subtitle">Deploying Multi-Class HF Space Model &bull; Real-time Streaming</p>
          </div>
        </div>
        <div className={`connection-badge ${connected ? 'conn-live' : 'conn-off'} hf-badge`}>
          {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
          {connected ? 'Live (HF Cloud)' : 'Offline'}
        </div>
      </header>

      {!latestEvent ? (
        <div className="empty-state">
          <Activity size={64} className="empty-icon" />
          <h2>Waiting for radar stream…</h2>
          <p>Start your RPi pipeline or simulate_sender.py</p>
        </div>
      ) : (
        <>
          {/* ─── CURRENT CAPTURE ────────────────────────────────────────── */}
          <section className="latest-event-box">
            <h2 className="section-title">CURRENT ACTIVITY</h2>
            <div
              className={`glass-panel featured-event featured-${latestCfg.category}`}
              style={{
                '--activity-color': latestCfg.color,
                '--activity-glow': latestCfg.bgGlow,
                '--activity-border': latestCfg.borderColor,
              }}
            >
              <div className="featured-header">
                <div className="featured-icon-wrapper">
                  <LatestIcon size={56} />
                </div>
                <div className="featured-title-area">
                  <div className="featured-title">{latestCfg.label}</div>
                  <div className="featured-class-badge" style={{ background: latestCfg.bgGlow, color: latestCfg.color }}>
                    {latestEvent.class_name}
                  </div>
                  <div className="featured-timestamp">
                    {new Date(latestEvent.timestamp).toLocaleString()} &nbsp;&bull;&nbsp; Frame #{latestEvent.frame_count}
                  </div>
                </div>
                <div className="featured-confidence">
                  <span className="label" style={{ color: 'rgba(255,255,255,0.6)' }}>Confidence</span>
                  <div className="conf-value" style={{ color: latestCfg.color }}>
                    {(latestEvent.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Probability Bars */}
              <div className="featured-probs">
                <span className="label">Class Probabilities</span>
                <ProbabilityChart probs={latestEvent.probs} />
              </div>

              {/* Metrics Row */}
              <div className="featured-metrics">
                <div className="metric">
                  <span className="label">Points</span>
                  <strong>{latestEvent.n_points}</strong>
                </div>
                <div className="metric">
                  <span className="label">Z-Mean</span>
                  <strong>{latestEvent.z_mean?.toFixed(3) || 'N/A'}</strong>
                </div>
                <div className="metric">
                  <span className="label">Height Range</span>
                  <strong>{latestEvent.height_range?.toFixed(3) || 'N/A'}</strong>
                </div>
                <div className="metric">
                  <span className="label">X-Mean</span>
                  <strong>{latestEvent.x_mean?.toFixed(3) || 'N/A'}</strong>
                </div>
              </div>
            </div>
          </section>

          {/* ─── ACTIVITY DISTRIBUTION ──────────────────────────────────── */}
          <section className="distribution-section">
            <h2 className="section-title">
              <BarChart3 size={16} style={{ verticalAlign: 'middle', marginRight: 8 }} />
              SESSION ACTIVITY BREAKDOWN
            </h2>
            <ActivityDistribution telemetry={telemetry} />
          </section>

          {/* ─── HISTORY ────────────────────────────────────────────────── */}
          <section className="history-section">
            <h2 className="section-title">
              INFERENCE HISTORY
              <span className="history-count">{history.length} events</span>
            </h2>
            <div className="history-list">
              {history.map((ev) => (
                <EventRow key={ev.id} event={ev} />
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default App;
