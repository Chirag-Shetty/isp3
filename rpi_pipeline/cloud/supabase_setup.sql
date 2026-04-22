-- ============================================================
--  supabase_setup.sql
--  Run this in: Supabase Dashboard → SQL Editor → New query
-- ============================================================

-- ── 1. Main predictions table ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS radar_predictions (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Timing
    timestamp        TIMESTAMPTZ,
    session_start    TEXT,
    frame_count      INTEGER,
    window_index     INTEGER,

    -- Point cloud summary
    n_points         INTEGER,
    x_mean           REAL,
    y_mean           REAL,
    z_mean           REAL,
    height_range     REAL,

    -- ML inference result
    class_id         SMALLINT,
    class_name       TEXT,
    confidence       REAL,
    is_fall          BOOLEAN,
    probs            JSONB,        -- array of 6 probabilities

    -- Full window for re-analysis (40 × 12 matrix)
    window_features  JSONB
);

-- ── 2. Index for fast time-range queries ─────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_radar_predictions_timestamp
    ON radar_predictions (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_radar_predictions_is_fall
    ON radar_predictions (is_fall)
    WHERE is_fall = TRUE;

-- ── 3. Enable Row-Level Security (keeps data safe with anon key) ─────────────
ALTER TABLE radar_predictions ENABLE ROW LEVEL SECURITY;

-- Allow the anon key to INSERT rows (RPi writes)
CREATE POLICY "anon_insert" ON radar_predictions
    FOR INSERT
    TO anon
    WITH CHECK (TRUE);

-- Allow the anon key to SELECT rows (dashboard / mobile app reads)
CREATE POLICY "anon_select" ON radar_predictions
    FOR SELECT
    TO anon
    USING (TRUE);

-- ── 4. Enable Realtime for live alerts in the mobile app ─────────────────────
-- Run this to broadcast new rows to WebSocket subscribers:
ALTER PUBLICATION supabase_realtime ADD TABLE radar_predictions;


-- ── 5. Convenience view: fall events only ────────────────────────────────────
CREATE OR REPLACE VIEW fall_events AS
SELECT
    id,
    timestamp,
    session_start,
    class_name,
    confidence,
    n_points,
    z_mean,
    height_range
FROM radar_predictions
WHERE is_fall = TRUE
ORDER BY timestamp DESC;


-- ── 6. Verification query (run after inserting a test row) ───────────────────
-- SELECT * FROM radar_predictions ORDER BY created_at DESC LIMIT 5;
-- SELECT COUNT(*) FROM radar_predictions WHERE is_fall = TRUE;
