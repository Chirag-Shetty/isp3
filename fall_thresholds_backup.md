# Fall Detection Thresholds — Backup (pre person_down change)
Date: 2026-07-02

## Constants (lines 56–62 of radar-api-main.py)

| Constant               | Value  | Meaning                                                        |
|------------------------|--------|----------------------------------------------------------------|
| `Z_DROP_THRESHOLD`     | 0.90   | z must drop ≥ 90 cm from P90 peak to count as height drop     |
| `BODY_FLAT_THRESH`     | 0.40   | height_range < 0.40 m → body is horizontal                    |
| `LOW_POINTS_THRESH`    | 8      | Upper bound on radar points (too many = still standing)        |
| `MIN_POINTS_VALID`     | 3      | Lower bound on radar points (< 3 = z_mean unreliable, skip)   |
| `DETECTION_WINDOW`     | 20     | Rolling look-back window in frames                             |
| `SUSTAINED_DROP_FRAMES`| 3      | Drop must persist ≥ 3 consecutive frames before triggering     |
| `COOLDOWN_FRAMES`      | 150    | ~8s at 18fps — no re-trigger while on cooldown                 |

## Fall trigger logic (all 3 must be True simultaneously + cooldown == 0)

```python
height_dropped = drop_streak >= SUSTAINED_DROP_FRAMES   # 3
body_flat      = hrng <= BODY_FLAT_THRESH                # 0.40 m
few_points     = MIN_POINTS_VALID <= npts <= LOW_POINTS_THRESH  # 3–8

is_fall = height_dropped and body_flat and few_points and self.cooldown == 0
```

## z_peak and z_current computation

```python
z_peak    = float(np.percentile(window_z, 90))    # 90th percentile of last 20 frames
z_current = float(np.mean(window_z[-3:]))          # mean of last 3 frames
z_drop    = z_peak - z_current
```

## What changed (person_down feature)

Added `RECOVERY_Z_THRESHOLD = 1.0` and `self.person_down` flag.
- `person_down` is set True on fall trigger
- Cleared when `z_current >= 1.0 m` (person stood back up)
- DynamoDB write only on initial trigger (`if is_fall and not self.person_down`)
- Returned `is_fall` = `momentary_fall OR person_down`
- No frontend changes required

## To revert

Copy `radar-api-main.backup.py` back to `radar-api-main.py` and redeploy.
