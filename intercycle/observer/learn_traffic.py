"""Learn hour-of-day traffic per zone from occupancy counts. Never imports ground truth (tested)."""
from __future__ import annotations

import numpy as np

DAY = 86_400.0


def hourly_means(t: np.ndarray, counts: np.ndarray, bin_s: float | None = None) -> np.ndarray:
    """Mean occupancy per zone per hour-of-day, pooled over all days present: [zone][24] (NaN if an hour has no bins)."""
    t, counts = np.asarray(t, float), np.asarray(counts, float)
    bin_s = bin_s or (float(np.median(np.diff(t))) if len(t) > 1 else 300.0)
    hour = (((t + bin_s / 2) % DAY) // 3600).astype(int)
    out = np.full((counts.shape[0], 24), np.nan)
    for h in range(24):
        sel = hour == h
        if sel.any():
            out[:, h] = counts[:, sel].mean(axis=1)
    return out


def learn_traffic(t: np.ndarray, counts: np.ndarray, bin_s: float | None = None) -> list[list[int]]:
    """Occupancy bins (bin start times, counts[zone][bin]; one or many days) -> levels [zone][24] in 0..15,
    each zone scaled to its own busiest hour. Hours never observed get the zone's mean level."""
    m = hourly_means(t, counts, bin_s)
    out = []
    for row in m:
        top = np.nanmax(row) if np.isfinite(row).any() else 0.0
        lv = np.floor(15 * row / top + 0.5) if top > 0 else np.zeros(24)
        fill = np.nanmean(lv) if np.isfinite(lv).any() else 0.0
        out.append([int(np.clip(round(float(np.nan_to_num(v, nan=fill))), 0, 15)) for v in lv])
    return out
