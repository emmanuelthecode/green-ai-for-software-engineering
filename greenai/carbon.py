"""Carbon-aware scheduling (Section 4.9).

The idea: the same kWh emits a different amount of CO2 depending on when
(and where) it's used, because the grid mix changes through the day. For
deferrable jobs like nightly retraining or batch repository analysis, you can
just pick the cleanest window.

`EXAMPLE_PROFILE` is an illustrative 24-hour curve shaped like a typical UK
day (cleaner overnight and around midday solar, dirtier at the evening
peak). It's for demonstration only. For real numbers, plug in a forecast
from your grid operator, e.g. the GB Carbon Intensity API, with
`load_profile()`.
"""

import csv

import numpy as np

# gCO2e/kWh for hours 00:00 .. 23:00 (illustrative)
EXAMPLE_PROFILE = np.array([
    150, 140, 130, 125, 125, 135, 165, 200, 215, 205, 185, 165,
    155, 150, 160, 180, 220, 255, 265, 250, 225, 200, 180, 165,
], dtype=float)


def load_profile(path):
    """CSV with columns hour,intensity (24 rows)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return np.array([float(r["intensity"]) for r in sorted(rows, key=lambda r: int(r["hour"]))])


def window_intensity(profile, start, hours):
    idx = [(start + h) % len(profile) for h in range(hours)]
    return float(profile[idx].mean())


def best_window(profile, hours):
    """Start hour with the lowest average intensity for a job of `hours`."""
    scores = [window_intensity(profile, s, hours) for s in range(len(profile))]
    start = int(np.argmin(scores))
    return start, scores[start]


def compare(energy_kwh, profile=EXAMPLE_PROFILE, hours=2, naive_start=17):
    """Emissions if the job runs at `naive_start` vs in the best window."""
    naive = window_intensity(profile, naive_start, hours)
    start, best = best_window(profile, hours)
    return {
        "naive_start": naive_start,
        "naive_g": energy_kwh * naive,
        "best_start": start,
        "best_g": energy_kwh * best,
        "saving_pct": 100 * (1 - best / naive),
    }
