import json
import glob
import numpy as np

COMPOUND_IDX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

# Stage D params
BASE_PARAMS = [-1.95049364, 1.32673000, 9.05720595, 18.40432358,
               27.94283633, 0.86983298, 0.34446878, 0.14992882,
               1.50198125,  0.03303927, 0.01560123, 0.01089012]

print("Loading races...")
all_races = []
for fp in sorted(glob.glob('data/historical_races/races_*.json'))[:10]:
    with open(fp) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} races\n")

def simulate(race, params):
    cfg     = race['race_config']
    base    = cfg['base_lap_time']
    penalty = cfg['pit_lane_time']
    laps    = cfg['total_laps']
    temp    = cfg['track_temp']
    offsets   = [params[0], 0.0,        params[1]]
    graces    = [params[2], params[3],  params[4]]
    degs      = [params[5], params[6],  params[7]]
    temp_sens = [params[9], params[10], params[11]]
    deg_exp   = params[8]
    times = {}
    for strat in race['strategies'].values():
        did  = strat['driver_id']
        tire = COMPOUND_IDX[strat['starting_tire']]
        age  = 0
        t    = 0.0
        pits = {s['lap']: COMPOUND_IDX[s['to_tire']]
                for s in strat['pit_stops']}
        for lap in range(1, laps + 1):
            age += 1
            lpg  = age - graces[tire]
            if lpg < 0.0: lpg = 0.0
            t += (base + offsets[tire]
                  + (lpg ** deg_exp) * degs[tire]
                  + temp_sens[tire] * temp)
            if lap in pits:
                t   += penalty
                tire = pits[lap]
                age  = 0
        times[did] = t
    return sorted(times, key=times.get)

def score(params, races, n=1000):
    return sum(
        1 for r in races[:n]
        if simulate(r, params) == r['finishing_positions']
    )

baseline = score(BASE_PARAMS, all_races)
print(f"Baseline (Stage D): {baseline}/1000 = {baseline/10:.1f}%\n")

# ── TARGETED FIX: test HARD-specific params ──────────────────
# Failure pattern shows HARD long stints are overpredicted
# Test: shorter grace period, larger initial penalty, higher deg rate

print("Testing HARD grace period (currently 27.94):")
for hard_grace in [15, 17, 18, 19, 20, 22, 24, 25, 26, 27, 28]:
    p = BASE_PARAMS[:]
    p[4] = hard_grace
    s = score(p, all_races)
    marker = " ← BETTER" if s > baseline else ""
    print(f"  hard_grace={hard_grace:4.0f}  →  {s}/1000 = {s/10:.1f}%{marker}")

print("\nTesting HARD initial offset (currently 1.327):")
for hard_off in [1.0, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.5]:
    p = BASE_PARAMS[:]
    p[1] = hard_off
    s = score(p, all_races)
    marker = " ← BETTER" if s > baseline else ""
    print(f"  hard_off={hard_off:.1f}  →  {s}/1000 = {s/10:.1f}%{marker}")

print("\nTesting HARD degradation rate (currently 0.149):")
for hard_deg in [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
    p = BASE_PARAMS[:]
    p[7] = hard_deg
    s = score(p, all_races)
    marker = " ← BETTER" if s > baseline else ""
    print(f"  hard_deg={hard_deg:.2f}  →  {s}/1000 = {s/10:.1f}%{marker}")

# ── COMBINE: best values from each test ──────────────────────
print("\nTesting combinations of best HARD values found above...")
best_score  = baseline
best_params = BASE_PARAMS[:]

hard_graces = [18, 19, 20, 22, 24, 25]
hard_offs   = [1.5, 1.6, 1.7, 1.8, 2.0]
hard_degs   = [0.15, 0.18, 0.20, 0.25]

for hg, ho, hd in [(hg, ho, hd)
                    for hg in hard_graces
                    for ho in hard_offs
                    for hd in hard_degs]:
    p = BASE_PARAMS[:]
    p[1] = ho
    p[4] = hg
    p[7] = hd
    s = score(p, all_races)
    if s > best_score:
        best_score  = s
        best_params = p[:]
        print(f"  NEW BEST: hard_grace={hg} hard_off={ho} hard_deg={hd}"
              f"  →  {s}/1000 = {s/10:.1f}%")

labels = ['soft_off','hard_off','soft_grace','med_grace','hard_grace',
          'soft_deg','med_deg','hard_deg','deg_exp',
          'temp_soft','temp_med','temp_hard']

print(f"\nBest score: {best_score}/1000 = {best_score/10:.1f}%")
print(f"Improvement: +{best_score - baseline} races")
print("\nPASTE INTO race_simulator.py:")
print("PARAMS = [")
for label, val in zip(labels, best_params):
    print(f"    {val},  # {label}")
print("]")