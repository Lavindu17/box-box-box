import json
import glob
import numpy as np
from scipy.optimize import minimize

print("Loading races...")
all_races = []
for filepath in sorted(glob.glob('data/historical_races/races_*.json'))[:1]:
    with open(filepath) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} races for logic testing")

train_races = all_races[:800]
test_races  = all_races[800:]

# ── FOUR LOGIC VARIANTS ──────────────────────────────────────
# We test every combination of two binary decisions:
#
# Decision 1 — Tire age start:
#   START_AT_1: fresh tire first lap = age 1  (our current)
#   START_AT_0: fresh tire first lap = age 0
#
# Decision 2 — Pit lap timing:
#   PIT_AFTER:  drive lap on old tire, THEN pit  (our current)
#   PIT_BEFORE: pit first, drive lap on NEW tire

def simulate_variant(race, params, start_at_1, pit_after):
    cfg     = race['race_config']
    base    = cfg['base_lap_time']
    penalty = cfg['pit_lane_time']
    laps    = cfg['total_laps']
    temp    = cfg['track_temp']

    soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_mult = params
    offsets   = {'SOFT': soft_off, 'MEDIUM': 0.0, 'HARD': hard_off}
    deg_rates = {'SOFT': soft_deg, 'MEDIUM': med_deg, 'HARD': hard_deg}

    def lt(tire, age):
        return (base
                + offsets[tire]
                + (age ** 2) * deg_rates[tire] * (1 + temp_mult * temp))

    times = {}
    for strat in race['strategies'].values():
        did  = strat['driver_id']
        tire = strat['starting_tire']
        age  = 0 if not start_at_1 else -1  # -1 so first += gives 0 or 1
        t    = 0.0
        pits = {s['lap']: s['to_tire'] for s in strat['pit_stops']}

        for lap in range(1, laps + 1):
            if pit_after:
                # OLD tire for this lap, then pit
                age += 1
                t   += lt(tire, age)
                if lap in pits:
                    t   += penalty
                    tire = pits[lap]
                    age  = 0 if not start_at_1 else -1
            else:
                # Pit FIRST, then drive lap on new tire
                if lap in pits:
                    t   += penalty
                    tire = pits[lap]
                    age  = 0 if not start_at_1 else -1
                age += 1
                t   += lt(tire, age)

        times[did] = t
    return sorted(times, key=times.get)

# ── FIT AND SCORE ALL FOUR VARIANTS ─────────────────────────
SAMPLE = 1000

variants = {
    'V1: age starts at 1, pit AFTER lap  (current)': (True,  True),
    'V2: age starts at 0, pit AFTER lap':             (False, True),
    'V3: age starts at 1, pit BEFORE lap':            (True,  False),
    'V4: age starts at 0, pit BEFORE lap':            (False, False),
}

# Use best params from shape test as starting point
x0 = np.array([-0.35, 0.49, 0.0099, 0.0028, 0.0009, 0.000001])

results = {}

for name, (start_at_1, pit_after) in variants.items():
    print(f"\nTesting: {name}")

    def make_loss(s1, pa):
        def loss(params):
            if params[0] > 0 or params[1] < 0:
                return 1e9
            if any(params[i] < 0 for i in range(2, len(params))):
                return 1e9
            error = 0
            for race in train_races[:SAMPLE]:
                predicted = simulate_variant(race, params, s1, pa)
                actual    = race['finishing_positions']
                for pred_pos, driver in enumerate(predicted):
                    error += abs(pred_pos - actual.index(driver))
            return error
        return loss

    res = minimize(
        make_loss(start_at_1, pit_after),
        x0.copy(),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1.0}
    )

    best_p = res.x
    correct = sum(
        1 for r in test_races[:200]
        if simulate_variant(r, best_p, start_at_1, pit_after)
           == r['finishing_positions']
    )
    acc = correct / 200 * 100
    results[name] = (acc, best_p, res.fun)
    print(f"  Loss: {res.fun:.0f}  |  Accuracy: {acc:.1f}%")
    print(f"  Params: {np.round(best_p, 5)}")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "="*60)
print("LOGIC VARIANT SUMMARY")
print("="*60)
for name, (acc, params, lv) in sorted(results.items(),
                                       key=lambda x: -x[1][0]):
    print(f"  {acc:5.1f}%  |  {name}")

winner = max(results.items(), key=lambda x: x[1][0])
print(f"\nWINNING VARIANT: {winner[0]}")
print(f"Accuracy: {winner[1][0]:.1f}%")
print(f"Params:   {np.round(winner[1][1], 6)}")