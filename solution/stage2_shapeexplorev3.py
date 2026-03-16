import json
import glob
import numpy as np
from scipy.optimize import minimize

print("Loading races...")
all_races = []
for filepath in sorted(glob.glob('data/historical_races/races_*.json')):
    with open(filepath) as f:
        all_races.extend(json.load(f))

train_races = all_races[:int(len(all_races) * 0.8)]
print(f"Loaded {len(all_races)} races, using {len(train_races)} for training")

# ── DEFINE ALL CANDIDATE SHAPES ──────────────────────────────

def simulate_with_shape(race, params, shape):
    cfg     = race['race_config']
    base    = cfg['base_lap_time']
    penalty = cfg['pit_lane_time']
    laps    = cfg['total_laps']
    temp    = cfg['track_temp']
    times   = {}

    for strat in race['strategies'].values():
        did  = strat['driver_id']
        tire = strat['starting_tire']
        age  = 0
        t    = 0.0
        pits = {s['lap']: s['to_tire'] for s in strat['pit_stops']}

        for lap in range(1, laps + 1):
            age += 1
            t   += shape(base, tire, age, temp, params)
            if lap in pits:
                t   += penalty
                tire = pits[lap]
                age  = 0
        times[did] = t
    return sorted(times, key=times.get)

# Shape A: current — temp multiplies degradation
# params: [soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_mult]
def shape_A(base, tire, age, temp, p):
    offsets   = {'SOFT': p[0], 'MEDIUM': 0.0, 'HARD': p[1]}
    deg_rates = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    return base + offsets[tire] + age * deg_rates[tire] * (1 + p[5] * temp)

# Shape B: temp adds independently to degradation
# params: [soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_add]
def shape_B(base, tire, age, temp, p):
    offsets   = {'SOFT': p[0], 'MEDIUM': 0.0, 'HARD': p[1]}
    deg_rates = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    return base + offsets[tire] + age * deg_rates[tire] + p[5] * temp

# Shape C: no temperature effect at all
# params: [soft_off, hard_off, soft_deg, med_deg, hard_deg]
def shape_C(base, tire, age, temp, p):
    offsets   = {'SOFT': p[0], 'MEDIUM': 0.0, 'HARD': p[1]}
    deg_rates = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    return base + offsets[tire] + age * deg_rates[tire]

# Shape D: quadratic degradation, no temperature
# params: [soft_off, hard_off, soft_deg, med_deg, hard_deg]
def shape_D(base, tire, age, temp, p):
    offsets   = {'SOFT': p[0], 'MEDIUM': 0.0, 'HARD': p[1]}
    deg_rates = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    return base + offsets[tire] + (age ** 2) * deg_rates[tire]

# Shape E: temperature multiplies the entire delta (offset + degradation)
# params: [soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_mult]
def shape_E(base, tire, age, temp, p):
    offsets   = {'SOFT': p[0], 'MEDIUM': 0.0, 'HARD': p[1]}
    deg_rates = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    delta = offsets[tire] + age * deg_rates[tire]
    return base + delta * (1 + p[5] * temp)

# Shape F: quadratic degradation WITH temperature multiplier
# params: [soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_mult]
def shape_F(base, tire, age, temp, p):
    offsets   = {'SOFT': p[0], 'MEDIUM': 0.0, 'HARD': p[1]}
    deg_rates = {'SOFT': p[2], 'MEDIUM': p[3], 'HARD': p[4]}
    return base + offsets[tire] + (age ** 2) * deg_rates[tire] * (1 + p[5] * temp)

# ── FIT AND TEST EACH SHAPE ───────────────────────────────────
SAMPLE = 2000  # smaller sample for speed — just comparing shapes

shapes = {
    'A: linear deg, temp × deg':          (shape_A, [-0.4, 0.8, 0.07, 0.04, 0.02, 0.0003]),
    'B: linear deg, temp + deg':           (shape_B, [-0.4, 0.8, 0.07, 0.04, 0.02, 0.001]),
    'C: linear deg, no temp':              (shape_C, [-0.4, 0.8, 0.07, 0.04, 0.02]),
    'D: quadratic deg, no temp':           (shape_D, [-0.4, 0.8, 0.003, 0.001, 0.0005]),
    'E: linear deg, temp × whole delta':   (shape_E, [-0.4, 0.8, 0.07, 0.04, 0.02, 0.0003]),
    'F: quadratic deg, temp × deg':        (shape_F, [-0.4, 0.8, 0.003, 0.001, 0.0005, 0.0003]),
}

results = {}

for name, (shape_fn, x0) in shapes.items():
    print(f"\nFitting shape: {name}")

    def make_loss(shape_fn):
        def loss(params):
            # Basic bounds enforcement
            if params[0] > 0 or params[1] < 0:
                return 1e9
            if any(params[i] < 0 for i in range(2, len(params))):
                return 1e9
            error = 0
            for race in train_races[:SAMPLE]:
                predicted = simulate_with_shape(race, params, shape_fn)
                actual    = race['finishing_positions']
                for pred_pos, driver in enumerate(predicted):
                    error += abs(pred_pos - actual.index(driver))
            return error
        return loss

    res = minimize(
        make_loss(shape_fn),
        np.array(x0, dtype=float),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 10.0}
    )

    best_p = res.x

    # Measure exact-match accuracy on 500 held-out races
    correct = sum(
        1 for r in train_races[SAMPLE:SAMPLE+500]
        if simulate_with_shape(r, best_p, shape_fn) == r['finishing_positions']
    )
    accuracy = correct / 500 * 100
    results[name] = (accuracy, best_p, res.fun)
    print(f"  Loss: {res.fun:.0f}  |  Accuracy: {accuracy:.1f}%  |  Params: {np.round(best_p, 4)}")

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "="*60)
print("SHAPE COMPARISON SUMMARY")
print("="*60)
for name, (acc, params, loss_val) in sorted(results.items(),
                                             key=lambda x: -x[1][0]):
    print(f"  {acc:5.1f}%  |  {name}")

winner = max(results.items(), key=lambda x: x[1][0])
print(f"\nWINNING SHAPE: {winner[0]}")
print(f"Accuracy: {winner[1][0]:.1f}%")
print(f"Params:   {np.round(winner[1][1], 6)}")