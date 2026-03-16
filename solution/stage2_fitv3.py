import json
import glob
import numpy as np
from scipy.optimize import minimize

# ── Load ALL 30,000 races ────────────────────────────────────
print("Loading all historical races...")
all_races = []
for filepath in sorted(glob.glob('data/historical_races/races_*.json')):
    with open(filepath) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} total races")

split       = int(len(all_races) * 0.8)
train_races = all_races[:split]
val_races   = all_races[split:]
print(f"Train: {len(train_races)}  |  Validate: {len(val_races)}")

# ── FORMULA ──────────────────────────────────────────────────
# Spec says explicitly:
#   - 4 components: base + compound + degradation + temperature
#   - Temperature "interacts with degradation behavior"
#   - Temperature "impacts how tires degrade" (not base lap time)
#   - Grace period exists, varies by compound
#   - Ordering from spec: soft_grace < med_grace < hard_grace
#                         soft_deg   > med_deg   > hard_deg

def lap_time(base, tire, age, temp, params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg,
     temp_mult) = params

    offsets = {'SOFT': soft_off, 'MEDIUM': 0.0, 'HARD': hard_off}
    graces  = {'SOFT': soft_grace, 'MEDIUM': med_grace, 'HARD': hard_grace}
    degs    = {'SOFT': soft_deg,   'MEDIUM': med_deg,   'HARD': hard_deg}

    laps_past_grace    = max(0.0, age - graces[tire])
    degradation_effect = laps_past_grace * degs[tire] * (1.0 + temp_mult * temp)

    return base + offsets[tire] + degradation_effect

def simulate(race, params):
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
            t   += lap_time(base, tire, age, temp, params)
            if lap in pits:
                t   += penalty
                tire = pits[lap]
                age  = 0

        times[did] = t
    return sorted(times, key=times.get)

# ── LOSS FUNCTION ────────────────────────────────────────────
SAMPLE = 5000

def loss(params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg,
     temp_mult) = params

    # Hard constraints — all derived directly from spec
    if soft_off >= 0 or hard_off <= 0:                return 1e9
    if temp_mult < 0:                                  return 1e9
    if not (0 < soft_grace < med_grace < hard_grace):  return 1e9
    if not (soft_deg > med_deg > hard_deg > 0):        return 1e9

    error = 0
    for race in train_races[:SAMPLE]:
        predicted = simulate(race, params)
        actual    = race['finishing_positions']
        for pred_pos, driver in enumerate(predicted):
            error += abs(pred_pos - actual.index(driver))
    return error

# ── INITIAL GUESS ────────────────────────────────────────────
x0 = np.array([
    -0.4,  0.8,        # soft_off, hard_off
     5.0, 10.0, 15.0,  # soft_grace, med_grace, hard_grace
     0.08, 0.04, 0.02, # soft_deg, med_deg, hard_deg
     0.001             # temp_mult
])

print(f"\nRunning optimizer on {SAMPLE} races...")
print("(This takes 2–5 minutes)\n")

call_count = [0]
def loss_with_progress(params):
    call_count[0] += 1
    if call_count[0] % 100 == 0:
        print(f"  iteration {call_count[0]}...", flush=True)
    return loss(params)

result = minimize(
    loss_with_progress,
    x0,
    method='Nelder-Mead',
    options={
        'maxiter': 10000,
        'xatol':   1e-6,
        'fatol':   0.5,
        'disp':    True
    }
)

best = result.x
print("\n✓ Optimization complete!")

labels = ['soft_off', 'hard_off',
          'soft_grace', 'med_grace', 'hard_grace',
          'soft_deg', 'med_deg', 'hard_deg',
          'temp_mult']

print(f"\nBEST PARAMETERS FOUND:")
for label, val in zip(labels, best):
    print(f"  {label:<12} = {val:.6f}")

# Sanity check — did constraints hold?
print(f"\nConstraint check:")
print(f"  soft_off < 0:                {best[0]:.4f} < 0  →  {best[0] < 0}")
print(f"  hard_off > 0:                {best[1]:.4f} > 0  →  {best[1] > 0}")
print(f"  grace order (s<m<h):         {best[2]:.2f} < {best[3]:.2f} < {best[4]:.2f}  →  {best[2] < best[3] < best[4]}")
print(f"  deg order   (s>m>h):         {best[5]:.4f} > {best[6]:.4f} > {best[7]:.4f}  →  {best[5] > best[6] > best[7]}")
print(f"  temp_mult >= 0:              {best[8]:.6f}  →  {best[8] >= 0}")

# ── ACCURACY ─────────────────────────────────────────────────
print("\nChecking accuracy on 1000 training races...")
correct_train = sum(
    1 for r in train_races[:1000]
    if simulate(r, best) == r['finishing_positions']
)
print(f"  Train accuracy: {correct_train}/1000 = {correct_train/10:.1f}%")

print("\nChecking accuracy on 1000 validation races (never seen)...")
correct_val = sum(
    1 for r in val_races[:1000]
    if simulate(r, best) == r['finishing_positions']
)
print(f"  Val accuracy:   {correct_val}/1000 = {correct_val/10:.1f}%")

# ── ALSO CHECK: how close are we? position error breakdown ───
print("\nPosition error breakdown (1000 val races):")
pos_errors = []
for race in val_races[:1000]:
    predicted = simulate(race, best)
    actual    = race['finishing_positions']
    for pred_pos, driver in enumerate(predicted):
        pos_errors.append(abs(pred_pos - actual.index(driver)))

errors = np.array(pos_errors)
print(f"  Exactly correct:   {(errors==0).mean()*100:.1f}% of drivers")
print(f"  Off by ≤1 pos:     {(errors<=1).mean()*100:.1f}% of drivers")
print(f"  Off by ≤2 pos:     {(errors<=2).mean()*100:.1f}% of drivers")
print(f"  Mean error:        {errors.mean():.3f} positions")

# ── COPY-PASTE BLOCK ─────────────────────────────────────────
print("\n" + "="*60)
print("PASTE THIS INTO race_simulator.py:")
print("="*60)
print("PARAMS = [")
for label, val in zip(labels, best):
    print(f"    {val:.8f},  # {label}")
print("]")