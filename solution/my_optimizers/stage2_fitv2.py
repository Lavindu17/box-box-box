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

# 80% train, 20% validate — NEVER mix these
split       = int(len(all_races) * 0.8)
train_races = all_races[:split]
val_races   = all_races[split:]
print(f"Train: {len(train_races)}  |  Validate: {len(val_races)}")

# ── THE FORMULA ──────────────────────────────────────────────
# Based on regulations re-read:
#
# lap_time = base
#          + compound_effect        (flat offset, MEDIUM = 0 reference)
#          + degradation_effect     (ZERO until grace period ends)
#          + temperature_effect     (separate additive term)
#
# Grace period = laps of consistent performance before deg kicks in
# Varies by compound: SOFT shortest, HARD longest
#
# params:
#   [0] soft_off    — SOFT speed advantage (negative)
#   [1] hard_off    — HARD speed penalty (positive)
#   [2] soft_grace  — laps before SOFT degrades
#   [3] med_grace   — laps before MEDIUM degrades
#   [4] hard_grace  — laps before HARD degrades
#   [5] soft_deg    — SOFT degradation rate per lap past grace
#   [6] med_deg     — MEDIUM degradation rate per lap past grace
#   [7] hard_deg    — HARD degradation rate per lap past grace
#   [8] temp_effect — additive temperature contribution per °C

def lap_time(base, tire, age, temp, params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg,
     temp_effect) = params

    offsets = {'SOFT': soft_off, 'MEDIUM': 0.0, 'HARD': hard_off}
    graces  = {'SOFT': soft_grace, 'MEDIUM': med_grace, 'HARD': hard_grace}
    degs    = {'SOFT': soft_deg,   'MEDIUM': med_deg,   'HARD': hard_deg}

    laps_past_grace    = max(0.0, age - graces[tire])
    degradation_effect = laps_past_grace * degs[tire]
    temperature_effect = temp_effect * temp

    return base + offsets[tire] + degradation_effect + temperature_effect

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
# Total position error across all drivers across all sampled races.
# Smooth signal — scipy can see "getting warmer" even before
# exact matches appear.

SAMPLE = 5000

def loss(params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg,
     temp_effect) = params

    # Enforce physical constraints manually
    # (Nelder-Mead ignores the bounds parameter)
    if soft_off > 0 or hard_off < 0:
        return 1e9
    if any(x < 0 for x in [soft_grace, med_grace, hard_grace,
                             soft_deg, med_deg, hard_deg, temp_effect]):
        return 1e9
    # Grace period ordering: SOFT <= MEDIUM <= HARD
    if not (soft_grace <= med_grace <= hard_grace):
        return 1e9

    error = 0
    for race in train_races[:SAMPLE]:
        predicted = simulate(race, params)
        actual    = race['finishing_positions']
        for pred_pos, driver in enumerate(predicted):
            error += abs(pred_pos - actual.index(driver))
    return error

# ── INITIAL GUESS ────────────────────────────────────────────
# Calibrated from Stage 1 findings:
#   - compound spread is small (~3%) → small offsets
#   - temperature effect is real but modest → small temp_effect
#   - grace periods: SOFT short, HARD long (from regulations)
#
# [soft_off, hard_off,
#  soft_grace, med_grace, hard_grace,
#  soft_deg, med_deg, hard_deg,
#  temp_effect]
x0 = np.array([
    -0.4,  0.8,       # compound offsets
     5.0, 10.0, 15.0, # grace periods (laps)
     0.08, 0.04, 0.02, # degradation rates
     0.01              # temperature additive
])

print(f"\nRunning optimizer on {SAMPLE} races...")
print("(This takes 2–5 minutes)\n")

call_count = [0]
def loss_with_progress(params):
    call_count[0] += 1
    if call_count[0] % 50 == 0:
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
print(f"\nBEST PARAMETERS FOUND:")
print(f"  soft_offset  = {best[0]:.6f}")
print(f"  hard_offset  = {best[1]:.6f}")
print(f"  soft_grace   = {best[2]:.4f} laps")
print(f"  med_grace    = {best[3]:.4f} laps")
print(f"  hard_grace   = {best[4]:.4f} laps")
print(f"  soft_deg     = {best[5]:.6f} s/lap")
print(f"  med_deg      = {best[6]:.6f} s/lap")
print(f"  hard_deg     = {best[7]:.6f} s/lap")
print(f"  temp_effect  = {best[8]:.8f} s/°C")

# ── ACCURACY CHECKS ──────────────────────────────────────────
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

# ── COPY-PASTE BLOCK ─────────────────────────────────────────
print("\n" + "="*60)
print("PASTE THIS INTO stage3_validate.py AND race_simulator.py:")
print("="*60)
print(f"PARAMS = [")
print(f"    {best[0]:.6f},  # soft_offset")
print(f"    {best[1]:.6f},  # hard_offset")
print(f"    {best[2]:.6f},  # soft_grace (laps)")
print(f"    {best[3]:.6f},  # med_grace  (laps)")
print(f"    {best[4]:.6f},  # hard_grace (laps)")
print(f"    {best[5]:.6f},  # soft_deg   (s/lap past grace)")
print(f"    {best[6]:.6f},  # med_deg    (s/lap past grace)")
print(f"    {best[7]:.6f},  # hard_deg   (s/lap past grace)")
print(f"    {best[8]:.8f},  # temp_effect (s/°C)")
print(f"]")