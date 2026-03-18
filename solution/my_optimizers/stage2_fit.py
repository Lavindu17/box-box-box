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
def lap_time(base, tire, tire_age, track_temp, params):
    soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_mult = params
    offsets   = {'SOFT': soft_off, 'MEDIUM': 0.0, 'HARD': hard_off}
    deg_rates = {'SOFT': soft_deg, 'MEDIUM': med_deg, 'HARD': hard_deg}
    return (base
            + offsets[tire]
            + tire_age * deg_rates[tire] * (1.0 + temp_mult * track_temp))

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
    # Enforce bounds manually — Nelder-Mead ignores bounds parameter
    soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_mult = params
    if soft_off > 0 or hard_off < 0:
        return 1e9
    if any(d < 0 for d in [soft_deg, med_deg, hard_deg, temp_mult]):
        return 1e9

    error = 0
    for race in train_races[:SAMPLE]:
        predicted = simulate(race, params)
        actual    = race['finishing_positions']
        for pred_pos, driver in enumerate(predicted):
            actual_pos = actual.index(driver)
            error += abs(pred_pos - actual_pos)
    return error

# ── INITIAL GUESS — calibrated from Stage 1 findings ────────
# Stage 1 showed compound spread is small (~3%)
# so offsets are small, not the aggressive -1.0 / +1.5
# [soft_off, hard_off, soft_deg, med_deg, hard_deg, temp_mult]
x0 = np.array([-0.4, 0.8, 0.07, 0.04, 0.02, 0.0003])

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
        'maxiter': 5000,
        'xatol':   1e-5,
        'fatol':   1.0,
        'disp':    True
    }
)

best = result.x
print("\n✓ Optimization complete!")
print(f"\nBEST PARAMETERS FOUND:")
print(f"  soft_offset  = {best[0]:.6f}")
print(f"  hard_offset  = {best[1]:.6f}")
print(f"  soft_deg     = {best[2]:.6f}")
print(f"  med_deg      = {best[3]:.6f}")
print(f"  hard_deg     = {best[4]:.6f}")
print(f"  temp_mult    = {best[5]:.8f}")

# ── ACCURACY CHECK ───────────────────────────────────────────
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
print(f"  Val accuracy: {correct_val}/1000 = {correct_val/10:.1f}%")

# ── COPY-PASTE BLOCK ─────────────────────────────────────────
print("\n" + "="*55)
print("PASTE THIS INTO stage3_validate.py AND race_simulator.py:")
print("="*55)
print(f"PARAMS = [{best[0]:.6f}, {best[1]:.6f}, {best[2]:.6f},")
print(f"          {best[3]:.6f}, {best[4]:.6f}, {best[5]:.8f}]")