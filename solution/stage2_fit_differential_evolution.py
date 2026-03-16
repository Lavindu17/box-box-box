import json
import glob
import numpy as np
from scipy.optimize import differential_evolution

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

def lap_time(base, tire, age, temp, params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg,
     deg_exp,
     temp_soft, temp_med, temp_hard) = params

    offsets   = {'SOFT': soft_off,  'MEDIUM': 0.0,      'HARD': hard_off}
    graces    = {'SOFT': soft_grace,'MEDIUM': med_grace, 'HARD': hard_grace}
    degs      = {'SOFT': soft_deg,  'MEDIUM': med_deg,   'HARD': hard_deg}
    temp_sens = {'SOFT': temp_soft, 'MEDIUM': temp_med,  'HARD': temp_hard}

    laps_past_grace    = max(0.0, age - graces[tire])
    degradation_effect = (laps_past_grace ** deg_exp) * degs[tire]
    temperature_effect = temp_sens[tire] * temp

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

# ── LOSS — no manual constraint checks needed ────────────────
# differential_evolution enforces bounds strictly by design
# We still add ordering constraints as a penalty

SAMPLE = 500  # start small — DE is slower per iteration

def loss(params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg,
     deg_exp,
     temp_soft, temp_med, temp_hard) = params

    # Ordering constraints — DE handles bounds but not ordering
    if not (soft_grace < med_grace < hard_grace): return 1e9
    if not (soft_deg > med_deg > hard_deg):       return 1e9
    if not (temp_soft >= temp_med >= temp_hard):  return 1e9

    error = 0
    for race in train_races[:SAMPLE]:
        predicted = simulate(race, params)
        actual    = race['finishing_positions']
        for pred_pos, driver in enumerate(predicted):
            error += abs(pred_pos - actual.index(driver))
    return error

# ── BOUNDS — DE enforces these strictly ─────────────────────
# Every parameter gets a min and max
# Derived from spec + Stage 1 findings
#
#                    min      max
bounds = [
    (-3.0,  -0.1),   # soft_off:    must be negative
    ( 0.1,   3.0),   # hard_off:    must be positive
    ( 1.0,  10.0),   # soft_grace:  shortest
    ( 5.0,  20.0),   # med_grace:   middle
    (10.0,  40.0),   # hard_grace:  longest
    ( 0.01,  0.5),   # soft_deg:    fastest degradation
    ( 0.005, 0.3),   # med_deg:     medium
    ( 0.001, 0.1),   # hard_deg:    slowest
    ( 0.5,   3.0),   # deg_exp:     curve shape
    ( 0.0,   0.05),  # temp_soft:   most sensitive
    ( 0.0,   0.05),  # temp_med
    ( 0.0,   0.05),  # temp_hard:   least sensitive
]

labels = ['soft_off','hard_off','soft_grace','med_grace','hard_grace',
          'soft_deg','med_deg','hard_deg','deg_exp',
          'temp_soft','temp_med','temp_hard']

call_count = [0]
def loss_with_progress(params):
    call_count[0] += 1
    if call_count[0] % 100 == 0:
        # Show current best
        print(f"  eval {call_count[0]}...", flush=True)
    return loss(params)

# ── PROGRESSIVE STAGES ───────────────────────────────────────
stages = [
    (200,  50,  "A — shape check   (fast, ~2 min)"),
    (1000, 100, "B — refinement    (~10 min)"),
    (5000, 200, "C — precision     (~40 min)"),
]

current_best = None

for sample_size, popsize_mult, stage_name in stages:
    # Update sample size in loss
    SAMPLE = sample_size

    print(f"\n{'='*55}")
    print(f"Stage {stage_name}")
    print(f"Sample: {sample_size} races  |  Population: {popsize_mult * len(bounds)}")
    print(f"{'='*55}")

    result = differential_evolution(
        loss_with_progress,
        bounds,
        maxiter=popsize_mult,
        popsize=15,          # 15 × 12 params = 180 candidates per generation
        tol=0.001,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,             # reproducible
        disp=True,
        polish=True,         # runs Nelder-Mead locally at the end to refine
        init='latinhypercube'
    )

    best = result.x
    current_best = best

    # Quick accuracy check
    correct_train = sum(
        1 for r in train_races[:200]
        if simulate(r, best) == r['finishing_positions']
    )
    correct_val = sum(
        1 for r in val_races[:200]
        if simulate(r, best) == r['finishing_positions']
    )

    train_acc = correct_train / 2
    val_acc   = correct_val   / 2

    print(f"\nStage {stage_name[0]} result:")
    for label, val in zip(labels, best):
        print(f"  {label:<14} = {val:.5f}")

    print(f"\n  Train accuracy (200): {train_acc:.1f}%")
    print(f"  Val   accuracy (200): {val_acc:.1f}%")
    print(f"  Deg exponent:         {best[8]:.4f}", end="  →  ")
    if   best[8] < 1.1: print("LINEAR")
    elif best[8] < 1.7: print("MODERATE CURVE (~1.5)")
    else:               print("QUADRATIC/STRONG (~2.0)")

    # Hard stop if still broken after Stage B
    if val_acc < 2.0 and sample_size >= 1000:
        print(f"\n  STOP — {val_acc:.1f}% after {sample_size} races.")
        print("  The formula shape is wrong.")
        print("  Fix the formula before running again.")
        break

    if val_acc >= 80.0:
        print(f"\n  EXCELLENT — {val_acc:.1f}%! Formula is correct.")
        break

    print(f"\n  Proceeding to next stage...")

# ── FINAL OUTPUT ─────────────────────────────────────────────
if current_best is not None:
    print(f"\n{'='*55}")
    print("PASTE INTO race_simulator.py:")
    print(f"{'='*55}")
    print("PARAMS = [")
    for label, val in zip(labels, current_best):
        print(f"    {val:.8f},  # {label}")
    print("]")