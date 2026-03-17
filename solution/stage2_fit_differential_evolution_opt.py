import json
import glob
import numpy as np
from scipy.optimize import differential_evolution
import os

# ── Load data ONCE in main process ───────────────────────────
# Workers inherit this via fork — no duplicate loading
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
print(f"CPU cores available: {os.cpu_count()}")
print(f"Using 4 workers (safe for 16GB RAM)\n")

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

SAMPLE = 1000

def loss(params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg,
     deg_exp,
     temp_soft, temp_med, temp_hard) = params

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

def accuracy(params, races, n=300):
    return sum(
        1 for r in races[:n]
        if simulate(r, params) == r['finishing_positions']
    ) / n * 100

bounds = [
    (-3.0,  -0.1),
    ( 0.1,   3.0),
    ( 1.0,  15.0),
    ( 5.0,  25.0),
    (10.0,  45.0),
    ( 0.01,  1.0),
    ( 0.005, 0.5),
    ( 0.001, 0.2),
    ( 0.5,   3.0),
    ( 0.0,   0.1),
    ( 0.0,   0.1),
    ( 0.0,   0.1),
]

labels = ['soft_off','hard_off','soft_grace','med_grace','hard_grace',
          'soft_deg','med_deg','hard_deg','deg_exp',
          'temp_soft','temp_med','temp_hard']

# ── Stage A confirmed result ──────────────────────────────────
stage_a_best = np.array([
    -1.23536,
     1.50227,
     8.74053,
    17.09840,
    26.56549,
     0.39339,
     0.11865,
     0.05240,
     1.58851,
     0.04044,
     0.02870,
     0.00049,
])

stages = [
    (1000,  80,  "B — refinement",   300),
    (5000,  120, "C — precision",    500),
    (15000, 150, "D — final",       1000),
]

current_best = stage_a_best

if __name__ == '__main__':

    for sample_size, maxiter, stage_name, acc_n in stages:
        SAMPLE = sample_size

        print(f"\n{'='*55}")
        print(f"Stage {stage_name}")
        print(f"Races: {sample_size}  |  Generations: {maxiter}")
        print(f"Workers: 4  |  Population: {15 * len(bounds)} candidates")
        print(f"{'='*55}")

        result = differential_evolution(
            loss,
            bounds,
            maxiter       = maxiter,
            popsize       = 15,       # full population restored
            tol           = 0.0001,
            mutation      = (0.5, 1.5),
            recombination = 0.7,
            seed          = 42,
            disp          = True,
            polish        = True,
            workers       = 4,        # 4 cores, safe for 16GB RAM
            x0            = current_best,
            init          = 'latinhypercube'
        )

        best         = result.x
        current_best = best

        train_acc = accuracy(best, train_races, n=acc_n)
        val_acc   = accuracy(best, val_races,   n=acc_n)

        print(f"\nStage {stage_name[0]} result:")
        for label, val in zip(labels, best):
            print(f"  {label:<14} = {val:.6f}")
        print(f"\n  Train accuracy ({acc_n}): {train_acc:.1f}%")
        print(f"  Val   accuracy ({acc_n}): {val_acc:.1f}%")
        print(f"  Deg exponent: {best[8]:.4f}", end="  →  ")
        if   best[8] < 1.1: print("LINEAR")
        elif best[8] < 1.7: print("MODERATE (~1.5)")
        else:                print("QUADRATIC (~2.0)")

        # ── Print params after every stage — crash insurance ─
        print(f"\n  !! COPY THESE NOW IN CASE OF CRASH !!")
        print(f"  stage_{stage_name[0].lower()}_best = np.array([")
        for v in best:
            print(f"      {v:.8f},")
        print(f"  ])")

        if val_acc < 5.0 and sample_size >= 5000:
            print(f"\n  STOP — {val_acc:.1f}% at {sample_size} races.")
            print("  Formula needs revision.")
            break

        if val_acc >= 80.0:
            print(f"\n  EXCELLENT — {val_acc:.1f}%! Done.")
            break

        print(f"  Continuing to next stage...")

    print(f"\n{'='*55}")
    print("FINAL PARAMS — PASTE INTO race_simulator.py:")
    print(f"{'='*55}")
    print("PARAMS = [")
    for label, val in zip(labels, current_best):
        print(f"    {val:.8f},  # {label}")
    print("]")