import json
import glob
import numpy as np
from scipy.optimize import differential_evolution

# ── Compound encoding: integers instead of strings ───────────
# SOFT=0, MEDIUM=1, HARD=2
COMPOUND_IDX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def preprocess_races(races):
    """
    Convert race dicts into flat numpy arrays ONCE at startup.
    This eliminates all dict/string overhead from the hot loop.
    Returns a list of preprocessed race tuples.
    """
    processed = []
    for race in races:
        cfg     = race['race_config']
        base    = cfg['base_lap_time']
        penalty = cfg['pit_lane_time']
        laps    = cfg['total_laps']
        temp    = cfg['track_temp']

        drivers = []
        for strat in race['strategies'].values():
            # Encode starting tire as int
            start_tire = COMPOUND_IDX[strat['starting_tire']]
            # Encode pit stops as (lap, new_tire_int) tuples
            pits = {
                s['lap']: COMPOUND_IDX[s['to_tire']]
                for s in strat['pit_stops']
            }
            drivers.append((strat['driver_id'], start_tire, pits))

        processed.append((
            base, penalty, laps, temp,
            drivers,
            race['finishing_positions']
        ))
    return processed

print("Loading all historical races...")
all_races = []
for filepath in sorted(glob.glob('data/historical_races/races_*.json')):
    with open(filepath) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} total races")

split = int(len(all_races) * 0.8)

print("Preprocessing race data into arrays...")
train_processed = preprocess_races(all_races[:split])
val_processed   = preprocess_races(all_races[split:])
print(f"Train: {len(train_processed)}  |  Val: {len(val_processed)}")
print("Preprocessing complete — no more dict lookups in hot loop\n")

# ── Optimized simulate — no dicts, no string ops ─────────────
def simulate_fast(processed_race, params):
    base, penalty, laps, temp, drivers, _ = processed_race

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = params

    # Inline arrays — index by compound int
    # SOFT=0, MEDIUM=1, HARD=2
    offsets   = (p0,  0.0, p1)   # compound speed offset
    graces    = (p2,  p3,  p4)   # grace periods
    degs      = (p5,  p6,  p7)   # degradation rates
    temp_sens = (p9,  p10, p11)  # temp sensitivity
    deg_exp   = p8

    times = {}
    for did, start_tire, pits in drivers:
        tire = start_tire
        age  = 0
        t    = 0.0

        for lap in range(1, laps + 1):
            age += 1
            # Inline lap_time — no function call overhead
            lpg = age - graces[tire]
            if lpg < 0.0:
                lpg = 0.0
            t += (base
                  + offsets[tire]
                  + (lpg ** deg_exp) * degs[tire]
                  + temp_sens[tire] * temp)

            if lap in pits:
                t   += penalty
                tire = pits[lap]
                age  = 0

        times[did] = t

    return sorted(times, key=times.get)

SAMPLE = 8000

def loss(params):
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = params

    # Constraints — fast early exit
    if p2 >= p3 or p3 >= p4: return 1e9
    if p5 <= p6 or p6 <= p7: return 1e9
    if p9 < p10 or p10 < p11: return 1e9

    error = 0
    for race in train_processed[:SAMPLE]:
        predicted = simulate_fast(race, params)
        actual    = race[5]  # finishing_positions at index 5
        for pred_pos, driver in enumerate(predicted):
            error += abs(pred_pos - actual.index(driver))
    return error

def accuracy(params, processed, n=500):
    return sum(
        1 for r in processed[:n]
        if simulate_fast(r, params) == r[5]
    ) / n * 100

bounds = [
    (-4.0,  -0.5),
    ( 0.5,   3.0),
    ( 5.0,  15.0),
    (12.0,  25.0),
    (20.0,  40.0),
    ( 0.3,   2.0),
    ( 0.1,   0.8),
    ( 0.05,  0.4),
    ( 1.0,   2.5),
    ( 0.01,  0.08),
    ( 0.005, 0.05),
    ( 0.0,   0.04),
]

labels = ['soft_off','hard_off','soft_grace','med_grace','hard_grace',
          'soft_deg','med_deg','hard_deg','deg_exp',
          'temp_soft','temp_med','temp_hard']

stage_d_best = np.array([
    -1.95049364,
     1.32673000,
     9.05720595,
    18.40432358,
    27.94283633,
     0.86983298,
     0.34446878,
     0.14992882,
     1.50198125,
     0.03303927,
     0.01560123,
     0.01089012,
])

# ── Quick benchmark before full run ──────────────────────────
import time
print("Benchmarking speed...")
t0 = time.time()
test_loss = loss(stage_d_best)
t1 = time.time()
per_eval = t1 - t0
print(f"Single loss eval: {per_eval:.2f}s on {SAMPLE} races")
print(f"Estimated per generation (180 candidates / 4 workers): "
      f"{per_eval * 180 / 4:.0f}s = "
      f"{per_eval * 180 / 4 / 60:.1f} min")
print(f"Estimated total (200 generations): "
      f"{per_eval * 180 / 4 * 200 / 3600:.1f} hours\n")

stages = [
    (8000,  150, "E1 — fast precision",  500),
    (15000, 200, "E2 — full precision", 1000),
]

current_best = stage_d_best

if __name__ == '__main__':

    for sample_size, maxiter, stage_name, acc_n in stages:
        SAMPLE = sample_size

        print(f"\n{'='*55}")
        print(f"Stage {stage_name}")
        print(f"Races: {sample_size}  |  Generations: {maxiter}")
        print(f"{'='*55}")

        result = differential_evolution(
            loss,
            bounds,
            maxiter       = maxiter,
            popsize       = 15,
            tol           = 0.00001,
            mutation      = (0.3, 1.0),
            recombination = 0.9,
            seed          = 42,
            disp          = True,
            polish        = True,
            workers       = 4,
            x0            = current_best,
            init          = 'latinhypercube'
        )

        best         = result.x
        current_best = best

        train_acc = accuracy(best, train_processed, n=acc_n)
        val_acc   = accuracy(best, val_processed,   n=acc_n)

        print(f"\nStage {stage_name[0]} result:")
        for label, val in zip(labels, best):
            print(f"  {label:<14} = {val:.6f}")
        print(f"\n  Train accuracy ({acc_n}): {train_acc:.1f}%")
        print(f"  Val   accuracy ({acc_n}): {val_acc:.1f}%")
        print(f"  Deg exponent: {best[8]:.4f}")

        print(f"\n  !! COPY THESE NOW !!")
        print(f"  best = np.array([")
        for v in best:
            print(f"      {v:.8f},")
        print(f"  ])")

        if val_acc >= 80.0:
            print(f"\n  EXCELLENT — {val_acc:.1f}%!")
            break

        print(f"  Continuing...")

    print(f"\n{'='*55}")
    print("PASTE INTO race_simulator.py:")
    print(f"{'='*55}")
    print("PARAMS = [")
    for label, val in zip(labels, current_best):
        print(f"    {val:.8f},  # {label}")
    print("]")