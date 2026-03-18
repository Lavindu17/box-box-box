import json
import glob
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import time

COMPOUND_IDX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

@njit(cache=True)
def simulate_numba(base, penalty, laps, temp,
                   start_tires,
                   pit_laps,
                   pit_tires,
                   n_pits,
                   offsets,
                   graces,
                   degs,
                   temp_sens,
                   deg_exp):
    n_drivers = start_tires.shape[0]
    times = np.zeros(n_drivers)

    for d in range(n_drivers):
        tire = start_tires[d]
        age  = 0
        t    = 0.0
        np_  = n_pits[d]

        for lap in range(1, laps + 1):
            age += 1
            lpg  = float(age) - graces[tire]
            if lpg < 0.0:
                lpg = 0.0
            t += (base
                  + offsets[tire]
                  + (lpg ** deg_exp) * degs[tire]
                  + temp_sens[tire] * temp)

            for p in range(np_):
                if pit_laps[d, p] == lap:
                    t   += penalty
                    tire = pit_tires[d, p]
                    age  = 0
                    break

        times[d] = t  # ← FIXED: inside for d loop

    return times

MAX_PITS = 5

def preprocess_races(races):
    processed = []
    for race in races:
        cfg     = race['race_config']
        base    = float(cfg['base_lap_time'])
        penalty = float(cfg['pit_lane_time'])
        laps    = int(cfg['total_laps'])
        temp    = float(cfg['track_temp'])

        strats     = list(race['strategies'].values())
        n_drivers  = len(strats)

        start_tires = np.zeros(n_drivers, dtype=np.int32)
        pit_laps    = np.zeros((n_drivers, MAX_PITS), dtype=np.int32)
        pit_tires   = np.zeros((n_drivers, MAX_PITS), dtype=np.int32)
        n_pits_arr  = np.zeros(n_drivers, dtype=np.int32)
        driver_ids  = []

        for i, strat in enumerate(strats):
            driver_ids.append(strat['driver_id'])
            start_tires[i] = COMPOUND_IDX[strat['starting_tire']]
            stops = strat['pit_stops']
            n_pits_arr[i] = min(len(stops), MAX_PITS)
            for j, stop in enumerate(stops):
                if j >= MAX_PITS:
                    break
                pit_laps[i, j]  = int(stop['lap'])
                pit_tires[i, j] = COMPOUND_IDX[stop['to_tire']]

        processed.append((
            base, penalty, laps, temp,
            start_tires, pit_laps, pit_tires, n_pits_arr,
            driver_ids,
            race['finishing_positions']
        ))
    return processed

def get_finishing_order(times, driver_ids):
    order = np.argsort(times)
    return [driver_ids[i] for i in order]

# ── Load and preprocess ONCE ──────────────────────────────────
print("Loading all historical races...")
all_races = []
for filepath in sorted(glob.glob('data/historical_races/races_*.json')):
    with open(filepath) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} total races")

split = int(len(all_races) * 0.8)
print("Preprocessing into numpy arrays...")
train_processed = preprocess_races(all_races[:split])
val_processed   = preprocess_races(all_races[split:])
print(f"Train: {len(train_processed)}  |  Val: {len(val_processed)}\n")

# ── Warm up Numba JIT ─────────────────────────────────────────
print("Warming up Numba JIT compiler (~10 seconds)...")
_d = train_processed[0]
_p = np.array([-1.95, 1.33, 9.06, 18.40, 27.94,
                0.87,  0.34, 0.15,  1.50,
                0.033, 0.016, 0.011])
_ = simulate_numba(
    _d[0], _d[1], _d[2], _d[3],
    _d[4], _d[5], _d[6], _d[7],
    np.array([_p[0], 0.0, _p[1]]),
    np.array([_p[2], _p[3], _p[4]]),
    np.array([_p[5], _p[6], _p[7]]),
    np.array([_p[9], _p[10], _p[11]]),
    _p[8]
)
print("JIT compilation done\n")

SAMPLE = 5000

def loss(params):
    p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 = params

    if p2 >= p3 or p3 >= p4:  return 1e9
    if p5 <= p6 or p6 <= p7:  return 1e9
    if p9 < p10 or p10 < p11: return 1e9

    offsets   = np.array([p0,  0.0, p1])
    graces    = np.array([p2,  p3,  p4])
    degs      = np.array([p5,  p6,  p7])
    temp_sens = np.array([p9,  p10, p11])

    error = 0
    for race in train_processed[:SAMPLE]:
        base, penalty, laps, temp, \
        start_tires, pit_laps, pit_tires, n_pits, \
        driver_ids, actual = race

        times     = simulate_numba(
            base, penalty, laps, temp,
            start_tires, pit_laps, pit_tires, n_pits,
            offsets, graces, degs, temp_sens, p8
        )
        predicted = get_finishing_order(times, driver_ids)

        for pred_pos, driver in enumerate(predicted):
            error += abs(pred_pos - actual.index(driver))

    return error

def accuracy(params, processed, n=500):
    p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 = params
    offsets   = np.array([p0,  0.0, p1])
    graces    = np.array([p2,  p3,  p4])
    degs      = np.array([p5,  p6,  p7])
    temp_sens = np.array([p9,  p10, p11])
    correct   = 0
    for race in processed[:n]:
        base, penalty, laps, temp, \
        start_tires, pit_laps, pit_tires, n_pits, \
        driver_ids, actual = race
        times     = simulate_numba(
            base, penalty, laps, temp,
            start_tires, pit_laps, pit_tires, n_pits,
            offsets, graces, degs, temp_sens, p8
        )
        predicted = get_finishing_order(times, driver_ids)
        if predicted == actual:
            correct += 1
    return correct / n * 100

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

# ── Verify fix before benchmarking ───────────────────────────
print("Verifying fix — checking accuracy on 100 races...")
quick_acc = accuracy(stage_d_best, val_processed, n=100)
print(f"Quick accuracy check: {quick_acc:.1f}%")
if quick_acc < 5.0:
    print("ERROR: Still getting 0% — check simulate_numba indentation")
    exit()
print("Fix confirmed — proceeding to benchmark\n")

# ── Benchmark ─────────────────────────────────────────────────
print("Benchmarking...")
t0 = time.time()
loss(stage_d_best)
t1 = time.time()
per_eval = t1 - t0
per_gen  = per_eval * 180  # popsize=15 × 12 params = 180 candidates
print(f"Per eval:         {per_eval:.3f}s  (was ~3s before Numba)")
print(f"Speedup:          ~{3.0/max(per_eval,0.001):.0f}x faster")
print(f"Per generation:   {per_gen:.1f}s = {per_gen/60:.1f} min")
print(f"200 generations:  {per_gen*200/3600:.1f} hours\n")

stages = [
    (5000,  200, "E1 — fast precision",   500),
    (15000, 200, "E2 — full precision",  1000),
    (24000, 150, "E3 — maximum",         1000),
]

current_best = stage_d_best

for sample_size, maxiter, stage_name, acc_n in stages:
    SAMPLE = sample_size

    print(f"\n{'='*55}")
    print(f"Stage {stage_name}")
    print(f"Races: {sample_size}  |  Generations: {maxiter}  |  workers=1")
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
        workers       = 1,
        x0            = current_best,
        init          = 'latinhypercube'
    )

    best         = result.x
    current_best = best

    train_acc = accuracy(best, train_processed, n=acc_n)
    val_acc   = accuracy(best, val_processed,   n=acc_n)

    print(f"\nStage {stage_name} result:")
    for label, val in zip(labels, best):
        print(f"  {label:<14} = {val:.6f}")
    print(f"\n  Train accuracy ({acc_n}): {train_acc:.1f}%")
    print(f"  Val   accuracy ({acc_n}): {val_acc:.1f}%")
    print(f"  Deg exponent: {best[8]:.4f}")

    print(f"\n  !! COPY THESE NOW IN CASE OF CRASH !!")
    print(f"  best = np.array([")
    for v in best:
        print(f"      {v:.8f},")
    print(f"  ])")

    if val_acc >= 80.0:
        print(f"\n  EXCELLENT — {val_acc:.1f}%!")
        break

    print(f"  Continuing to next stage...")

print(f"\n{'='*55}")
print("FINAL PARAMS — PASTE INTO race_simulator.py:")
print(f"{'='*55}")
print("PARAMS = [")
for label, val in zip(labels, current_best):
    print(f"    {val:.8f},  # {label}")
print("]")