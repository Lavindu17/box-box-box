import json
import glob
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
import time

COMPOUND_IDX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
MAX_PITS = 5

@njit(cache=True)
def simulate_numba(base, penalty, laps, temp,
                   start_tires, pit_laps, pit_tires, n_pits,
                   offsets, graces, degs, temp_sens, deg_exp):
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
        times[d] = t  # INSIDE the for d loop
    return times

def preprocess(races):
    processed = []
    for race in races:
        cfg     = race['race_config']
        base    = float(cfg['base_lap_time'])
        penalty = float(cfg['pit_lane_time'])
        laps    = int(cfg['total_laps'])
        temp    = float(cfg['track_temp'])
        strats  = list(race['strategies'].values())
        n       = len(strats)
        start_tires = np.zeros(n, dtype=np.int32)
        pit_laps    = np.zeros((n, MAX_PITS), dtype=np.int32)
        pit_tires   = np.zeros((n, MAX_PITS), dtype=np.int32)
        n_pits_arr  = np.zeros(n, dtype=np.int32)
        driver_ids  = []
        for i, strat in enumerate(strats):
            driver_ids.append(strat['driver_id'])
            start_tires[i] = COMPOUND_IDX[strat['starting_tire']]
            stops = strat['pit_stops']
            n_pits_arr[i] = min(len(stops), MAX_PITS)
            for j, stop in enumerate(stops[:MAX_PITS]):
                pit_laps[i, j]  = int(stop['lap'])
                pit_tires[i, j] = COMPOUND_IDX[stop['to_tire']]
        processed.append((base, penalty, laps, temp,
                          start_tires, pit_laps, pit_tires, n_pits_arr,
                          driver_ids, race['finishing_positions']))
    return processed

def get_order(times, driver_ids):
    return [driver_ids[i] for i in np.argsort(times)]

# ── Load and preprocess ONCE ──────────────────────────────────
print("Loading races...")
all_races = []
for fp in sorted(glob.glob('data/historical_races/races_*.json')):
    with open(fp) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} races")

split = int(len(all_races) * 0.8)
print("Preprocessing...")
train_p = preprocess(all_races[:split])
val_p   = preprocess(all_races[split:])
print(f"Train: {len(train_p)}  Val: {len(val_p)}\n")

# ── Warm up JIT ───────────────────────────────────────────────
print("Warming up Numba JIT (~10s)...")
_d = train_p[0]
_dummy = np.array([-2.0, 0.0, 1.3,
                    9.0, 18.0, 28.0,
                    0.9, 0.35, 0.15,
                    1.5,
                    0.033, 0.016, 0.011])
simulate_numba(_d[0], _d[1], _d[2], _d[3],
               _d[4], _d[5], _d[6], _d[7],
               np.array([_dummy[0], 0.0, _dummy[2]]),
               np.array([_dummy[3], _dummy[4], _dummy[5]]),
               np.array([_dummy[6], _dummy[7], _dummy[8]]),
               np.array([_dummy[10], _dummy[11], _dummy[12]]),
               _dummy[9])
print("JIT ready\n")

SAMPLE = 8000

def loss(params):
    p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 = params
    if p5 <= p6 or p6 <= p7:  return 1e9
    if p9 < p10 or p10 < p11: return 1e9
    offsets   = np.array([p0,  0.0, p1])
    graces    = np.array([p2,  p3,  p4])
    degs      = np.array([p5,  p6,  p7])
    temp_sens = np.array([p9,  p10, p11])
    error = 0
    for race in train_p[:SAMPLE]:
        base, penalty, laps, temp, \
        start_tires, pit_laps, pit_tires, n_pits, \
        driver_ids, actual = race
        times     = simulate_numba(base, penalty, laps, temp,
                                   start_tires, pit_laps, pit_tires, n_pits,
                                   offsets, graces, degs, temp_sens, p8)
        predicted = get_order(times, driver_ids)
        for pp, driver in enumerate(predicted):
            error += abs(pp - actual.index(driver))
    return error

def accuracy(params, processed, n=500):
    p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 = params
    offsets   = np.array([p0,  0.0, p1])
    graces    = np.array([p2,  p3,  p4])
    degs      = np.array([p5,  p6,  p7])
    temp_sens = np.array([p9,  p10, p11])
    correct = 0
    for race in processed[:n]:
        base, penalty, laps, temp, \
        start_tires, pit_laps, pit_tires, n_pits, \
        driver_ids, actual = race
        times     = simulate_numba(base, penalty, laps, temp,
                                   start_tires, pit_laps, pit_tires, n_pits,
                                   offsets, graces, degs, temp_sens, p8)
        predicted = get_order(times, driver_ids)
        if predicted == actual:
            correct += 1
    return correct / n * 100

# ── LOCKED VALUES ─────────────────────────────────────────────
# Consistent across every optimizer run — locking reduces
# search from 12D to 8D
SOFT_GRACE = 9.0
MED_GRACE  = 18.0
HARD_GRACE = 28.0
DEG_EXP    = 1.5

EPS = 1e-9

bounds = [
    (-2.5,  -1.2),                        # soft_off
    ( 1.0,   1.8),                        # hard_off
    (SOFT_GRACE - EPS, SOFT_GRACE + EPS), # soft_grace LOCKED
    (MED_GRACE  - EPS, MED_GRACE  + EPS), # med_grace  LOCKED
    (HARD_GRACE - EPS, HARD_GRACE + EPS), # hard_grace LOCKED
    ( 0.5,   1.3),                        # soft_deg
    ( 0.2,   0.55),                       # med_deg
    ( 0.08,  0.25),                       # hard_deg
    (DEG_EXP - EPS, DEG_EXP + EPS),       # deg_exp    LOCKED
    ( 0.015, 0.055),                      # temp_soft
    ( 0.008, 0.030),                      # temp_med
    ( 0.004, 0.020),                      # temp_hard
]

labels = ['soft_off','hard_off','soft_grace','med_grace','hard_grace',
          'soft_deg','med_deg','hard_deg','deg_exp',
          'temp_soft','temp_med','temp_hard']

x0 = np.array([
    -2.0,  1.3,
     9.0, 18.0, 28.0,
     0.9,  0.35, 0.15,
     1.5,
     0.033, 0.016, 0.011,
])

# ── Benchmark ─────────────────────────────────────────────────
print("Benchmarking with Numba...")
t0 = time.time()
loss(x0)
t1 = time.time()
per_eval = t1 - t0
per_gen  = per_eval * 240  # popsize=20 × 12 params = 240 candidates
print(f"Per eval:        {per_eval:.3f}s")
print(f"Per generation:  {per_gen:.0f}s = {per_gen/60:.1f} min")
print(f"200 generations: {per_gen*200/3600:.1f} hours\n")

print("Checking seed accuracy...")
seed_acc = accuracy(x0, val_p, n=500)
print(f"Seed val accuracy: {seed_acc:.1f}%\n")

stages = [
    (8000,  200, "F1 — 8 free params, tight bounds",  500),
    (20000, 200, "F2 — full data precision",          1000),
]

current_best = x0

for sample_size, maxiter, stage_name, acc_n in stages:
    SAMPLE = sample_size
    print(f"\n{'='*55}")
    print(f"Stage {stage_name}")
    print(f"Races: {sample_size}  |  Gens: {maxiter}  |  workers=1")
    print(f"{'='*55}")

    result = differential_evolution(
        loss,
        bounds,
        maxiter       = maxiter,
        popsize       = 20,
        tol           = 0.000001,
        mutation      = (0.2, 0.8),
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

    train_acc = accuracy(best, train_p, n=acc_n)
    val_acc   = accuracy(best, val_p,   n=acc_n)

    print(f"\nStage {stage_name} result:")
    for label, val in zip(labels, best):
        print(f"  {label:<14} = {val:.6f}")
    print(f"\n  Train accuracy ({acc_n}): {train_acc:.1f}%")
    print(f"  Val   accuracy ({acc_n}): {val_acc:.1f}%")

    print(f"\n  !! COPY NOW IN CASE OF CRASH !!")
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