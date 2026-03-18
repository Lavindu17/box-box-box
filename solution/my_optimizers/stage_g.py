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
            deg_effect = 0.0
            if lpg > 0.0:
                deg_effect = (lpg ** deg_exp) * degs[tire]
            t += (base
                  + offsets[tire]
                  + deg_effect
                  + temp_sens[tire] * temp)
            for p in range(np_):
                if pit_laps[d, p] == lap:
                    t   += penalty
                    tire = pit_tires[d, p]
                    age  = 0
                    break
        times[d] = t
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
    return [driver_ids[i] for i in np.argsort(times, kind='mergesort')]

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

print("Warming up Numba JIT (~10s)...")
_d = train_p[0]
simulate_numba(_d[0], _d[1], _d[2], _d[3],
               _d[4], _d[5], _d[6], _d[7],
               np.array([-1.95, 0.0, 1.33]),
               np.array([9.06, 18.40, 27.94]),
               np.array([0.87, 0.34, 0.15]),
               np.array([0.033, 0.016, 0.011]),
               1.50)
print("JIT ready\n")

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

stage_d = np.array([
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

bounds = [
    (-2.4,  -1.6),
    ( 1.1,   1.6),
    ( 8.0,  10.5),
    (16.5,  20.5),
    (25.5,  30.5),
    ( 0.70,  1.05),
    ( 0.28,  0.42),
    ( 0.12,  0.18),
    ( 1.30,  1.70),
    ( 0.025, 0.042),
    ( 0.010, 0.022),
    ( 0.007, 0.016),
]

labels = ['soft_off','hard_off','soft_grace','med_grace','hard_grace',
          'soft_deg','med_deg','hard_deg','deg_exp',
          'temp_soft','temp_med','temp_hard']

print("Checking Stage D accuracy...")
stage_d_acc = accuracy(stage_d, val_p, n=1000)
print(f"Stage D val accuracy: {stage_d_acc:.1f}%")

if stage_d_acc < 10.0:
    print("ERROR: Stage D accuracy too low — Numba bug still present")
    print("Delete __pycache__ and rerun")
    exit()

print("Accuracy confirmed — proceeding\n")

t0 = time.time()
loss(stage_d)
t1 = time.time()
per_eval = t1 - t0
per_gen  = per_eval * 240
print(f"Per eval: {per_eval:.2f}s | Per gen: {per_gen/60:.1f} min")
print(f"300 gens: {per_gen*300/3600:.1f} hours\n")

stages = [
    (5000, 150, "G1 — tight bounds, all free",  500),
    (15000, 100, "G2 — full data",               1000),
]

current_best = stage_d

for sample_size, maxiter, stage_name, acc_n in stages:
    SAMPLE = sample_size
    print(f"\n{'='*55}")
    print(f"Stage {stage_name}")
    print(f"Races: {sample_size}  |  Gens: {maxiter}  |  All 12 params free")
    print(f"{'='*55}")

    result = differential_evolution(
        loss,
        bounds,
        maxiter       = maxiter,
        popsize       = 20,
        tol           = 0.000001,
        mutation      = (0.2, 0.8),
        recombination = 0.9,
        seed          = 99,
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

best_so_far = [stage_d.copy()]

def callback(xk, convergence):
    current = loss(xk)
    if current < loss(best_so_far[0]):
        best_so_far[0] = xk.copy()
        acc = accuracy(xk, val_p, n=200)
        print(f"  >> New best: {acc:.1f}% — params saved")
        # Write to file immediately so crash can't lose it
        with open('solution/best_params_live.txt', 'w') as f:
            for v in xk:
                f.write(f"{v:.8f}\n")