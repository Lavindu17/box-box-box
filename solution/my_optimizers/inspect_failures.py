import json
import glob
import numpy as np

PARAMS = [
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
]

COMPOUND_IDX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def lap_time(base, tire_idx, age, temp):
    offsets   = [PARAMS[0], 0.0,       PARAMS[1]]
    graces    = [PARAMS[2], PARAMS[3], PARAMS[4]]
    degs      = [PARAMS[5], PARAMS[6], PARAMS[7]]
    temp_sens = [PARAMS[9], PARAMS[10],PARAMS[11]]
    lpg = max(0.0, age - graces[tire_idx])
    return (base
            + offsets[tire_idx]
            + (lpg ** PARAMS[8]) * degs[tire_idx]
            + temp_sens[tire_idx] * temp)

def simulate(race):
    cfg     = race['race_config']
    base    = cfg['base_lap_time']
    penalty = cfg['pit_lane_time']
    laps    = cfg['total_laps']
    temp    = cfg['track_temp']
    times   = {}
    for strat in race['strategies'].values():
        did  = strat['driver_id']
        tire = COMPOUND_IDX[strat['starting_tire']]
        age  = 0
        t    = 0.0
        pits = {s['lap']: COMPOUND_IDX[s['to_tire']]
                for s in strat['pit_stops']}
        for lap in range(1, laps + 1):
            age += 1
            t   += lap_time(base, tire, age, temp)
            if lap in pits:
                t   += penalty
                tire = pits[lap]
                age  = 0
        times[did] = t
    return sorted(times, key=times.get)

with open('data/historical_races/races_00000-00999.json') as f:
    races = json.load(f)

# Find races we get WRONG and inspect them
wrong_races = []
for race in races[:500]:
    pred   = simulate(race)
    actual = race['finishing_positions']
    if pred != actual:
        wrong_races.append((race, pred, actual))

print(f"Wrong: {len(wrong_races)}/500")
print(f"\n--- INSPECTING FIRST 5 WRONG RACES ---")

for race, pred, actual in wrong_races[:5]:
    cfg = race['race_config']
    print(f"\nRace {race['race_id']}  "
          f"Track: {cfg['track']}  "
          f"Temp: {cfg['track_temp']}°C  "
          f"Laps: {cfg['total_laps']}")

    # Find which drivers swapped
    swaps = [(i, pred[i], actual[i])
             for i in range(20) if pred[i] != actual[i]]
    print(f"  Swapped positions: {len(swaps)}")

    # Show the two drivers closest to swapping
    for i, p_driver, a_driver in swaps[:2]:
        for strat in race['strategies'].values():
            if strat['driver_id'] == p_driver:
                ps = strat
            if strat['driver_id'] == a_driver:
                as_ = strat
        print(f"\n  Predicted pos {i+1}: {p_driver} "
              f"({ps['starting_tire']} → "
              f"{[s['to_tire'] for s in ps['pit_stops']]}  "
              f"pit@{[s['lap'] for s in ps['pit_stops']]})")
        print(f"  Actual    pos {i+1}: {a_driver} "
              f"({as_['starting_tire']} → "
              f"{[s['to_tire'] for s in as_['pit_stops']]}  "
              f"pit@{[s['lap'] for s in as_['pit_stops']]})")