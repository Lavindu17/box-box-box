import json
import glob
import numpy as np

# Your Stage G1 "God-Tier" Parameters
PARAMS = [
    -1.62795984,  # soft_off
     1.14978744,  # hard_off
     9.13138752,  # soft_grace
    18.79647773,  # med_grace
    28.66262315,  # hard_grace
     0.78875823,  # soft_deg
     0.35096600,  # med_deg
     0.16267841,  # hard_deg
     1.41521022,  # deg_exp
     0.03346107,  # temp_soft
     0.01634178,  # temp_med
     0.00833413,  # temp_hard
]

def simulate(race, params):
    cfg = race['race_config']
    base = cfg['base_lap_time']
    penalty = cfg['pit_lane_time']
    laps = cfg['total_laps']
    temp = cfg['track_temp']

    offsets   = {'SOFT': params[0], 'MEDIUM': 0.0,       'HARD': params[1]}
    graces    = {'SOFT': params[2], 'MEDIUM': params[3], 'HARD': params[4]}
    degs      = {'SOFT': params[5], 'MEDIUM': params[6], 'HARD': params[7]}
    temp_sens = {'SOFT': params[9], 'MEDIUM': params[10], 'HARD': params[11]}
    deg_exp   = params[8]

    times = {}
    for strat in race['strategies'].values():
        did  = strat['driver_id']
        tire = strat['starting_tire']
        age  = 0
        t    = 0.0
        pits = {s['lap']: s['to_tire'] for s in strat['pit_stops']}
        
        for lap in range(1, laps + 1):
            age += 1
            lpg  = float(age) - graces[tire]
            if lpg < 0.0:
                lpg = 0.0
                
            # Safe degradation math
            deg_effect = 0.0
            if lpg > 0.0:
                deg_effect = (lpg ** deg_exp) * degs[tire]
                
            t += (base + offsets[tire] + deg_effect + temp_sens[tire] * temp)
            
            if lap in pits:
                t   += penalty
                tire = pits[lap]
                age  = 0
                
        times[did] = t
        
    # Python's built-in sorted() is naturally stable
    return sorted(times, key=times.get)

print("Loading historical data...")
with open('data/historical_races/races_00000-00999.json') as f:
    races = json.load(f)

# Position error breakdown
pos_errors = []
exact = 0
TEST_SIZE = 500

for race in races[:TEST_SIZE]:
    pred   = simulate(race, PARAMS)
    actual = race['finishing_positions']
    if pred == actual:
        exact += 1
    for pp, driver in enumerate(pred):
        pos_errors.append(abs(pp - actual.index(driver)))

errors = np.array(pos_errors)

print("\n" + "="*45)
print(" STAGE G1 - ERROR ANALYSIS REPORT")
print("="*45)
print(f"Perfect Races:          {exact}/{TEST_SIZE} = {exact/TEST_SIZE*100:.1f}%")
print(f"Drivers Exactly Right:  {(errors==0).mean()*100:.1f}%")
print(f"Drivers Off by ≤1 Pos:  {(errors<=1).mean()*100:.1f}%")
print(f"Drivers Off by ≤2 Pos:  {(errors<=2).mean()*100:.1f}%")
print(f"Mean Position Error:    {errors.mean():.3f} positions")
print(f"Max Position Error:     {errors.max()} positions")
print("="*45)