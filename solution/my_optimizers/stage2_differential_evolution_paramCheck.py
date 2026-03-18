import json
import glob
import numpy as np

PARAMS = np.array([
    -1.23536, 1.50227, 8.74053, 17.09840, 26.56549,
     0.39339, 0.11865, 0.05240, 1.58851,
     0.04044, 0.02870, 0.00049,
])

def lap_time(base, tire, age, temp, params):
    (soft_off, hard_off,
     soft_grace, med_grace, hard_grace,
     soft_deg, med_deg, hard_deg, deg_exp,
     temp_soft, temp_med, temp_hard) = params
    offsets   = {'SOFT': soft_off,  'MEDIUM': 0.0,   'HARD': hard_off}
    graces    = {'SOFT': soft_grace,'MEDIUM': med_grace,'HARD': hard_grace}
    degs      = {'SOFT': soft_deg,  'MEDIUM': med_deg,  'HARD': hard_deg}
    temp_sens = {'SOFT': temp_soft, 'MEDIUM': temp_med, 'HARD': temp_hard}
    laps_past_grace    = max(0.0, age - graces[tire])
    degradation_effect = (laps_past_grace ** deg_exp) * degs[tire]
    return base + offsets[tire] + degradation_effect + temp_sens[tire] * temp

def simulate(race, params):
    cfg = race['race_config']
    times = {}
    for strat in race['strategies'].values():
        did  = strat['driver_id']
        tire = strat['starting_tire']
        age  = 0
        t    = 0.0
        pits = {s['lap']: s['to_tire'] for s in strat['pit_stops']}
        for lap in range(1, cfg['total_laps'] + 1):
            age += 1
            t   += lap_time(cfg['base_lap_time'], tire, age,
                            cfg['track_temp'], params)
            if lap in pits:
                t   += cfg['pit_lane_time']
                tire = pits[lap]
                age  = 0
        times[did] = t
    return sorted(times, key=times.get)

with open('data/historical_races/races_00000-00999.json') as f:
    races = json.load(f)

# Position error breakdown
pos_errors = []
exact = 0
for race in races[:500]:
    pred   = simulate(race, PARAMS)
    actual = race['finishing_positions']
    if pred == actual:
        exact += 1
    for pp, driver in enumerate(pred):
        pos_errors.append(abs(pp - actual.index(driver)))

errors = np.array(pos_errors)
print(f"Exact match:        {exact}/500 = {exact/5:.1f}%")
print(f"Drivers exactly right: {(errors==0).mean()*100:.1f}%")
print(f"Drivers off by ≤1:    {(errors<=1).mean()*100:.1f}%")
print(f"Drivers off by ≤2:    {(errors<=2).mean()*100:.1f}%")
print(f"Mean position error:  {errors.mean():.3f}")
print(f"Max position error:   {errors.max()}")