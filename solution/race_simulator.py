import json
import sys

PARAMS = [
      -1.62795984,  # soft_off
      1.14978744,   # hard_off
      9.13138752,   # soft_grace
      18.79647773,  # med_grace
      28.66262315,  # hard_grace
      0.78875823,   # soft_deg
      0.35096600,   # med_deg
      0.16267841,   # hard_deg
      1.41521022,   # deg_exp
      0.03346107,   # temp_soft
      0.01634178,   # temp_med
      0.00833413,   # temp_hard
]

COMPOUND_IDX = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}

def simulate(race_input):
    cfg     = race_input['race_config']
    base    = cfg['base_lap_time']
    penalty = cfg['pit_lane_time']
    laps    = cfg['total_laps']
    temp    = cfg['track_temp']

    offsets   = [PARAMS[0], 0.0,        PARAMS[1]]
    graces    = [PARAMS[2], PARAMS[3],  PARAMS[4]]
    degs      = [PARAMS[5], PARAMS[6],  PARAMS[7]]
    temp_sens = [PARAMS[9], PARAMS[10], PARAMS[11]]
    deg_exp   = PARAMS[8]

    times = {}
    for strat in race_input['strategies'].values():
        did  = strat['driver_id']
        tire = COMPOUND_IDX[strat['starting_tire']]
        age  = 0
        t    = 0.0
        pits = {s['lap']: COMPOUND_IDX[s['to_tire']]
                for s in strat['pit_stops']}

        for lap in range(1, laps + 1):
            age += 1
            lpg  = age - graces[tire]
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

data   = json.load(sys.stdin)
output = {
    "race_id":             data["race_id"],
    "finishing_positions": simulate(data)
}
print(json.dumps(output))