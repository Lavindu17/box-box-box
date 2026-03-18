import json
import glob

print("Loading races...")
all_races = []
for fp in sorted(glob.glob('data/historical_races/races_*.json'))[:5]:
    with open(fp) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} races\n")

# Count how many pairs pass each filter progressively
# so we know which filter is killing the data

same_compound       = 0
both_one_stop       = 0
same_second_tire    = 0
pit_within_3        = 0
same_stint2_len     = 0  # the strictest filter — was requiring exact match
within_1_stint2_len = 0

for race in all_races:
    total_laps = race['race_config']['total_laps']
    strats = list(race['strategies'].values())
    for i in range(len(strats)):
        for j in range(i+1, len(strats)):
            s1, s2 = strats[i], strats[j]

            if s1['starting_tire'] != s2['starting_tire']:
                continue
            same_compound += 1

            if len(s1['pit_stops']) != 1 or len(s2['pit_stops']) != 1:
                continue
            both_one_stop += 1

            p1, p2 = s1['pit_stops'][0], s2['pit_stops'][0]

            if p1['to_tire'] != p2['to_tire']:
                continue
            same_second_tire += 1

            if abs(p1['lap'] - p2['lap']) > 3:
                continue
            pit_within_3 += 1

            s1_len2 = total_laps - p1['lap']
            s2_len2 = total_laps - p2['lap']

            if s1_len2 == s2_len2:
                same_stint2_len += 1

            if abs(s1_len2 - s2_len2) <= 1:
                within_1_stint2_len += 1

print("Filter funnel:")
print(f"  Same starting compound:         {same_compound:,}")
print(f"  Both exactly 1 stop:            {both_one_stop:,}")
print(f"  Same second compound:           {same_second_tire:,}")
print(f"  Pit laps within 3 of each other:{pit_within_3:,}")
print(f"  Exact same second stint length: {same_stint2_len:,}")
print(f"  Second stint within ±1 lap:     {within_1_stint2_len:,}")