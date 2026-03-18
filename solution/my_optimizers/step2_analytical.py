import json
import glob

print("Loading all races...")
all_races = []
for fp in sorted(glob.glob('data/historical_races/races_*.json')):
    with open(fp) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} races\n")

for compound in ['SOFT', 'MEDIUM', 'HARD']:
    stint_results = {}

    for race in all_races:
        finishing  = race['finishing_positions']
        total_laps = race['race_config']['total_laps']
        strats     = list(race['strategies'].values())

        for i in range(len(strats)):
            for j in range(i+1, len(strats)):
                s1, s2 = strats[i], strats[j]

                if s1['starting_tire'] != compound: continue
                if s2['starting_tire'] != compound: continue
                if len(s1['pit_stops']) != 1:       continue
                if len(s2['pit_stops']) != 1:       continue

                p1 = s1['pit_stops'][0]
                p2 = s2['pit_stops'][0]

                if p1['to_tire'] != p2['to_tire']:  continue

                # Second stint lengths must be EXACTLY equal
                # (this controls the second half of the race)
                s1_stint2 = total_laps - p1['lap']
                s2_stint2 = total_laps - p2['lap']
                if s1_stint2 != s2_stint2:          continue

                # Pit laps can differ by 1–5
                # (the difference IS the first stint length difference)
                diff = abs(p1['lap'] - p2['lap'])
                if diff < 1 or diff > 5:            continue

                # Who stayed out longer?
                if p1['lap'] < p2['lap']:
                    early_d, late_d = s1, s2
                    late_pit_lap    = p2['lap']
                else:
                    early_d, late_d = s2, s1
                    late_pit_lap    = p1['lap']

                r_early = finishing.index(early_d['driver_id'])
                r_late  = finishing.index(late_d['driver_id'])
                late_won = r_late < r_early

                if late_pit_lap not in stint_results:
                    stint_results[late_pit_lap] = []
                stint_results[late_pit_lap].append(late_won)

    print(f"\n{compound} tire — staying out longer hurt?")
    print(f"{'Pit lap':>8} {'Helped%':>9} {'n':>6}  {'Signal'}")
    print("-" * 50)

    crossover = None
    prev_rate = None
    for lap in sorted(stint_results.keys()):
        results = stint_results[lap]
        if len(results) < 10: continue
        rate = sum(results) / len(results)
        signal = ""
        if rate < 0.45 and crossover is None:
            signal = "← GRACE PERIOD ENDS HERE"
            crossover = lap
        change = ""
        if prev_rate is not None:
            delta = rate - prev_rate
            change = f"({delta:+.2f})"
        print(f"{lap:>8} {rate*100:>8.1f}% {len(results):>6}  {signal} {change}")
        prev_rate = rate

    if crossover:
        print(f"\n  >>> {compound} grace period = {crossover - 1} laps <<<")
    else:
        # Find the trend even without clean crossover
        laps_sorted = sorted(k for k in stint_results if len(stint_results[k]) >= 10)
        if laps_sorted:
            rates = [(lap, sum(stint_results[lap])/len(stint_results[lap]))
                     for lap in laps_sorted]
            min_rate_lap, min_rate = min(rates, key=lambda x: x[1])
            max_rate_lap, max_rate = max(rates, key=lambda x: x[1])
            print(f"\n  No clean crossover found.")
            print(f"  Highest helped%: {max_rate*100:.1f}% at lap {max_rate_lap}")
            print(f"  Lowest  helped%: {min_rate*100:.1f}% at lap {min_rate_lap}")
            print(f"  Trend: {'degrading earlier than lap range shows' if min_rate < 0.4 else 'may degrade later than data range'}")