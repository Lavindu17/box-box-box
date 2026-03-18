import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# ── Load first batch of races (enough for exploration) ──────
with open('data/historical_races/races_00000-00999.json') as f:
    races = json.load(f)

print(f"Loaded {len(races)} races")
print("\n── FIRST RACE STRUCTURE ──")
r = races[0]
print(f"Track:        {r['race_config']['track']}")
print(f"Total laps:   {r['race_config']['total_laps']}")
print(f"Base laptime: {r['race_config']['base_lap_time']}s")
print(f"Pit penalty:  {r['race_config']['pit_lane_time']}s")
print(f"Track temp:   {r['race_config']['track_temp']}°C")
print(f"\nFinishing order: {r['finishing_positions']}")

# ── EXPERIMENT 1: How fast is each compound relative to each other?
# Find pairs of drivers who both run a single no-stop race
# (or pit at the same lap) — one on SOFT, one on HARD/MEDIUM.
# The position difference tells us compound speed order.
print("\n── COMPOUND SPEED CHECK ──")
compound_wins = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}
compound_counts = {'SOFT': 0, 'MEDIUM': 0, 'HARD': 0}

for race in races[:500]:
    finishing = race['finishing_positions']
    for pos_key, strat in race['strategies'].items():
        driver = strat['driver_id']
        rank = finishing.index(driver)  # 0 = 1st place
        compound = strat['starting_tire']
        compound_counts[compound] += 1
        if rank < 10:  # top half
            compound_wins[compound] += 1

for c in ['SOFT', 'MEDIUM', 'HARD']:
    if compound_counts[c] > 0:
        top_half_rate = compound_wins[c] / compound_counts[c]
        print(f"  {c}: top-half finish rate = {top_half_rate:.1%}  (n={compound_counts[c]})")

# ── EXPERIMENT 2: Does degradation look linear or curved?
# Key idea: find drivers who DO NOT pit at all.
# We know their total_time = base_laptime*laps + offset + degradation.
# Compare drivers with same compound, same track, different race lengths
# to see if time-per-lap grows linearly or accelerates.
print("\n── DEGRADATION SHAPE CHECK ──")
no_stop_races = []
for race in races:
    config = race['race_config']
    for pos_key, strat in race['strategies'].items():
        if len(strat['pit_stops']) == 0:
            no_stop_races.append({
                'compound': strat['starting_tire'],
                'laps': config['total_laps'],
                'base': config['base_lap_time'],
                'temp': config['track_temp'],
                'driver': strat['driver_id'],
                'race': race
            })

print(f"Found {len(no_stop_races)} no-stop stints to analyze")

# Group by compound
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    stints = [s for s in no_stop_races if s['compound'] == compound]
    print(f"\n  {compound}: {len(stints)} no-stop stints")
    if len(stints) > 0:
        lens = [s['laps'] for s in stints]
        print(f"    Race lengths range: {min(lens)} – {max(lens)} laps")

# ── EXPERIMENT 3: Temperature distribution in the dataset
temps = [r['race_config']['track_temp'] for r in races]
print(f"\n── TRACK TEMPERATURE RANGE ──")
print(f"  Min: {min(temps)}°C  Max: {max(temps)}°C  Mean: {np.mean(temps):.1f}°C")
print(f"  Unique values: {sorted(set(temps))}")

# ── PLOT: compound offset — how much faster is SOFT vs MEDIUM?
# Isolate pairs: SOFT driver and MEDIUM driver,
# same race, both do exactly 1 pit stop at the same lap.
print("\n── BUILDING COMPOUND COMPARISON PLOT ──")
soft_advantage = []  # positive = SOFT faster (lower total time)

for race in races[:2000]:
    finishing = race['finishing_positions']
    strategies = list(race['strategies'].values())
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            d1, d2 = strategies[i], strategies[j]
            tires = {d1['starting_tire'], d2['starting_tire']}
            if tires != {'SOFT', 'MEDIUM'}:
                continue
            if len(d1['pit_stops']) != 1 or len(d2['pit_stops']) != 1:
                continue
            if d1['pit_stops'][0]['lap'] != d2['pit_stops'][0]['lap']:
                continue
            stint = d1['pit_stops'][0]['lap']
            r1 = finishing.index(d1['driver_id'])
            r2 = finishing.index(d2['driver_id'])
            soft_driver = d1 if d1['starting_tire'] == 'SOFT' else d2
            soft_rank = finishing.index(soft_driver['driver_id'])
            med_driver = d2 if d1['starting_tire'] == 'SOFT' else d1
            med_rank = finishing.index(med_driver['driver_id'])
            soft_advantage.append({
                'stint': stint,
                'soft_ahead': soft_rank < med_rank,
                'temp': race['race_config']['track_temp']
            })

if soft_advantage:
    # Plot: at what stint length does SOFT stop winning?
    bins = range(5, 70, 5)
    x_vals, y_vals, n_vals = [], [], []
    for b in bins:
        group = [s for s in soft_advantage if b <= s['stint'] < b+5]
        if len(group) >= 5:
            rate = sum(s['soft_ahead'] for s in group) / len(group)
            x_vals.append(b + 2.5)
            y_vals.append(rate)
            n_vals.append(len(group))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(x_vals, y_vals, width=4, color='#E8593C', alpha=0.8)
    plt.axhline(0.5, color='gray', linestyle='--', label='50/50 line')
    plt.xlabel('Stint length (laps)')
    plt.ylabel('SOFT win rate vs MEDIUM')
    plt.title('Does SOFT degrade faster? (crossover = yes)')
    plt.legend()

    # Plot: does temperature affect the crossover point?
    cool = [s for s in soft_advantage if s['temp'] < 30]
    hot  = [s for s in soft_advantage if s['temp'] >= 40]
    plt.subplot(1, 2, 2)
    for group, label, color in [(cool, 'Cool (<30°C)', '#3B8BD4'),
                                  (hot,  'Hot (≥40°C)', '#E8593C')]:
        x_vals, y_vals = [], []
        for b in bins:
            g = [s for s in group if b <= s['stint'] < b+5]
            if len(g) >= 3:
                x_vals.append(b + 2.5)
                y_vals.append(sum(s['soft_ahead'] for s in g) / len(g))
        if x_vals:
            plt.plot(x_vals, y_vals, marker='o', label=label, color=color)
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.xlabel('Stint length (laps)')
    plt.ylabel('SOFT win rate vs MEDIUM')
    plt.title('Temperature shifts the crossover? (yes = temp affects deg)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('solution/shape_analysis.png', dpi=120)
    print("Saved plot to solution/shape_analysis.png — open it!")
    plt.show()