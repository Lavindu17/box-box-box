import json
import glob
import matplotlib.pyplot as plt
import numpy as np

print("Loading races...")
all_races = []
for filepath in sorted(glob.glob('data/historical_races/races_*.json'))[:10]:
    with open(filepath) as f:
        all_races.extend(json.load(f))
print(f"Loaded {len(all_races)} races")

# ── THE PERFECT CONTROL EXPERIMENT ──────────────────────────
# Find pairs in the same race where:
#   Driver A: SOFT  → X laps → [any tire T] → finishes
#   Driver B: MEDIUM → X laps → [same tire T] → finishes
#   AND both second stints are the same length
#
# Under these conditions, second stint cancels out entirely.
# Finishing rank difference = pure SOFT vs MEDIUM first stint signal.

print("\n── PERFECT CONTROL PAIRS ──")
controlled_pairs = []

for race in all_races:
    finishing  = race['finishing_positions']
    total_laps = race['race_config']['total_laps']
    strategies = list(race['strategies'].values())

    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            d1, d2 = strategies[i], strategies[j]

            # One must be SOFT starter, one MEDIUM starter
            if {d1['starting_tire'], d2['starting_tire']} != {'SOFT', 'MEDIUM'}:
                continue

            soft_d = d1 if d1['starting_tire'] == 'SOFT' else d2
            med_d  = d2 if d1['starting_tire'] == 'SOFT' else d1

            # Both must have exactly 1 pit stop (simple 2-stint race)
            if len(soft_d['pit_stops']) != 1 or len(med_d['pit_stops']) != 1:
                continue

            soft_pit_lap  = soft_d['pit_stops'][0]['lap']
            med_pit_lap   = med_d['pit_stops'][0]['lap']
            soft_tire2    = soft_d['pit_stops'][0]['to_tire']
            med_tire2     = med_d['pit_stops'][0]['to_tire']

            # CONTROL 1: same second compound
            if soft_tire2 != med_tire2:
                continue

            # CONTROL 2: same second stint length
            soft_stint2_len = total_laps - soft_pit_lap
            med_stint2_len  = total_laps - med_pit_lap
            if soft_stint2_len != med_stint2_len:
                continue

            # Everything after the first stint is now IDENTICAL
            # → finishing rank difference = pure first-stint effect
            soft_rank = finishing.index(soft_d['driver_id'])
            med_rank  = finishing.index(med_d['driver_id'])

            controlled_pairs.append({
                'soft_stint1': soft_pit_lap,      # how long on SOFT
                'med_stint1':  med_pit_lap,        # how long on MEDIUM
                'second_tire': soft_tire2,
                'second_len':  soft_stint2_len,
                'soft_ahead':  soft_rank < med_rank,
                'temp':        race['race_config']['track_temp'],
                'rank_diff':   med_rank - soft_rank  # positive = SOFT won
            })

print(f"Found {len(controlled_pairs)} perfectly controlled pairs")

if len(controlled_pairs) == 0:
    print("No pairs found — try relaxing CONTROL 2 to within ±1 lap")
    # Fallback: allow ±1 lap difference in second stint length
    for race in all_races:
        finishing  = race['finishing_positions']
        total_laps = race['race_config']['total_laps']
        strategies = list(race['strategies'].values())
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                d1, d2 = strategies[i], strategies[j]
                if {d1['starting_tire'], d2['starting_tire']} != {'SOFT', 'MEDIUM'}:
                    continue
                soft_d = d1 if d1['starting_tire'] == 'SOFT' else d2
                med_d  = d2 if d1['starting_tire'] == 'SOFT' else d1
                if len(soft_d['pit_stops']) != 1 or len(med_d['pit_stops']) != 1:
                    continue
                soft_tire2 = soft_d['pit_stops'][0]['to_tire']
                med_tire2  = med_d['pit_stops'][0]['to_tire']
                if soft_tire2 != med_tire2:
                    continue
                soft_pit_lap = soft_d['pit_stops'][0]['lap']
                med_pit_lap  = med_d['pit_stops'][0]['lap']
                soft_stint2  = total_laps - soft_pit_lap
                med_stint2   = total_laps - med_pit_lap
                # RELAXED: within 1 lap
                if abs(soft_stint2 - med_stint2) > 1:
                    continue
                soft_rank = finishing.index(soft_d['driver_id'])
                med_rank  = finishing.index(med_d['driver_id'])
                controlled_pairs.append({
                    'soft_stint1': soft_pit_lap,
                    'med_stint1':  med_pit_lap,
                    'second_tire': soft_tire2,
                    'second_len':  soft_stint2,
                    'soft_ahead':  soft_rank < med_rank,
                    'temp':        race['race_config']['track_temp'],
                    'rank_diff':   med_rank - soft_rank
                })
    print(f"With ±1 lap fallback: {len(controlled_pairs)} pairs")

# ── ANALYSIS ─────────────────────────────────────────────────
if controlled_pairs:
    stint_lens = [s['soft_stint1'] for s in controlled_pairs]
    print(f"First stint range: {min(stint_lens)} – {max(stint_lens)} laps")

    # Second tire distribution
    tire2_counts = {}
    for p in controlled_pairs:
        t = p['second_tire']
        tire2_counts[t] = tire2_counts.get(t, 0) + 1
    print(f"Second tire distribution: {tire2_counts}")

    # ── PLOT ─────────────────────────────────────────────────
    bins = range(5, 75, 5)

    x_vals, y_vals, n_vals = [], [], []
    for b in bins:
        group = [p for p in controlled_pairs
                 if b <= p['soft_stint1'] < b + 5]
        if len(group) >= 5:
            rate = sum(p['soft_ahead'] for p in group) / len(group)
            x_vals.append(b + 2.5)
            y_vals.append(rate)
            n_vals.append(len(group))

    cool = [p for p in controlled_pairs if p['temp'] < 30]
    hot  = [p for p in controlled_pairs if p['temp'] >= 35]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left — win rate by stint length
    bars = axes[0].bar(x_vals, y_vals, width=4,
                       color='#E8593C', alpha=0.85, edgecolor='white')
    axes[0].axhline(0.5, color='gray', linestyle='--', label='50/50')
    for bar, n in zip(bars, n_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'n={n}', ha='center', va='bottom', fontsize=8)
    axes[0].set_xlabel('First stint length on SOFT (laps)')
    axes[0].set_ylabel('SOFT win rate vs MEDIUM')
    axes[0].set_title('Pure SOFT vs MEDIUM — second stint controlled out\n'
                      'Crossover lap = where SOFT stops being worth it')
    axes[0].set_ylim(0, 0.85)
    axes[0].legend()

    # Right — temperature split
    for group, label, color in [
        (cool, 'Cool (<30°C)', '#3B8BD4'),
        (hot,  'Hot (≥35°C)', '#E8593C')
    ]:
        xv, yv = [], []
        for b in bins:
            g = [p for p in group if b <= p['soft_stint1'] < b + 5]
            if len(g) >= 5:
                xv.append(b + 2.5)
                yv.append(sum(p['soft_ahead'] for p in g) / len(g))
        if xv:
            axes[1].plot(xv, yv, marker='o', label=label,
                         color=color, linewidth=2, markersize=7)

    axes[1].axhline(0.5, color='gray', linestyle='--')
    axes[1].set_xlabel('First stint length on SOFT (laps)')
    axes[1].set_ylabel('SOFT win rate vs MEDIUM')
    axes[1].set_title('Same signal split by temperature\n'
                      'Lines diverging = temp changes degradation rate')
    axes[1].set_ylim(0.2, 0.8)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('solution/shape_analysis_v3.png', dpi=120)
    print("\nSaved to solution/shape_analysis_v3.png")
    plt.show()

    # ── CROSSOVER TABLE ──────────────────────────────────────
    print("\n── CROSSOVER TABLE (pure signal) ──")
    for b in bins:
        group = [p for p in controlled_pairs
                 if b <= p['soft_stint1'] < b + 5]
        if len(group) >= 5:
            rate = sum(p['soft_ahead'] for p in group) / len(group)
            flag = " ← SOFT starts losing here" if rate < 0.5 else ""
            print(f"  Stint {b:2d}–{b+4:2d} laps: "
                  f"SOFT wins {rate:.1%}  (n={len(group)}){flag}")