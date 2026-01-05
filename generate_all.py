#!/usr/bin/env python3
"""
Full level generation with all mechanics
"""

import json
import random
import time
from multiprocessing import Pool, cpu_count
from generator_v2 import generate_level

def gen_one(args):
    config, seed = args
    random.seed(seed)
    return generate_level(config)

def gen_batch(config, count):
    args = [(config, random.randint(0, 10000000) + i) for i in range(count)]
    with Pool(cpu_count()) as pool:
        results = pool.map(gen_one, args)
    return [r for r in results if r]

def main():
    print(f"Steam Factory Full Generation")
    print(f"CPU cores: {cpu_count()}")
    print()

    all_levels = []
    start = time.time()

    # Size combinations from 5x9 to 11x17
    sizes = []
    for w in range(5, 12):
        for h in range(9, 18):
            if h > w + 3 and h < w * 3:  # Reasonable aspect ratios
                sizes.append((w, h))

    # Mechanic combinations (progressive difficulty)
    mechanic_sets = [
        [],  # Pure movement
        ['button'],  # Button + gate
        ['valve'],  # Valve (needs barrel)
        ['conveyor'],  # Conveyor
        ['button', 'valve'],
        ['button', 'conveyor'],
        ['teleporter'],  # New: teleporter
        ['oneway'],  # New: one-way doors
        ['button', 'teleporter'],
        ['button', 'oneway'],
        ['valve', 'teleporter'],
        ['button', 'valve', 'conveyor'],
        ['button', 'valve', 'teleporter'],
        ['button', 'valve', 'oneway'],
        ['button', 'conveyor', 'teleporter'],
        ['button', 'valve', 'conveyor', 'teleporter'],
        ['button', 'valve', 'conveyor', 'teleporter', 'oneway'],  # Everything!
    ]

    candidates_per_config = 500

    for mech_set in mechanic_sets:
        mech_name = '+'.join(mech_set) if mech_set else 'pure'
        print(f"\n=== Mechanics: {mech_name} ===")

        for w, h in sizes:
            config = {
                'width': w,
                'height': h,
                'min_moves': 5,
                'mechanics': mech_set
            }

            batch = gen_batch(config, candidates_per_config)

            if batch:
                batch.sort(key=lambda x: -x['quality'])
                best = batch[:20]  # Keep top 20 per config
                all_levels.extend(best)
                top = batch[0]
                print(f"  {w}x{h}: {len(batch)} levels, best quality {top['quality']}, par {top['par']}")
            else:
                print(f"  {w}x{h}: 0 levels")

    elapsed = time.time() - start
    print(f"\n=== COMPLETE ===")
    print(f"Total levels: {len(all_levels)}")
    print(f"Time: {elapsed:.1f}s")

    # Sort by difficulty (par) and quality
    all_levels.sort(key=lambda x: (x['par'], -x['quality']))

    # Save all levels
    with open('levels_v2_all.json', 'w') as f:
        json.dump(all_levels, f)

    # Create compact version for the game
    # Take top levels per difficulty tier
    difficulty_tiers = {}
    for lv in all_levels:
        par = lv['par']
        tier = par // 5 * 5  # Group by 5s
        if tier not in difficulty_tiers:
            difficulty_tiers[tier] = []
        difficulty_tiers[tier].append(lv)

    compact = []
    for tier in sorted(difficulty_tiers.keys()):
        levels = difficulty_tiers[tier]
        levels.sort(key=lambda x: -x['quality'])
        compact.extend(levels[:15])  # 15 per tier

    # Sort by difficulty
    compact.sort(key=lambda x: (x['par'], -x['quality']))

    # Save compact version
    with open('levels_compact.json', 'w') as f:
        json.dump(compact, f)

    print(f"Compact levels: {len(compact)}")

    # Show some stats
    mech_counts = {}
    for lv in compact:
        for m in lv.get('mechanics', []):
            mech_counts[m] = mech_counts.get(m, 0) + 1

    print("\nMechanic distribution in compact set:")
    for m, c in sorted(mech_counts.items()):
        print(f"  {m}: {c}")

if __name__ == '__main__':
    main()
