#!/usr/bin/env python3
"""
Steam Factory Level Generator v2
With all mechanics: buttons, gates, valves, conveyors, teleporters, one-way doors
"""

import json
import random
import time
from collections import deque
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple, Dict, FrozenSet

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
DIR_NAMES = ['up', 'down', 'left', 'right']

# Tile types
# # = wall
# . = floor
# S = start
# X = exit
# B = barrel
# O = button (momentary pressure plate)
# G = gate (opens when button pressed)
# V = valve (deadly steam, seal with barrel)
# > < ^ v = conveyors
# 1 2 = teleporter pair
# ) ( = one-way doors (right-only, left-only)
# [ ] = one-way doors (down-only, up-only)  -- using brackets to avoid arrow confusion

class Level:
    def __init__(self, w: int, h: int, grid: List[List[str]] = None):
        self.w = w
        self.h = h
        self.grid = grid if grid else [['.' for _ in range(w)] for _ in range(h)]
        self.entry = None
        self.exit = None

    def get(self, x: int, y: int) -> str:
        if x < 0 or y < 0 or x >= self.w or y >= self.h:
            return '#'
        return self.grid[y][x]

    def set(self, x: int, y: int, c: str):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.grid[y][x] = c

    def to_strings(self) -> List[str]:
        return [''.join(row) for row in self.grid]

    def find_all(self, char: str) -> List[Tuple[int, int]]:
        """Find all cells with given character"""
        result = []
        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y][x] == char:
                    result.append((x, y))
        return result

    def floor_cells(self) -> List[Tuple[int, int]]:
        """Get all empty floor cells"""
        return [(x, y) for y in range(1, self.h - 1)
                for x in range(1, self.w - 1) if self.grid[y][x] == '.']


def create_bordered_room(w: int, h: int) -> Level:
    lv = Level(w, h)
    for x in range(w):
        lv.set(x, 0, '#')
        lv.set(x, h - 1, '#')
    for y in range(h):
        lv.set(0, y, '#')
        lv.set(w - 1, y, '#')
    return lv


def create_random_maze(w: int, h: int) -> Level:
    lv = create_bordered_room(w, h)
    density = 0.2 + random.random() * 0.15
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if random.random() < density:
                lv.set(x, y, '#')
    return lv


def place_entry_exit(lv: Level) -> bool:
    """Place entry at top, exit at bottom"""
    entry_x = 1 + random.randint(0, max(0, lv.w - 3))
    exit_x = 1 + random.randint(0, max(0, lv.w - 3))

    # Ensure they're offset
    if abs(entry_x - exit_x) < 2:
        exit_x = max(1, min(lv.w - 2, lv.w - 1 - entry_x))

    lv.set(entry_x, 0, 'S')
    lv.set(entry_x, 1, '.')
    lv.set(exit_x, lv.h - 1, 'X')
    lv.set(exit_x, lv.h - 2, '.')
    lv.entry = (entry_x, 0)
    lv.exit = (exit_x, lv.h - 1)
    return True


def place_barrel(lv: Level) -> bool:
    """Place a barrel on a floor cell"""
    cells = lv.floor_cells()
    if not cells:
        return False
    x, y = random.choice(cells)
    lv.set(x, y, 'B')
    return True


def place_button_gate(lv: Level) -> bool:
    """Place a button and gate pair"""
    cells = lv.floor_cells()
    if len(cells) < 2:
        return False

    # Button in upper half, gate in lower half
    upper = [(x, y) for x, y in cells if y < lv.h // 2]
    lower = [(x, y) for x, y in cells if y >= lv.h // 2]

    if not upper or not lower:
        return False

    bx, by = random.choice(upper)
    gx, gy = random.choice(lower)

    lv.set(bx, by, 'O')  # Button
    lv.set(gx, gy, 'G')  # Gate
    return True


def place_valve(lv: Level) -> bool:
    """Place a valve (deadly steam)"""
    cells = lv.floor_cells()
    if not cells:
        return False
    x, y = random.choice(cells)
    lv.set(x, y, 'V')
    return True


def place_conveyor(lv: Level) -> bool:
    """Place a conveyor belt"""
    cells = lv.floor_cells()
    if not cells:
        return False
    x, y = random.choice(cells)
    direction = random.choice(['>', '<', '^', 'v'])
    lv.set(x, y, direction)
    return True


def place_teleporter_pair(lv: Level) -> bool:
    """Place teleporters to enable barrel teleportation to button.

    Strategy: Put teleporter 1 in barrel's push path, teleporter 2 near button
    so barrel exits teleporter and slides onto button.
    """
    barrels = lv.find_all('B')
    buttons = lv.find_all('O')

    # If we have barrel + button, try smart placement
    if barrels and buttons:
        bx, by = barrels[0]
        ox, oy = buttons[0]

        # Try each direction for barrel push
        for dx, dy in DIRS:
            # Find where to put teleporter 1 (in barrel's path)
            t1x, t1y = bx + dx, by + dy
            # Skip if that's a wall or other object
            if lv.get(t1x, t1y) != '.':
                continue

            # Find where to put teleporter 2 (so barrel exits toward button)
            # Barrel will continue in same direction after teleport
            # So t2 should be positioned such that (t2 + direction) leads to button

            # Check if button is reachable from some position going in direction (dx, dy)
            # Place t2 such that sliding from t2 in direction (dx, dy) hits button

            # Work backwards from button
            check_x, check_y = ox - dx, oy - dy
            # Find a valid t2 position (must be floor, not too close to t1)
            found_t2 = False
            for dist in range(1, 6):
                t2x = ox - dx * dist
                t2y = oy - dy * dist
                if lv.get(t2x, t2y) == '.' and abs(t1x - t2x) + abs(t1y - t2y) >= 3:
                    # Verify path from t2 to button is clear
                    clear = True
                    cx, cy = t2x + dx, t2y + dy
                    while (cx, cy) != (ox, oy):
                        if lv.get(cx, cy) not in '.O':
                            clear = False
                            break
                        cx, cy = cx + dx, cy + dy
                    if clear:
                        lv.set(t1x, t1y, '1')
                        lv.set(t2x, t2y, '2')
                        return True

    # Fallback: random placement
    cells = lv.floor_cells()
    if len(cells) < 2:
        return False

    random.shuffle(cells)
    x1, y1 = cells[0]
    x2, y2 = cells[1]

    if abs(x1 - x2) + abs(y1 - y2) < 4:
        return False

    lv.set(x1, y1, '1')
    lv.set(x2, y2, '2')
    return True


def place_oneway_door(lv: Level) -> bool:
    """Place a one-way door"""
    cells = lv.floor_cells()
    if not cells:
        return False
    x, y = random.choice(cells)
    # ) = can only enter from left (pass right)
    # ( = can only enter from right (pass left)
    # ] = can only enter from top (pass down)
    # [ = can only enter from bottom (pass up)
    direction = random.choice([')', '(', ']', '['])
    lv.set(x, y, direction)
    return True


# ============ SOLVER ============

@dataclass(frozen=True)
class State:
    """Immutable game state for BFS"""
    pos: Tuple[int, int]
    barrels: FrozenSet[Tuple[int, int]]
    sealed_valves: FrozenSet[Tuple[int, int]]

    def key(self):
        return (self.pos, self.barrels, self.sealed_valves)


def solve_level(lv: Level, max_depth: int = 60) -> Optional[Dict]:
    """Solve level using BFS with all mechanics"""
    if not lv.entry or not lv.exit:
        return None

    # Find all objects
    barrels = frozenset(lv.find_all('B'))
    buttons = set(lv.find_all('O'))
    gates = set(lv.find_all('G'))
    valves = set(lv.find_all('V'))
    teleporter1 = lv.find_all('1')
    teleporter2 = lv.find_all('2')
    teleporters = {}
    if teleporter1 and teleporter2:
        teleporters[teleporter1[0]] = teleporter2[0]
        teleporters[teleporter2[0]] = teleporter1[0]

    start_state = State(pos=lv.entry, barrels=barrels, sealed_valves=frozenset())
    visited = set()
    # Queue: (state, moves, path, directions)
    queue = deque([(start_state, 0, [lv.entry], [])])

    while queue:
        state, moves, path, dirs = queue.popleft()

        if moves > max_depth:
            continue

        # Win check
        if state.pos == lv.exit:
            return {'moves': moves, 'solution': dirs, 'path': path}

        if state.key() in visited:
            continue
        visited.add(state.key())

        # Check if gate is open
        gate_open = len(gates) == 0
        if not gate_open:
            # Check if player or any barrel is on a button
            if state.pos in buttons:
                gate_open = True
            else:
                for b in state.barrels:
                    if b in buttons:
                        gate_open = True
                        break

        # Try each direction
        for di, (dx, dy) in enumerate(DIRS):
            result = simulate_move(lv, state, dx, dy, buttons, gates, valves,
                                   teleporters, gate_open)
            if result is None:
                continue  # Death or invalid

            new_state = result
            if new_state.key() not in visited:
                queue.append((new_state, moves + 1, path + [new_state.pos], dirs + [DIR_NAMES[di]]))

    return None


def simulate_move(lv: Level, state: State, dx: int, dy: int,
                  buttons: Set, gates: Set, valves: Set,
                  teleporters: Dict, gate_open: bool) -> Optional[State]:
    """Simulate a move, return new state or None if death"""
    x, y = state.pos
    barrels = set(state.barrels)
    sealed = set(state.sealed_valves)

    started_on_button = (x, y) in buttons
    left_button = False

    for step in range(50):
        nx, ny = x + dx, y + dy
        cell = lv.get(nx, ny)

        # Mid-slide gate closing check
        if started_on_button and not left_button and (x, y) not in buttons:
            left_button = True
            # Check if any barrel still on button
            barrel_on_button = any(b in buttons for b in barrels)
            if not barrel_on_button:
                gate_open = False

        # Wall stops movement
        if cell == '#':
            break

        # Closed gate stops movement
        if (nx, ny) in gates and not gate_open:
            break

        # One-way door check
        if cell == ')' and dx != 1:  # Can only pass going right
            break
        if cell == '(' and dx != -1:  # Can only pass going left
            break
        if cell == ']' and dy != 1:  # Can only pass going down
            break
        if cell == '[' and dy != -1:  # Can only pass going up
            break

        # Barrel - try to push (but not if it's sealing a valve)
        if (nx, ny) in barrels and (nx, ny) not in sealed:
            push_result = push_barrel(lv, nx, ny, dx, dy, barrels, sealed,
                                       buttons, valves, gates, gate_open)
            if push_result is None:
                break  # Can't push, stop here
            barrels, sealed = push_result
            x, y = nx, ny
            break  # Stop after pushing

        # Barrel on sealed valve - blocks but can't push
        if (nx, ny) in barrels and (nx, ny) in sealed:
            break

        # Unsealed valve = death
        if (nx, ny) in valves and (nx, ny) not in sealed:
            return None

        x, y = nx, ny

        # Teleporter
        if (x, y) in teleporters:
            x, y = teleporters[(x, y)]
            # Continue sliding from teleporter destination

        # Button stops player
        if (x, y) in buttons:
            break

        # Exit stops player
        if (x, y) == lv.exit:
            break

        # Conveyor changes direction
        cur_cell = lv.get(x, y)
        if cur_cell == '>': dx, dy = 1, 0
        elif cur_cell == '<': dx, dy = -1, 0
        elif cur_cell == '^': dx, dy = 0, -1
        elif cur_cell == 'v': dx, dy = 0, 1

    # Check if we actually moved
    if (x, y) == state.pos:
        return None  # No movement = invalid

    return State(pos=(x, y), barrels=frozenset(barrels), sealed_valves=frozenset(sealed))


def push_barrel(lv: Level, bx: int, by: int, dx: int, dy: int,
                barrels: Set, sealed: Set, buttons: Set, valves: Set,
                gates: Set, gate_open: bool) -> Optional[Tuple[Set, Set]]:
    """Push a barrel, return (new_barrels, new_sealed) or None"""
    nx, ny = bx + dx, by + dy
    cell = lv.get(nx, ny)

    # Can't push into wall, closed gate, or another barrel
    if cell == '#':
        return None
    if (nx, ny) in gates and not gate_open:
        return None
    if (nx, ny) in barrels:
        return None

    # One-way doors block barrels too
    if cell == ')' and dx != 1:
        return None
    if cell == '(' and dx != -1:
        return None
    if cell == ']' and dy != 1:
        return None
    if cell == '[' and dy != -1:
        return None

    new_barrels = set(barrels)
    new_barrels.remove((bx, by))
    new_sealed = set(sealed)

    x, y = nx, ny
    while True:
        # Button stops barrel
        if (x, y) in buttons:
            new_barrels.add((x, y))
            return (new_barrels, new_sealed)

        # Valve seals when barrel hits it
        if (x, y) in valves and (x, y) not in new_sealed:
            new_sealed.add((x, y))
            new_barrels.add((x, y))
            return (new_barrels, new_sealed)

        # Teleporter - barrel warps through
        cell = lv.get(x, y)
        if cell == '1' or cell == '2':
            target_char = '2' if cell == '1' else '1'
            targets = lv.find_all(target_char)
            if targets:
                x, y = targets[0]

        # Check next cell
        next_x, next_y = x + dx, y + dy
        next_cell = lv.get(next_x, next_y)

        # Stop conditions
        if next_cell == '#':
            new_barrels.add((x, y))
            return (new_barrels, new_sealed)
        if (next_x, next_y) in new_barrels:
            new_barrels.add((x, y))
            return (new_barrels, new_sealed)
        if (next_x, next_y) in gates:
            new_barrels.add((x, y))
            return (new_barrels, new_sealed)

        x, y = next_x, next_y


def barrel_teleports_in_solution(lv: Level, solution_dirs: List[str]) -> bool:
    """Check if the solution requires barrel to pass through a teleporter.
    Returns True only if a barrel actually teleports during the solution."""

    DIRS_MAP = {'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0)}

    # Find all objects
    barrels = set(lv.find_all('B'))
    buttons = set(lv.find_all('O'))
    valves = set(lv.find_all('V'))
    gates = set(lv.find_all('G'))
    sealed = set()

    t1 = lv.find_all('1')
    t2 = lv.find_all('2')
    teleporters = {}
    if t1 and t2:
        teleporters[t1[0]] = t2[0]
        teleporters[t2[0]] = t1[0]

    if not teleporters or not barrels:
        return False

    # Find start position
    start = lv.find_all('S')
    if not start:
        return False
    px, py = start[0]

    barrel_teleported = False

    for move in solution_dirs:
        if move not in DIRS_MAP:
            continue
        dx, dy = DIRS_MAP[move]

        # Simulate player slide
        for _ in range(50):
            nx, ny = px + dx, py + dy
            cell = lv.get(nx, ny)

            if cell == '#':
                break

            # Check for barrel push
            if (nx, ny) in barrels and (nx, ny) not in sealed:
                # Try to push barrel
                bx, by = nx + dx, ny + dy
                if lv.get(bx, by) == '#' or (bx, by) in barrels:
                    break  # Can't push

                # Push the barrel
                barrels.remove((nx, ny))
                x, y = bx, by

                while True:
                    # Button stops barrel
                    if (x, y) in buttons:
                        barrels.add((x, y))
                        break
                    # Valve stops barrel
                    if (x, y) in valves and (x, y) not in sealed:
                        sealed.add((x, y))
                        barrels.add((x, y))
                        break
                    # TELEPORTER - barrel warps!
                    if (x, y) in teleporters:
                        barrel_teleported = True  # THIS IS WHAT WE'RE LOOKING FOR
                        x, y = teleporters[(x, y)]
                    # Check next cell
                    next_x, next_y = x + dx, y + dy
                    next_cell = lv.get(next_x, next_y)
                    if next_cell == '#' or (next_x, next_y) in barrels or (next_x, next_y) in gates:
                        barrels.add((x, y))
                        break
                    x, y = next_x, next_y

                px, py = nx, ny
                break

            if (nx, ny) in barrels:
                break

            px, py = nx, ny

            # Player teleporter
            if (px, py) in teleporters:
                px, py = teleporters[(px, py)]

            if (px, py) in buttons:
                break
            if lv.get(px, py) == 'X':
                break

    return barrel_teleported


def score_quality(solution: Dict, lv: Level) -> int:
    """Score level quality based on solution and mechanics"""
    if not solution:
        return 0

    dirs = solution['solution']
    score = len(dirs) * 10  # Base score from length

    # Direction changes
    changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
    score += changes * 5

    # Backtracking
    opp = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    backtracks = sum(1 for i in range(1, len(dirs)) if dirs[i] == opp.get(dirs[i-1], ''))
    score += backtracks * 15

    # Mechanic bonuses
    if lv.find_all('O'):  # Has button
        score += 80
    if lv.find_all('V'):  # Has valve
        score += 50
    if any(lv.find_all(c) for c in '><^v'):  # Has conveyor
        score += 30
    if lv.find_all('1'):  # Has teleporter
        score += 100
    if any(lv.find_all(c) for c in ')(['):  # Has one-way door
        score += 60

    return score


def generate_level(config: Dict) -> Optional[Dict]:
    """Generate a single level with given config"""
    w = config.get('width', 7)
    h = config.get('height', 11)
    min_moves = config.get('min_moves', 5)
    mechanics = config.get('mechanics', [])  # List of mechanics to include

    for attempt in range(300):
        lv = create_random_maze(w, h)
        place_entry_exit(lv)

        # Place requested mechanics
        placed = {'barrel': False, 'button': False, 'valve': False,
                  'conveyor': False, 'teleporter': False, 'oneway': False}

        if 'button' in mechanics:
            if place_button_gate(lv):
                placed['button'] = True
                # Need a barrel to hold the button
                place_barrel(lv)
                placed['barrel'] = True

        if 'valve' in mechanics:
            if place_valve(lv):
                placed['valve'] = True
                if not placed['barrel']:
                    place_barrel(lv)
                    placed['barrel'] = True

        if 'conveyor' in mechanics:
            if place_conveyor(lv):
                placed['conveyor'] = True

        if 'teleporter' in mechanics:
            if place_teleporter_pair(lv):
                placed['teleporter'] = True

        if 'oneway' in mechanics:
            if place_oneway_door(lv):
                placed['oneway'] = True

        # Always have at least a barrel if nothing else
        if not any(placed.values()):
            place_barrel(lv)

        # Solve
        solution = solve_level(lv)
        if not solution:
            continue

        if solution['moves'] < min_moves:
            continue

        quality = score_quality(solution, lv)

        return {
            'grid': lv.to_strings(),
            'width': w,
            'height': h,
            'par': solution['moves'],
            'solution': solution['solution'],
            'quality': quality,
            'mechanics': [k for k, v in placed.items() if v]
        }

    return None


def gen_one(args):
    """Generate one level (for multiprocessing)"""
    config, seed = args
    random.seed(seed)
    return generate_level(config)


def gen_batch(config: Dict, count: int) -> List[Dict]:
    """Generate batch of levels using multiprocessing"""
    args = [(config, random.randint(0, 10000000) + i) for i in range(count)]
    with Pool(cpu_count()) as pool:
        results = pool.map(gen_one, args)
    return [r for r in results if r]


if __name__ == '__main__':
    print(f"Steam Factory Generator v2")
    print(f"CPU cores: {cpu_count()}")

    # Test generation with all mechanics
    configs = [
        {'width': 6, 'height': 10, 'mechanics': ['button']},
        {'width': 7, 'height': 11, 'mechanics': ['button', 'valve']},
        {'width': 7, 'height': 12, 'mechanics': ['button', 'conveyor']},
        {'width': 8, 'height': 13, 'mechanics': ['button', 'teleporter']},
        {'width': 8, 'height': 14, 'mechanics': ['button', 'oneway']},
        {'width': 9, 'height': 15, 'mechanics': ['button', 'valve', 'conveyor', 'teleporter']},
    ]

    for config in configs:
        print(f"\n{config['width']}x{config['height']} with {config['mechanics']}:")
        start = time.time()
        batch = gen_batch(config, 100)
        elapsed = time.time() - start

        if batch:
            batch.sort(key=lambda x: -x['quality'])
            best = batch[0]
            print(f"  Generated {len(batch)} levels in {elapsed:.1f}s")
            print(f"  Best: quality {best['quality']}, par {best['par']}")
            for row in best['grid']:
                print(f"    {row}")
        else:
            print(f"  Failed to generate any levels")
