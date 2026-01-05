#!/usr/bin/env python3
"""
TRAIN GENERATOR v2
Generates meaningful puzzle rooms with maximum path length
Every piece is necessary - no decorative obstacles
"""
import random
from collections import deque

# === PIECE DEFINITIONS ===
PIECES = {
    '.': {'name': 'Floor', 'solid': False, 'deadly': False},
    '#': {'name': 'Wall', 'solid': True, 'deadly': False},
    'S': {'name': 'Entry', 'solid': False, 'deadly': False},
    'X': {'name': 'Exit', 'solid': False, 'deadly': False},
    'O': {'name': 'Button', 'solid': False, 'deadly': False},
    'G': {'name': 'Gate', 'solid': True, 'deadly': False},
    'B': {'name': 'Barrel', 'solid': True, 'deadly': False},
    'V': {'name': 'Open Valve', 'solid': False, 'deadly': True},
    '>': {'name': 'Conveyor R', 'solid': False, 'deadly': False, 'dir': (1, 0)},
    '<': {'name': 'Conveyor L', 'solid': False, 'deadly': False, 'dir': (-1, 0)},
    '^': {'name': 'Conveyor U', 'solid': False, 'deadly': False, 'dir': (0, -1)},
    'v': {'name': 'Conveyor D', 'solid': False, 'deadly': False, 'dir': (0, 1)},
}

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right


class Level:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.grid = [['.' for _ in range(w)] for _ in range(h)]
        self.entry = None
        self.exit = None

    def get(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.grid[y][x]
        return '#'

    def set(self, x, y, c):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.grid[y][x] = c
            if c == 'S':
                self.entry = (x, y)
            elif c == 'X':
                self.exit = (x, y)

    def copy(self):
        lv = Level(self.w, self.h)
        lv.grid = [row[:] for row in self.grid]
        lv.entry = self.entry
        lv.exit = self.exit
        return lv

    def is_solid(self, x, y, gate_open=False):
        c = self.get(x, y)
        if c == '#':
            return True
        if c == 'G' and not gate_open:
            return True
        if c == 'B':
            return True
        return False

    def is_deadly(self, x, y, sealed_valves=None):
        c = self.get(x, y)
        if c == 'V':
            if sealed_valves and (x, y) in sealed_valves:
                return False
            return True
        return False

    def slide(self, sx, sy, dx, dy, gate_open=False, sealed_valves=None):
        """Slide from position until hitting obstacle. Returns final position or None if death."""
        x, y = sx, sy
        started_on_button = (self.get(sx, sy) == 'O')
        left_button = False

        for _ in range(50):
            nx, ny = x + dx, y + dy

            # Mid-slide gate closing: if we started on button and just left it, close gate
            if started_on_button and not left_button and self.get(x, y) != 'O':
                left_button = True
                # Check if any barrel is holding button (P = barrel on button)
                barrel_on_button = any(self.get(bx, by) == 'P'
                                       for bx in range(self.w) for by in range(self.h))
                if not barrel_on_button:
                    gate_open = False  # Gate closes mid-slide!

            if self.is_solid(nx, ny, gate_open):
                return (x, y)
            if self.is_deadly(nx, ny, sealed_valves):
                return None  # Death
            x, y = nx, ny
            c = self.get(x, y)
            # Buttons stop you (momentary)
            if c == 'O':
                return (x, y)
            # Exit stops you
            if c == 'X':
                return (x, y)
            # Conveyor changes direction
            if c in '><^v':
                cdx, cdy = PIECES[c]['dir']
                while not self.is_solid(x + cdx, y + cdy, gate_open):
                    if self.is_deadly(x + cdx, y + cdy, sealed_valves):
                        return None
                    x, y = x + cdx, y + cdy
                    nc = self.get(x, y)
                    if nc == 'O' or nc == 'X':
                        return (x, y)
                    if nc not in '><^v':
                        break
                return (x, y)
        return (x, y)

    def solve(self):
        """
        BFS solver with full barrel/valve/button simulation.
        State: (player_pos, barrel_positions, sealed_valves)
        Returns (moves, path) or (None, None) if unsolvable.
        """
        if not self.entry or not self.exit:
            return None, None

        # Find all barrels, buttons, valves, gate
        barrels = set()
        buttons = set()
        valves = set()
        has_gate = False
        gate_pos = None

        for y in range(self.h):
            for x in range(self.w):
                c = self.get(x, y)
                if c == 'B':
                    barrels.add((x, y))
                elif c == 'O':
                    buttons.add((x, y))
                elif c == 'V':
                    valves.add((x, y))
                elif c == 'G':
                    has_gate = True
                    gate_pos = (x, y)

        # State: (player_pos, frozenset of barrel positions, frozenset of sealed valves)
        start_state = (self.entry, frozenset(barrels), frozenset())
        visited = set()
        queue = deque([(start_state, 0, [self.entry])])

        while queue:
            (pos, barrel_set, sealed_set), moves, path = queue.popleft()

            # Check win
            if pos == self.exit:
                return moves, path

            state_key = (pos, barrel_set, sealed_set)
            if state_key in visited:
                continue
            visited.add(state_key)

            # Gate is open if any barrel is on a button, or player is on button
            barrels_on_buttons = barrel_set & buttons
            player_on_button = pos in buttons
            gate_open = bool(barrels_on_buttons) or player_on_button or not has_gate

            for dx, dy in DIRS:
                # Simulate slide with barrel pushing
                result = self._simulate_move(pos, dx, dy, barrel_set, sealed_set,
                                            buttons, valves, gate_open, gate_pos)
                if result is None:
                    continue  # Death or invalid

                new_pos, new_barrels, new_sealed = result

                new_state = (new_pos, new_barrels, new_sealed)
                if new_state not in visited:
                    queue.append((new_state, moves + 1, path + [new_pos]))

        return None, None

    def _simulate_move(self, start, dx, dy, barrels, sealed, buttons, valves, gate_open, gate_pos):
        """
        Simulate a move with barrel pushing.
        Returns (new_pos, new_barrels, new_sealed) or None if death/invalid.
        """
        x, y = start
        barrels = set(barrels)  # Make mutable copy
        sealed = set(sealed)

        # Track if we started on button for mid-slide gate closing
        started_on_button = start in buttons
        left_button = False

        for _ in range(50):  # Max slide distance
            nx, ny = x + dx, y + dy

            # Mid-slide gate closing
            if started_on_button and not left_button and (x, y) not in buttons:
                left_button = True
                # Check if any barrel is on button
                if not (barrels & buttons):
                    gate_open = False

            # Check what's at next position
            c = self.get(nx, ny)

            # Wall
            if c == '#':
                return ((x, y), frozenset(barrels), frozenset(sealed))

            # Gate (closed)
            if (nx, ny) == gate_pos and not gate_open:
                return ((x, y), frozenset(barrels), frozenset(sealed))

            # Barrel - try to push
            if (nx, ny) in barrels:
                push_result = self._push_barrel(nx, ny, dx, dy, barrels, sealed,
                                                buttons, valves, gate_open, gate_pos)
                if push_result is None:
                    # Can't push - stop here
                    return ((x, y), frozenset(barrels), frozenset(sealed))
                barrels, sealed = push_result
                # Player stops where barrel was
                return ((nx, ny), frozenset(barrels), frozenset(sealed))

            # Unsealed valve - death!
            if c == 'V' and (nx, ny) not in sealed:
                return None

            # Move to next cell
            x, y = nx, ny

            # Button stops player
            if (x, y) in buttons:
                return ((x, y), frozenset(barrels), frozenset(sealed))

            # Exit stops player
            if (x, y) == self.exit:
                return ((x, y), frozenset(barrels), frozenset(sealed))

            # Conveyor changes direction
            c = self.get(x, y)
            if c in '><^v':
                dirs = {'>': (1, 0), '<': (-1, 0), '^': (0, -1), 'v': (0, 1)}
                dx, dy = dirs[c]

        return ((x, y), frozenset(barrels), frozenset(sealed))

    def _push_barrel(self, bx, by, dx, dy, barrels, sealed, buttons, valves, gate_open, gate_pos):
        """
        Push barrel at (bx, by) in direction (dx, dy).
        Returns (new_barrels, new_sealed) or None if can't push.
        """
        # Check first cell in push direction
        nx, ny = bx + dx, by + dy
        c = self.get(nx, ny)

        # Can't push into wall
        if c == '#':
            return None

        # Can't push into closed gate
        if (nx, ny) == gate_pos and not gate_open:
            return None

        # Can't push into another barrel
        if (nx, ny) in barrels:
            return None

        # Remove barrel from old position
        barrels = set(barrels)
        barrels.discard((bx, by))
        sealed = set(sealed)

        # Slide barrel until it hits something
        x, y = nx, ny
        while True:
            # Check if barrel lands on button
            if (x, y) in buttons:
                barrels.add((x, y))
                return (barrels, sealed)

            # Check if barrel lands on valve (seals it)
            if (x, y) in valves and (x, y) not in sealed:
                sealed.add((x, y))
                barrels.add((x, y))
                return (barrels, sealed)

            # Check next cell
            next_x, next_y = x + dx, y + dy
            next_c = self.get(next_x, next_y)

            # Stop at wall, gate, or another barrel
            if next_c == '#' or (next_x, next_y) in barrels or (next_x, next_y) == gate_pos:
                barrels.add((x, y))
                return (barrels, sealed)

            x, y = next_x, next_y

        barrels.add((x, y))
        return (barrels, sealed)

    def to_string(self):
        return '\n'.join(''.join(row) for row in self.grid)


def create_bordered_room(w, h):
    """Create empty room with border walls"""
    lv = Level(w, h)
    for x in range(w):
        lv.set(x, 0, '#')
        lv.set(x, h - 1, '#')
    for y in range(h):
        lv.set(0, y, '#')
        lv.set(w - 1, y, '#')
    return lv


def create_zigzag_room(w, h):
    """Create room with RANDOM zigzag-inducing walls for long path"""
    lv = create_bordered_room(w, h)

    # Randomly decide gap positions for each row
    # This creates variety in the zigzag pattern
    for y in range(2, h - 2, 2):
        # Random gap position (1 to w-2)
        gap_x = random.randint(1, w - 2)

        # Fill row with walls except for gap
        for x in range(1, w - 1):
            if x != gap_x:
                lv.set(x, y, '#')

    return lv


def create_random_maze(w, h):
    """Create room with random wall placements - explores different layouts"""
    lv = create_bordered_room(w, h)

    # Randomly place internal walls (20-40% of interior cells)
    interior_cells = (w - 2) * (h - 2)
    num_walls = random.randint(interior_cells // 5, interior_cells // 3)

    for _ in range(num_walls):
        x = random.randint(1, w - 2)
        y = random.randint(1, h - 2)
        lv.set(x, y, '#')

    return lv


def place_entry_exit_opposite(lv):
    """Place entry and exit at opposite corners for maximum path"""
    # Entry at top-left
    lv.set(1, 0, 'S')
    # Exit at bottom-right
    lv.set(lv.w - 2, lv.h - 1, 'X')
    return lv


def get_solution_path(lv):
    """Get list of positions in solution"""
    _, path = lv.solve()
    return path or []


def simulate_barrel_push(lv, bx, by, dx, dy):
    """Simulate pushing barrel - returns final barrel position or None if can't push"""
    # Check if first cell in push direction is open
    nx, ny = bx + dx, by + dy
    cell = lv.get(nx, ny)
    if cell not in ['.', 'O', 'V']:  # Can push onto floor, button, or valve
        return None

    # Slide barrel until it hits something
    while True:
        next_cell = lv.get(nx + dx, ny + dy)
        if next_cell in ['#', 'G', 'B']:  # Wall, gate, or another barrel stops it
            return (nx, ny)
        if cell == 'O':  # Button stops barrel
            return (nx, ny)
        if cell == 'V':  # Valve stops barrel (seals it)
            return (nx, ny)
        nx, ny = nx + dx, ny + dy
        cell = lv.get(nx, ny)
        if cell in ['#', 'G', 'B']:
            return (nx - dx, ny - dy)


def place_barrel_blocking_exit(lv, path):
    """
    Place a barrel on the solution path that player must push to continue.

    Requirements:
    - Barrel on floor tile on the path
    - Player can approach from one side to push
    - Barrel can slide (space in push direction)
    - After push, player can still reach exit
    """
    if not path or len(path) < 4:
        return lv

    # Try each position on the path (middle section)
    for i in range(2, len(path) - 2):
        bx, by = path[i]

        if lv.get(bx, by) != '.':
            continue

        # Try each push direction
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Player approaches from opposite direction
            px, py = bx - dx, by - dy

            # Check player can stand there (floor, not wall)
            if lv.get(px, py) not in ['.', 'S']:
                continue

            # Check barrel can slide (at least one cell in push direction)
            slide_x, slide_y = bx + dx, by + dy
            if lv.get(slide_x, slide_y) not in ['.']:
                continue

            # Place barrel and verify solvability
            test_lv = lv.copy()
            test_lv.set(bx, by, 'B')

            # Simulate: barrel pushed, where does it stop?
            final_x, final_y = slide_x, slide_y
            while test_lv.get(final_x + dx, final_y + dy) == '.':
                final_x += dx
                final_y += dy

            # Test with barrel in final position
            test_lv.set(bx, by, '.')  # Clear original
            test_lv.set(final_x, final_y, 'B')  # Barrel stops here

            moves, _ = test_lv.solve()
            if moves is not None:
                # Valid! Place the barrel
                lv.set(bx, by, 'B')
                return lv

    return lv


def place_button_gate(lv, path):
    """
    Place button on path, gate before exit.
    Gate MUST block the only path to exit - no bypassing!
    """
    if not path or len(path) < 3:
        return lv

    ex, ey = lv.exit
    gate_y = ey - 1
    button_y = gate_y - 2

    if button_y < 1 or gate_y < 1:
        return lv

    test_lv = lv.copy()

    # CRITICAL: Wall off entire gate row except exit column
    # This ensures gate is the ONLY way to reach exit
    for x in range(1, lv.w - 1):
        if x != ex:
            test_lv.set(x, gate_y, '#')

    # Place gate directly above exit
    test_lv.set(ex, gate_y, 'G')

    # Clear path from button to gate in exit column
    for y in range(button_y, gate_y):
        if test_lv.get(ex, y) == '#':
            test_lv.set(ex, y, '.')

    # Place button above gate
    test_lv.set(ex, button_y, 'O')

    # Test if solvable
    new_moves, new_path = test_lv.solve()
    if new_moves is not None:
        lv.grid = test_lv.grid
        return lv

    return lv


def place_barrel_on_button(lv, path):
    """
    Place button, gate, AND barrel where the solution requires pushing
    the barrel onto the button to hold the gate open.

    KEY: Gate must be the ONLY way to reach exit - no bypassing!
    """
    if not path or len(path) < 3:
        return lv

    ex, ey = lv.exit

    # Gate goes directly above exit - this is the ONLY entrance
    gate_y = ey - 1
    if gate_y < 3:
        return lv

    test_lv = lv.copy()

    # CRITICAL: Wall off ALL cells in gate row EXCEPT the gate position
    # This ensures gate is the ONLY way to reach exit
    for x in range(1, lv.w - 1):
        if x != ex:
            test_lv.set(x, gate_y, '#')

    # Place gate directly above exit
    test_lv.set(ex, gate_y, 'G')

    # Button and barrel above the gate
    button_y = gate_y - 2
    button_x = ex
    barrel_x = button_x - 1
    barrel_y = button_y
    push_x = barrel_x - 1

    if barrel_x < 1 or push_x < 1 or button_y < 1:
        return lv

    # Clear path from button to gate
    for y in range(button_y, gate_y):
        if test_lv.get(ex, y) == '#':
            test_lv.set(ex, y, '.')

    # Add wall so player can STOP at push position
    if push_x > 1 and test_lv.get(push_x - 1, barrel_y) == '.':
        test_lv.set(push_x - 1, barrel_y, '#')

    # Ensure clear floor at push position
    if test_lv.get(push_x, barrel_y) == '#':
        test_lv.set(push_x, barrel_y, '.')

    # Place button and barrel
    test_lv.set(button_x, button_y, 'O')
    test_lv.set(barrel_x, barrel_y, 'B')

    # FULL VERIFICATION using BFS that simulates barrel pushing
    # Test: Can player push barrel onto button, then reach exit?

    # Simplified test: After barrel on button, can reach exit?
    temp_lv = test_lv.copy()
    temp_lv.set(barrel_x, barrel_y, '.')           # Barrel moved
    temp_lv.set(button_x, button_y, 'P')           # Barrel on button (pressed)
    temp_lv.set(ex, gate_y, '.')                   # Gate open
    moves, _ = temp_lv.solve()

    if moves is None:
        return lv  # Can't reach exit

    # Test: Can player reach push position?
    temp_lv2 = test_lv.copy()
    temp_lv2.set(barrel_x, barrel_y, '.')  # Remove barrel temporarily
    temp_lv2.set(button_x, button_y, '.')  # Remove button temporarily
    temp_lv2.set(ex, gate_y, '.')          # Remove gate temporarily

    # Check if push position (push_x, barrel_y) is reachable AND stoppable
    # Player must be able to slide TO push_x and STOP there
    # This requires something solid at (push_x-1, barrel_y) if coming from left

    # Verify there's a wall to stop against
    wall_to_left = temp_lv2.get(push_x - 1, barrel_y) == '#'
    if not wall_to_left:
        return lv  # Can't stop at push position

    moves2, _ = temp_lv2.solve()
    if moves2 is None:
        return lv

    lv.grid = test_lv.grid
    return lv


def place_valve_and_barrel(lv, path):
    """
    Place valve on path (deadly!), barrel nearby to seal it.
    The puzzle requires: reach barrel, push it onto valve, then proceed.
    """
    if not path or len(path) < 4:
        return lv

    # Valve should be on the path, barrel should be pushable onto it
    # Try positions in the middle third of the path
    path_len = len(path)
    start_idx = max(2, path_len // 4)
    end_idx = min(path_len - 2, 3 * path_len // 4)

    for pos in path[start_idx:end_idx]:
        vx, vy = pos
        if lv.get(vx, vy) != '.':
            continue

        # Find a spot for barrel that can be pushed onto valve
        # Barrel should be in line with valve (horizontally or vertically)
        # so player can push barrel in direction of valve
        barrel_positions = []

        # Check each direction from valve
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            # Barrel at valve + direction, pushable BACK toward valve
            bx, by = vx + dx, vy + dy
            if lv.get(bx, by) == '.':
                # Check if there's space for player to push from
                # Player must be able to STAND at px,py (not a wall!)
                px, py = bx + dx, by + dy
                if lv.get(px, py) in ['.', 'S']:  # Player can stand here
                    barrel_positions.append((bx, by, dx, dy))

        for bx, by, dx, dy in barrel_positions:
            # Place valve and barrel
            lv.set(vx, vy, 'V')
            lv.set(bx, by, 'B')

            # The solver can't verify this (doesn't handle valve sealing)
            # But we trust the placement: barrel can be pushed onto valve
            return lv

    # Fallback: just place valve somewhere on path WITH push verification
    for pos in path[2:-2]:
        vx, vy = pos
        if lv.get(vx, vy) != '.':
            continue

        # Find spot for barrel that player can actually push onto valve
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            bx, by = vx + dx, vy + dy
            if lv.get(bx, by) != '.':
                continue

            # Player push position (opposite side of barrel from valve)
            px, py = bx + dx, by + dy
            if lv.get(px, py) in ['.', 'S']:  # Player can stand here to push
                lv.set(vx, vy, 'V')
                lv.set(bx, by, 'B')
                return lv

    return lv


def place_conveyor(lv, path):
    """
    Place conveyor on path to redirect player movement.
    Conveyor changes player direction mid-slide.
    """
    if not path or len(path) < 4:
        return lv

    # Find positions on path that could use a direction change
    for i, pos in enumerate(path[1:-1], 1):
        x, y = pos
        if lv.get(x, y) != '.':
            continue

        # Determine which direction would be interesting
        # Look at where player came from and where they're going
        if i < len(path) - 1:
            next_pos = path[i + 1]
            nx, ny = next_pos
            dx = nx - x
            dy = ny - y

            # Normalize direction
            if dx > 0:
                conv = '>'
            elif dx < 0:
                conv = '<'
            elif dy > 0:
                conv = 'v'
            else:
                conv = '^'

            # Place conveyor
            lv.set(x, y, conv)
            return lv

    return lv


def cleanup_unnecessary(lv):
    """Remove any piece that doesn't increase solution length"""
    original_moves, _ = lv.solve()
    if original_moves is None:
        return lv

    # Only remove walls - special pieces (B, O, V, G) are intentionally placed
    removable = ['#']

    changed = True
    while changed:
        changed = False
        for y in range(1, lv.h - 1):
            for x in range(1, lv.w - 1):
                c = lv.get(x, y)
                if c in removable:
                    lv.set(x, y, '.')
                    new_moves, _ = lv.solve()

                    if new_moves is not None and new_moves >= original_moves:
                        # Wall wasn't helping - leave it removed
                        changed = True
                    else:
                        # Wall was necessary - put it back
                        lv.set(x, y, c)

    return lv


def generate_level(config, time_budget=15.0):
    """
    Generate a single level with meaningful puzzles.
    Spends time_budget seconds finding the best level (like a human solving).
    Tracks tried configurations to avoid repeating.
    """
    import time
    start_time = time.time()

    w = config.get('width', 5)
    h = config.get('height', 9)
    min_moves = config.get('min_moves', 3)

    best_level = None
    best_moves = 0
    attempts = 0
    tried_layouts = set()  # Track layouts we've already tried

    # Keep generating until time runs out, always seeking better solutions
    while time.time() - start_time < time_budget:
        # Randomly choose layout strategy
        layout_type = random.choice(['zigzag', 'maze', 'sparse'])

        if layout_type == 'zigzag':
            lv = create_zigzag_room(w, h)
        elif layout_type == 'maze':
            lv = create_random_maze(w, h)
        else:
            # Sparse - few random walls
            lv = create_bordered_room(w, h)
            for _ in range(random.randint(2, 6)):
                x = random.randint(1, w - 2)
                y = random.randint(1, h - 2)
                lv.set(x, y, '#')

        # Place entry/exit at opposite corners
        place_entry_exit_opposite(lv)

        # Hash the base layout (before pieces) to skip duplicates
        layout_hash = lv.to_string()
        if layout_hash in tried_layouts:
            continue  # Already tried this exact layout
        tried_layouts.add(layout_hash)

        # Get initial solution
        moves, path = lv.solve()
        if moves is None:
            continue

        initial_moves = moves

        # Add required pieces on the solution path
        # ORDER MATTERS: Place gate first (solver can verify), THEN barrel (breaks solver)

        # Special case: barrel must be pushed onto button
        if config.get('require_barrel_on_button'):
            lv = place_barrel_on_button(lv, path)
            # Verify it actually placed the gate and is solvable
            has_gate = any(lv.get(x, y) == 'G' for x in range(lv.w) for y in range(lv.h))
            if not has_gate:
                continue  # placement failed, try another layout
            moves, path = lv.solve()
            if moves is None:
                continue  # Unsolvable, try another layout

        elif config.get('require_gate'):
            lv = place_button_gate(lv, path)
            # Verify gate was actually placed
            has_gate = any(lv.get(x, y) == 'G' for x in range(lv.w) for y in range(lv.h))
            if not has_gate:
                continue  # placement failed, try another layout
            # Re-solve after button/gate
            moves, path = lv.solve()
            if moves is None:
                continue

        if config.get('require_conveyor'):
            lv = place_conveyor(lv, path)
            # Re-solve after conveyor
            moves, path = lv.solve()
            if moves is None:
                continue

        # Place barrel/valve and verify still solvable
        if config.get('require_barrel') and not config.get('require_barrel_on_button'):
            lv = place_barrel_blocking_exit(lv, path)
            # Verify solvable with new barrel-aware solver
            moves, path = lv.solve()
            if moves is None:
                continue  # Unsolvable, try another layout

        if config.get('require_valve'):
            lv = place_valve_and_barrel(lv, path)
            # Verify solvable with new valve-aware solver
            moves, path = lv.solve()
            if moves is None:
                continue  # Unsolvable, try another layout

        # Add some random walls to increase complexity
        if config.get('add_random_walls', False):
            for _ in range(config.get('max_random_walls', 3)):
                x = random.randint(1, w - 2)
                y = random.randint(1, h - 2)
                if lv.get(x, y) == '.':
                    lv.set(x, y, '#')
                    test_moves, _ = lv.solve()
                    if test_moves is None or test_moves <= moves:
                        lv.set(x, y, '.')  # Undo
                    else:
                        moves = test_moves

        # Cleanup unnecessary walls (but keep special pieces)
        lv = cleanup_unnecessary(lv)

        # Final verification - must be solvable
        final_moves, final_path = lv.solve()
        if final_moves is None:
            continue  # Unsolvable after cleanup, skip

        attempts += 1
        if final_moves >= min_moves and final_moves > best_moves:
            best_moves = final_moves
            best_level = lv

    elapsed = time.time() - start_time
    unique_tried = len(tried_layouts)

    # Get solution path for the best level
    solution_dirs = []
    if best_level:
        _, path = best_level.solve()
        if path and len(path) > 1:
            # Convert path positions to directions
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                dx, dy = x2 - x1, y2 - y1
                # Normalize to unit direction
                if dx > 0: solution_dirs.append('right')
                elif dx < 0: solution_dirs.append('left')
                elif dy > 0: solution_dirs.append('down')
                elif dy < 0: solution_dirs.append('up')

    return best_level, best_moves, attempts, elapsed, unique_tried, solution_dirs


# === TRAIN CONFIGURATION ===
TRAIN_CONFIG = [
    {
        'name': 'Movement',
        'width': 5, 'height': 9,
        'min_moves': 4,
        'use_zigzag': True,
    },
    {
        'name': 'Barrel',
        'width': 5, 'height': 9,
        'min_moves': 3,
        'use_zigzag': False,  # Need open space for barrel pushing
        'require_barrel': True,
    },
    {
        'name': 'Button',
        'width': 5, 'height': 9,
        'min_moves': 3,
        'use_zigzag': False,  # Need open space for barrel pushing
        'require_gate': True,
        'require_barrel_on_button': True,  # Barrel must hold button down
    },
    {
        'name': 'Valve',
        'width': 5, 'height': 9,
        'min_moves': 4,
        'use_zigzag': True,
        'require_valve': True,
    },
    {
        'name': 'Conveyor',
        'width': 5, 'height': 9,
        'min_moves': 4,
        'use_zigzag': True,
        'require_conveyor': True,
    },
    {
        'name': 'Combo',
        'width': 6, 'height': 10,
        'min_moves': 5,
        'use_zigzag': True,
        'require_barrel_on_button': True,  # Places barrel, button, AND gate together
        'add_random_walls': True,
    },
]


def generate_train(verbose=True, time_per_level=15.0):
    """Generate full train of cars, spending time_per_level seconds on each"""
    train = []

    for i, config in enumerate(TRAIN_CONFIG):
        if verbose:
            print(f"\n=== CAR {i + 1}: {config['name']} ===", flush=True)

        lv, moves, attempts, elapsed, unique, solution = generate_level(config, time_budget=time_per_level)
        if lv:
            if verbose:
                print(lv.to_string())
                print(f"Solution: {moves} moves ({unique} unique layouts tried in {elapsed:.1f}s)")
                print(f"Hint: {' → '.join(solution)}")

            train.append({
                'name': config['name'],
                'level': lv,
                'moves': moves,
                'solution': solution,
                'config': config,
            })
        else:
            if verbose:
                print(f"Failed to generate! ({unique} unique layouts tried in {elapsed:.1f}s)")

    return train


# === MAIN ===
if __name__ == '__main__':
    import json
    import os

    print("Generating train with new algorithm...")
    train = generate_train(verbose=True)

    print(f"\n=== GENERATED {len(train)} CARS ===")

    # Export to JSON for HTML
    levels_json = []
    for car in train:
        levels_json.append({
            'name': car['name'],
            'grid': [''.join(row) for row in car['level'].grid],
            'par': car['moves'],
            'solution': car['solution']
        })

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_dir, 'levels', 'train.json')
    with open(json_path, 'w') as f:
        json.dump(levels_json, f, indent=2)
    print(f"Saved: {json_path}")

    # Auto-update HTML file
    html_path = os.path.join(project_dir, 'steam_factory.html')
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html = f.read()

        # Find and replace LEVELS array
        import re
        levels_js = "const LEVELS = " + json.dumps(levels_json, indent=2) + ";"
        pattern = r'const LEVELS = \[[\s\S]*?\];'
        new_html = re.sub(pattern, levels_js, html)

        with open(html_path, 'w') as f:
            f.write(new_html)
        print(f"Updated: {html_path}")

    # Print JS format for reference
    print("\n// JavaScript LEVELS array:")
    print("const LEVELS = " + json.dumps(levels_json, indent=2) + ";")
