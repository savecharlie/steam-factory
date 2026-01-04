#!/usr/bin/env python3
"""
LEVEL VALIDATOR
- Removes useless pieces
- Checks for unintended solutions
- Ensures puzzle integrity
"""
from collections import deque
from game_pieces import PIECES, CHAR_TO_PIECE

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

class Level:
    def __init__(self, grid):
        self.grid = [list(row) for row in grid]
        self.h = len(grid)
        self.w = len(grid[0]) if self.h > 0 else 0
        self.find_special()

    def find_special(self):
        """Find entry, exit, buttons, gates, valves, barrels"""
        self.entry = self.exit = None
        self.buttons = []
        self.gates = []
        self.valves = []
        self.barrels = []

        for y in range(self.h):
            for x in range(self.w):
                c = self.grid[y][x]
                if c == 'S': self.entry = (x, y)
                elif c == 'X': self.exit = (x, y)
                elif c == 'O': self.buttons.append((x, y))
                elif c == 'G': self.gates.append((x, y))
                elif c == 'V': self.valves.append((x, y))
                elif c == 'B': self.barrels.append((x, y))

    def get(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            return self.grid[y][x]
        return '#'

    def set(self, x, y, val):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.grid[y][x] = val

    def copy(self):
        return Level([row[:] for row in self.grid])

    def is_solid(self, x, y, buttons_pressed=False, valves_sealed=None):
        """Check if cell blocks movement"""
        c = self.get(x, y)
        if c == '#': return True
        if c == 'G' and not buttons_pressed: return True  # Gate closed
        if c == 'B': return True  # Barrel blocks until pushed
        return False

    def is_deadly(self, x, y, valves_sealed=None):
        """Check if cell kills player"""
        c = self.get(x, y)
        if c == 'V':
            # Deadly unless sealed
            if valves_sealed and (x, y) in valves_sealed:
                return False
            return True
        return False

    def slide(self, sx, sy, dx, dy, buttons_pressed=False, valves_sealed=None):
        """Slide from position until hitting wall/obstacle"""
        x, y = sx, sy
        for _ in range(50):
            nx, ny = x + dx, y + dy
            if self.is_solid(nx, ny, buttons_pressed, valves_sealed):
                return (x, y)
            if self.is_deadly(nx, ny, valves_sealed):
                return None  # Death
            x, y = nx, ny
            # Conveyor handling
            c = self.get(x, y)
            if c in '><^v':
                cdirs = {'>': (1,0), '<': (-1,0), '^': (0,-1), 'v': (0,1)}
                cdx, cdy = cdirs[c]
                while not self.is_solid(x+cdx, y+cdy, buttons_pressed, valves_sealed):
                    if self.is_deadly(x+cdx, y+cdy, valves_sealed):
                        return None
                    x, y = x + cdx, y + cdy
                    nc = self.get(x, y)
                    if nc not in '><^v':
                        break
                return (x, y)
        return (x, y)

    def can_reach(self, start, goal, buttons_pressed=False, valves_sealed=None):
        """BFS: can we reach goal from start?"""
        if not start or not goal:
            return False
        visited = set()
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            if pos == goal:
                return True
            if pos in visited or pos is None:
                continue
            visited.add(pos)

            for dx, dy in DIRS:
                end = self.slide(pos[0], pos[1], dx, dy, buttons_pressed, valves_sealed)
                if end and end not in visited:
                    queue.append(end)
        return False

    def path_length(self, start, goal, buttons_pressed=False, valves_sealed=None):
        """BFS: shortest path length"""
        if not start or not goal:
            return -1
        visited = {}
        queue = deque([(start, 0)])

        while queue:
            pos, dist = queue.popleft()
            if pos == goal:
                return dist
            if pos in visited or pos is None:
                continue
            visited[pos] = dist

            for dx, dy in DIRS:
                end = self.slide(pos[0], pos[1], dx, dy, buttons_pressed, valves_sealed)
                if end and end not in visited:
                    queue.append((end, dist + 1))
        return -1


def validate_no_shortcuts(level):
    """
    Ensure puzzle can't be bypassed:
    - Can't reach exit without pressing button (if button/gate exists)
    - Can't avoid using barrels for valves (if valves block path)
    """
    issues = []

    # Check 1: If there's a button/gate, can we reach exit WITHOUT pressing button?
    if level.buttons and level.gates:
        # Try reaching exit with gate CLOSED
        can_skip_button = level.can_reach(level.entry, level.exit, buttons_pressed=False)
        if can_skip_button:
            issues.append("SHORTCUT: Can reach exit without pressing button!")

    # Check 2: If valves exist on path, can we reach exit without sealing them?
    if level.valves:
        # Check if any valve is between entry and exit
        can_skip_valves = level.can_reach(level.entry, level.exit, buttons_pressed=True, valves_sealed=None)
        if can_skip_valves:
            # Valves don't block the path - might be okay if they're optional hazards
            pass
        else:
            # Valves DO block - check if we have enough barrels
            if len(level.barrels) < len(level.valves):
                issues.append(f"UNSOLVABLE: {len(level.valves)} valves but only {len(level.barrels)} barrels")

    # Check 3: Basic solvability - can reach button, then exit?
    if level.buttons:
        can_reach_button = level.can_reach(level.entry, level.buttons[0], buttons_pressed=False)
        if not can_reach_button:
            issues.append("UNSOLVABLE: Can't reach button from entry")
        else:
            can_reach_exit = level.can_reach(level.entry, level.exit, buttons_pressed=True)
            if not can_reach_exit:
                issues.append("UNSOLVABLE: Can't reach exit even with button pressed")

    return issues


def find_useless_pieces(level):
    """Find pieces that don't affect the puzzle"""
    useless = []
    original_path = level.path_length(level.entry, level.exit, buttons_pressed=True)

    for y in range(1, level.h - 1):
        for x in range(1, level.w - 1):
            c = level.get(x, y)

            # Skip special pieces
            if c in '.SXOGBVv^<>':
                continue

            # Only test walls and decorative pieces
            if c == '#':
                # Try removing this wall
                test = level.copy()
                test.set(x, y, '.')

                new_path = test.path_length(test.entry, test.exit, buttons_pressed=True)

                # If path length unchanged, wall was useless
                if new_path >= 0 and new_path <= original_path:
                    useless.append((x, y, c, "doesn't affect path"))

    return useless


def cleanup_level(level):
    """Remove useless pieces, return cleaned level"""
    cleaned = level.copy()
    removed = []

    while True:
        useless = find_useless_pieces(cleaned)
        if not useless:
            break

        for x, y, c, reason in useless:
            cleaned.set(x, y, '.')
            removed.append((x, y, c))

    return cleaned, removed


def validate_level(grid):
    """Full validation of a level"""
    level = Level(grid)

    print("=== LEVEL VALIDATION ===\n")
    print("Grid:")
    for row in level.grid:
        print(''.join(row))

    print(f"\nEntry: {level.entry}")
    print(f"Exit: {level.exit}")
    print(f"Buttons: {level.buttons}")
    print(f"Gates: {level.gates}")
    print(f"Valves: {level.valves}")
    print(f"Barrels: {level.barrels}")

    # Check for shortcuts
    print("\n--- Shortcut Check ---")
    issues = validate_no_shortcuts(level)
    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✓ No shortcuts found")

    # Find useless pieces
    print("\n--- Useless Piece Check ---")
    useless = find_useless_pieces(level)
    if useless:
        for x, y, c, reason in useless:
            print(f"  ⚠ ({x},{y}) '{c}' - {reason}")
    else:
        print("  ✓ All pieces contribute to puzzle")

    # Cleanup
    if useless:
        print("\n--- Cleanup ---")
        cleaned, removed = cleanup_level(level)
        print(f"  Removed {len(removed)} useless pieces")
        print("\nCleaned grid:")
        for row in cleaned.grid:
            print(''.join(row))

    return level, issues, useless


# === TEST ===
if __name__ == '__main__':
    # Test level with some useless walls
    test_grid = [
        "##########",
        "#....#..O#",
        "S..#...>.#",
        "#..#.#...G",
        "#....#...X",
        "##########",
    ]

    validate_level(test_grid)

    print("\n" + "="*40 + "\n")

    # Test level with shortcut (can reach exit without button)
    shortcut_grid = [
        "##########",
        "#....#..O#",
        "S........#",
        "#........G",
        "#........X",
        "##########",
    ]

    print("TESTING SHORTCUT LEVEL:")
    validate_level(shortcut_grid)
