#!/usr/bin/env python3
"""
GAME PIECES MANIFEST
All pieces for the steam factory puzzle game
Used by generator and renderer
"""

# === PIECE TYPES ===
PIECES = {
    # TERRAIN
    'FLOOR': {
        'char': '.',
        'name': 'Floor',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'description': 'Empty floor tile, walkable',
    },
    'WALL': {
        'char': '#',
        'name': 'Wall',
        'solid': True,
        'deadly': False,
        'pushable': False,
        'description': 'Solid wall, blocks movement',
    },

    # ENTRY/EXIT
    'ENTRY': {
        'char': 'S',
        'name': 'Entry',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'description': 'Level start point',
    },
    'EXIT': {
        'char': 'X',
        'name': 'Exit',
        'solid': False,  # Solid when locked
        'deadly': False,
        'pushable': False,
        'description': 'Level exit, opens when button pressed',
    },

    # BUTTONS & GATES
    'BUTTON': {
        'char': 'O',
        'name': 'Pressure Button',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'activates': 'GATE',
        'activated_by': ['PLAYER', 'BARREL', 'GHOST'],
        'description': 'Pressure plate - player, barrel, or ghost can hold it',
    },
    'GATE': {
        'char': 'G',
        'name': 'Gate',
        'solid': True,  # Solid when closed
        'deadly': False,
        'pushable': False,
        'opened_by': 'BUTTON',
        'description': 'Bronze gate, opens when button pressed',
    },

    # HAZARDS
    'OPEN_VALVE': {
        'char': 'V',
        'name': 'Open Valve',
        'solid': False,
        'deadly': True,
        'pushable': False,
        'sealable_by': 'BARREL',
        'description': 'Dangerous steam blast - step on = death. Seal with barrel.',
    },

    # PUSHABLES
    'BARREL': {
        'char': 'B',
        'name': 'Barrel',
        'solid': True,  # Blocks movement until pushed
        'deadly': False,
        'pushable': True,
        'can_seal': 'OPEN_VALVE',
        'can_press': 'BUTTON',
        'description': 'Copper pressure barrel - push to move, seals valves, presses buttons',
    },

    # CONVEYORS
    'CONVEYOR_R': {
        'char': '>',
        'name': 'Conveyor Right',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'direction': (1, 0),
        'description': 'Belt conveyor - carries player/barrel right',
    },
    'CONVEYOR_L': {
        'char': '<',
        'name': 'Conveyor Left',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'direction': (-1, 0),
        'description': 'Belt conveyor - carries player/barrel left',
    },
    'CONVEYOR_U': {
        'char': '^',
        'name': 'Conveyor Up',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'direction': (0, -1),
        'description': 'Belt conveyor - carries player/barrel up',
    },
    'CONVEYOR_D': {
        'char': 'v',
        'name': 'Conveyor Down',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'direction': (0, 1),
        'description': 'Belt conveyor - carries player/barrel down',
    },

    # ONE-WAY GATES
    'ONEWAY_R': {
        'char': ')',
        'name': 'One-Way Right',
        'solid': False,  # Only from one side
        'deadly': False,
        'pushable': False,
        'passable_from': (-1, 0),  # Can enter from left
        'description': 'One-way gate - can only pass going right',
    },
    'ONEWAY_L': {
        'char': '(',
        'name': 'One-Way Left',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'passable_from': (1, 0),  # Can enter from right
        'description': 'One-way gate - can only pass going left',
    },
    'ONEWAY_U': {
        'char': 'A',
        'name': 'One-Way Up',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'passable_from': (0, 1),  # Can enter from below
        'description': 'One-way gate - can only pass going up',
    },
    'ONEWAY_D': {
        'char': 'Y',
        'name': 'One-Way Down',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'passable_from': (0, -1),  # Can enter from above
        'description': 'One-way gate - can only pass going down',
    },

    # DECORATION (safe)
    'PIPE': {
        'char': 'P',
        'name': 'Pipe',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'description': 'Decorative pipe section - walkable',
    },
    'STEAM_VENT': {
        'char': '~',
        'name': 'Steam Vent',
        'solid': False,
        'deadly': False,
        'pushable': False,
        'description': 'Small grated steam vent - safe, decorative',
    },
}

# === CHAR LOOKUPS ===
CHAR_TO_PIECE = {p['char']: name for name, p in PIECES.items()}
PIECE_TO_CHAR = {name: p['char'] for name, p in PIECES.items()}

# === CATEGORIES ===
SOLID_PIECES = [name for name, p in PIECES.items() if p['solid']]
DEADLY_PIECES = [name for name, p in PIECES.items() if p['deadly']]
PUSHABLE_PIECES = [name for name, p in PIECES.items() if p['pushable']]
CONVEYOR_PIECES = [name for name, p in PIECES.items() if 'direction' in p]
ONEWAY_PIECES = [name for name, p in PIECES.items() if 'passable_from' in p]

# === GENERATOR WEIGHTS ===
# How likely each piece is to be placed (relative weights)
GENERATOR_WEIGHTS = {
    'WALL': 10,
    'OPEN_VALVE': 3,
    'BARREL': 3,
    'CONVEYOR_R': 2,
    'CONVEYOR_L': 2,
    'CONVEYOR_U': 2,
    'CONVEYOR_D': 2,
    'ONEWAY_R': 1,
    'ONEWAY_L': 1,
    'ONEWAY_U': 1,
    'ONEWAY_D': 1,
    'PIPE': 4,
    'STEAM_VENT': 2,
}

# === PUZZLE CONSTRAINTS ===
CONSTRAINTS = {
    'min_path_length': 3,
    'max_barrels': 4,
    'max_open_valves': 5,
    'barrel_valve_ratio': 1.2,  # Slightly more barrels than valves
    'require_button_gate': True,
    'require_solvable': True,
}

if __name__ == '__main__':
    print("=== GAME PIECES MANIFEST ===\n")
    for name, piece in PIECES.items():
        print(f"[{piece['char']}] {name}: {piece['description']}")
    print(f"\nTotal pieces: {len(PIECES)}")
    print(f"Solid: {SOLID_PIECES}")
    print(f"Deadly: {DEADLY_PIECES}")
    print(f"Pushable: {PUSHABLE_PIECES}")
