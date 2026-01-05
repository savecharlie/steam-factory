# Steam Factory - Sliding Puzzle Game

## Core Goal
**The point of this project is to GENERATE levels procedurally, not hand-craft them.**

## IMPORTANT: Always Regenerate
After ANY change to game mechanics or generator logic:
1. Run `python3 src/train_generator.py` to regenerate levels
2. Copy output to `steam_factory.html` OR have generator update HTML directly
3. NEVER hand-edit levels in HTML - always regenerate

The generator (`src/train_generator.py`) must produce solvable, meaningful puzzles where:
- Every piece is necessary to solve the puzzle
- Path length is maximized (more moves = better puzzle)
- Mechanics are taught progressively (movement → barrel → button → valve → conveyor → combo)

## Key Mechanics
- **Sliding**: Player slides until hitting a wall
- **Barrels**: Push to move, slide until hitting wall. Can hold buttons or seal valves.
- **Buttons**: MOMENTARY - gate only open while something (player OR barrel) is on button
- **Valves**: Deadly unless sealed by barrel
- **Conveyors**: Redirect player mid-slide
- **Gates**: Block path until button is pressed
- **Teleporters**: Portal pairs (1/2). Player AND barrels warp through and continue sliding in same direction.

## Teleporter Generation Bug (SOLVED)
**Problem**: Random teleporter placement doesn't guarantee barrel teleportation is REQUIRED.
The BFS solver finds shortest solution - if barrel can reach button directly, teleportation won't be used.

**Solution**: For levels requiring barrel teleportation:
1. Generate level with teleporters
2. Simulate the found solution
3. Track if any barrel actually passes through a teleporter during simulation
4. Only keep levels where `barrel_teleported == True`

**Code pattern for filtering**:
```python
def barrel_teleports_in_solution(lv, solution):
    """Returns True only if solution REQUIRES barrel to teleport"""
    # Simulate solution step by step
    # When barrel slides into teleporter position, set barrel_teleported = True
    # Return barrel_teleported
```

## Generator Requirements
The generator must verify:
1. Level is solvable (BFS pathfinding)
2. Barrels can actually be pushed (space behind them)
3. Player can reach exit AFTER pushing barrel
4. Button levels teach momentary nature (player steps on button first, learns gate closes when leaving)
5. Every obstacle is required for the solution

## Train Order (Progressive Teaching)
1. Movement - basic sliding
2. Barrel - learn to push
3. Button - use barrel to hold button (must learn button is momentary first!)
4. Valve - seal deadly valve with barrel
5. Conveyor - get redirected
6. Combo - everything together

## Graphics
Burnished Bronze steampunk palette - original graphics are in `index.html`
