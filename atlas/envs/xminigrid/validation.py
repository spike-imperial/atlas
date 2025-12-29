"""
Solvability checker for filtering problem sets, described in Appendix E.3.

To evaluate whether a problem is valid/solvable, we decompose the RM into paths
to the accepting state and determine if the formulas along these paths are
satisfiable. An RM is considered solvable if the formulas in at least one path
are satisfiable. A formula is considered satisfiable if the objects associated
with it are reachable by the agent. For example, the formula front_ball is
satisfiable if there is a ball within the reach of the agent. Reachability is
determined by locked doors—hence, if a given formula cannot be satisfied by the
objects within reach, we select a locked door whose color matches a key within
reach. This procedure derives a tree where each child node increases the
reachability with respect to its parent. Maintaining a tree, keeping different
orderings on the opening of the locked doors, is important because only some of
them might guarantee solvability. For instance, if there is a single green
locked door and the reachability procedure opens it, a subsequent formula
front_door_green_locked will not be satisfiable.

NOTE: The code for the described functionality requires refactoring. Some
 functionalities exist within other modules (e.g., labeling function) and magic
 numbers are used. It is also not jittable. We keep it as originally used to
 ensure reproducibility, e.g. to produce the same validation and testing (CVaR)
 sets.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import permutations
import tempfile
from typing import List, Dict, Set, Tuple, Any, Optional

import pandas as pd
from xminigrid.core.constants import Colors, Tiles
import yaml

from .level import XMinigridLevel
from ...hrm.ops import dump
from ...hrm.types import HRM


GET_TILE = {
    0: "empty",
    1: "floor",
    2: "wall",
    3: "ball",
    4: "square",
    5: "pyramid",
    6: "goal",
    7: "key",
    8: "door_locked",
    9: "door_closed",
    10: "door_open",
    11: "hex",
    12: "star",
    27: "door_any"
}

GET_COLOR = {
    0: "empty",
    1: "red",
    2: "green",
    3: "blue",
    4: "purple",
    5: "yellow",
    6: "grey",
    7: "black",
    8: "orange",
    9: "white",
    10: "brown",
    11: "pink"
}


class ObservationType(Enum):
    FRONT = auto()     # Agent is in front of object
    CARRYING = auto()  # Agent is carrying object
    NEXT = auto()      # Two objects are adjacent


@dataclass
class Observation:
    """Represents an observation that must be satisfied."""
    negated: bool
    type: ObservationType
    objects: List[Tuple[int, int]]  # List of (object_type, object_color) tuples

    def __str__(self):
        """String representation of the observation."""
        negation = "not " if self.negated else ""
        if self.type == ObservationType.FRONT or self.type == ObservationType.CARRYING:
            obj = GET_TILE.get(self.objects[0][0], f"Unknown({self.objects[0][0]})")
            color = GET_COLOR.get(self.objects[0][1], f"Unknown({self.objects[0][1]})")
            return f"{negation}{self.type.name}({obj}, {color})"
        elif self.type == ObservationType.NEXT:
            obj1 = GET_TILE.get(self.objects[0][0], f"Unknown({self.objects[0][0]})")
            color1 = GET_COLOR.get(self.objects[0][1], f"Unknown({self.objects[0][1]})")
            obj2 = GET_TILE.get(self.objects[1][0], f"Unknown({self.objects[1][0]})")
            color2 = GET_COLOR.get(self.objects[1][1], f"Unknown({self.objects[1][1]})")
            return f"{negation}{self.type.name}(({obj1}, {color1}), ({obj2}, {color2}))"
        else:
            return f"{negation}{self.type.name}({self.objects})"


@dataclass
class ReachabilityState:
    """Represents the reachability state with a specific set of unlocked doors."""
    reachable_positions: Set[Tuple[int, int]]
    reachable_objects: Dict[int, List[Tuple[Tuple[int, int], int]]]
    collected_keys: Set[int]
    unlocked_doors: Set[Tuple[int, int]]

    def __hash__(self):
        """Make ReachabilityState hashable for use in sets."""
        return hash((
            frozenset(self.reachable_positions),
            frozenset((k, frozenset(v)) for k, v in self.reachable_objects.items()),
            frozenset(self.collected_keys),
            frozenset(self.unlocked_doors)
        ))

    def __eq__(self, other):
        """Define equality for ReachabilityState."""
        if not isinstance(other, ReachabilityState):
            return False
        return (self.reachable_positions == other.reachable_positions and
                self.collected_keys == other.collected_keys and
                self.unlocked_doors == other.unlocked_doors)


def check_sequence_satisfiability(
    level: "XMinigridLevel",
    observations: List[Observation]
) -> Dict[str, Any]:
    """
    Check if a sequence of observations can be satisfied in the given level,
    using a minimal number of door unlocks.

    Args:
        level: The XMinigridLevel object
        observations: List of observations that must be satisfied in sequence

    Returns:
        Dict containing satisfiability results and explanations
    """
    # Initialize results dictionary
    results = {
        'satisfiable': False,
        'explanation': '',
        'unlocked_doors': set(),
        'possible_states': []
    }

    # Setup the initial state
    # -----------------------
    # Find doors that are already unlocked in the level (open or closed doors)
    grid = level.grid
    initially_unlocked_doors = set()

    for y in range(level.height):
        for x in range(level.width):
            obj_type = int(grid[y, x, 0])
            if obj_type in [9, 10]:  # DOOR_CLOSED or DOOR_OPEN
                initially_unlocked_doors.add((y, x))

    # Compute initial reachability and create initial state
    initial_reachability = compute_reachability(level, initially_unlocked_doors)
    initial_state = ReachabilityState(
        reachable_positions=initial_reachability['reachable_positions'],
        reachable_objects=initial_reachability['reachable_objects'],
        collected_keys=initial_reachability['collected_keys'],
        unlocked_doors=initially_unlocked_doors
    )

    # Debug print the initial state
    # print_state_info("Initial state", initial_state)

    # Find all doors in the level for later reference
    doors = {}
    for y in range(level.height):
        for x in range(level.width):
            obj_type = int(grid[y, x, 0])
            if obj_type in [8, 9, 10]:  # Any door type
                doors[(y, x)] = (obj_type, int(grid[y, x, 1]))

    # Process each observation in sequence
    # -----------------------------------
    possible_states = [initial_state]

    for i, observation in enumerate(observations):
        # print(f"Processing observation {i+1}: {observation.type.name}")

        # Try to satisfy the observation with minimal door unlocking
        # First try without unlocking any additional doors
        satisfying_states = try_satisfy_observation(
            observation, level, possible_states, doors, unlock_doors=True
        )
        # print(f"Observation {i}:", len(satisfying_states))
        # for sat_state in satisfying_states:
        #     print_state_info("State", sat_state)

        if not satisfying_states:
            results['satisfiable'] = False
            results['explanation'] = f"Observation {i+1} cannot be satisfied, even with door unlocking."
            return results

        # Update possible states for next observation
        possible_states = satisfying_states

    # If we made it through all observations, the sequence is satisfiable
    results['satisfiable'] = True
    results['explanation'] = "The observation sequence can be satisfied."
    select_one_state = possible_states.pop()
    results['possible_state'] = select_one_state
    results['unlocked_doors'] = select_one_state.unlocked_doors - initially_unlocked_doors

    # Add detail about doors unlocked
    doors_unlocked = len(results['unlocked_doors'])
    if doors_unlocked == 0:
        results['explanation'] += " No doors needed to be unlocked."
    else:
        results['explanation'] += f" {doors_unlocked} door(s) needed to be unlocked."

    return results


def try_satisfy_observation(
    observation: Observation,
    level: "XMinigridLevel",
    states: List[ReachabilityState],
    doors: Dict[Tuple[int, int], Tuple[int, int]],
    unlock_doors: bool
):
    """
    Try to satisfy an observation with the given states.

    Args:
        observation: The observation to satisfy
        level: The XMinigridLevel object
        states: List of states to try
        doors: Dictionary of doors in the level
        unlock_doors: Whether to allow door unlocking

    Returns:
        List of states that satisfy the observation
    """
    satisfying_states = set()

    for state in states:
        satisfaction_results = can_satisfy_observation(
            observation, level, state, doors, unlock_doors=unlock_doors
        )

        if satisfaction_results['satisfiable']:
            # print("Found a satisfying state for observation", observation.type)
            satisfying_states.update(satisfaction_results['next_states'])

    return satisfying_states

def get_unlockable_doors(
    level: "XMinigridLevel",
    state: ReachabilityState,
    doors: Dict[Tuple[int, int], Tuple[int, int]]
) -> List[Tuple[Tuple[int, int], int]]:
    """
    Get all doors that could be unlocked with the keys we have collected.

    Args:
        level: The XMinigridLevel object
        state: The current state
        doors: Dictionary mapping positions to (door_type, door_color)

    Returns:
        List of (door_position, door_color) tuples
    """
    unlockable_doors = []

    for door_pos, (door_type, door_color) in doors.items():
        # Must be a locked door of a color we have a key for
        if (door_type == 8 and
            door_color in state.collected_keys and
            door_pos not in state.unlocked_doors):

            # Check if door is reachable from any adjacent position
            if is_position_reachable(door_pos, state.reachable_positions, level):
                unlockable_doors.append((door_pos, door_color))

    return unlockable_doors


def is_position_reachable(
   pos: Tuple[int, int],
   reachable_positions: Set[Tuple[int, int]],
   level: "XMinigridLevel"
) -> bool:
   """
   Check if a position is reachable based on adjacent reachable positions.
   A position is considered reachable if any adjacent position is reachable.

   Args:
       pos: The position to check
       reachable_positions: Set of positions that are reachable
       level: The XMinigridLevel object

   Returns:
       True if the position is reachable, False otherwise
   """
   y, x = pos

   for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
       ny, nx = y + dy, x + dx
       if ((ny, nx) in reachable_positions and
           0 <= ny < level.height and 0 <= nx < level.width):
           return True

   return False


def count_lockable_doors(level: "XMinigridLevel", state: ReachabilityState) -> int:
    """Count the maximum number of doors that could be unlocked."""
    # Count locked doors in the level
    grid = level.grid
    locked_door_count = 0

    for y in range(level.height):
        for x in range(level.width):
            if int(grid[y, x, 0]) == 8:  # DOOR_LOCKED
                locked_door_count += 1

    return locked_door_count


def print_state_info(title: str, state: ReachabilityState) -> None:
    """Print debug information about a state."""
    print(f"{title}:")
    print(f"  - Reachable positions: {len(state.reachable_positions)}")
    print(f"  - Collected keys: {[GET_COLOR.get(k, f'Unknown({k})') for k in state.collected_keys]}")
    print(f"  - Unlocked doors: {len(state.unlocked_doors)}")
    for door_pos in state.unlocked_doors:
        print(f"    - door at position {door_pos}")
    print(f"  - Locked doors")
    for obj_pos, color in state.reachable_objects.get(8, []):
        if obj_pos not in state.unlocked_doors:
            print(f"    - door at position {obj_pos} with color {GET_COLOR.get(color, f'Unknown({color})')}")
    print(f"  - Reachable objects:")
    for obj_type, objects in state.reachable_objects.items():
        if objects:
            obj_name = GET_TILE.get(obj_type, f"Unknown({obj_type})")
            print(f"    - {obj_name}: {len(objects)} instances")


def can_satisfy_observation(
   observation: Observation,
   level: "XMinigridLevel",
   state: ReachabilityState,
   doors: Dict[Tuple[int, int], Tuple[int, int]],
   unlock_doors: bool = False
) -> Dict[str, Any]:
    """
    Check if an observation can be satisfied from the current state,
    focusing only on basic reachability conditions.

    Args:
        observation: The observation to satisfy
        level: The XMinigridLevel object
        state: The current reachability state
        doors: Dictionary mapping positions to (door_type, door_color)
        unlock_doors: Whether to explore unlocking doors if necessary

    Returns:
        Dict containing satisfiability results and possible next states
    """
    result = {
        'satisfiable': False,
        'next_states': set()
    }

    if observation.type == ObservationType.FRONT:
            # Agent needs to be able to reach a specific object
            obj_type, obj_color = observation.objects[0]
            # Check if any matching object is reachable
            obj_reachable_objects = state.reachable_objects.get(obj_type, [])
            if obj_type == 9:
                obj_reachable_objects += state.reachable_objects.get(10, [])
            elif obj_type == 10:
                obj_reachable_objects += state.reachable_objects.get(9, [])

            for obj_pos, color in obj_reachable_objects:
                if color == obj_color or obj_color == 0:
                    if obj_type == 8 and obj_pos in state.unlocked_doors:
                        continue  # Skip this door as it cannot be relocked

                    new_state = deepcopy(state)
                    result['next_states'].add(new_state)
                    result['satisfiable'] = True

                # specific check for closed/open door if matching doors are locked
            if obj_type in [9, 10]:
            # Check if any locked doors could be used to satisfy this proposition
                for obj_pos, color in state.reachable_objects.get(8, []):
                    if color == obj_color or obj_color == 0:
                        # unlocked already, good to modify.
                        if obj_pos in state.unlocked_doors:
                            new_state = deepcopy(state)
                            result['next_states'].add(new_state)
                            result['satisfiable'] = True

                        # otherwise we must be able to unlock it
                        elif obj_color in state.collected_keys:
                            new_state = deepcopy(state)
                            new_state.unlocked_doors.add(obj_pos)
                            new_reachability = compute_reachability(level, new_state.unlocked_doors)
                            new_state.reachable_positions = new_reachability['reachable_positions']
                            new_state.reachable_objects = new_reachability['reachable_objects']
                            new_state.collected_keys = new_reachability['collected_keys']
                            result['next_states'].add(new_state)
                            result['satisfiable'] = True
            if obj_type == 27: # even more special case which is just a door in any state
                for obj_pos, color in state.reachable_objects.get(8, []) + state.reachable_objects.get(9, []) + state.reachable_objects.get(10, []):
                    if color == obj_color or obj_color == 0:
                        new_state = deepcopy(state)
                        result['next_states'].add(new_state)
                        result['satisfiable'] = True

                # If not satisfied and unlocking is allowed, try unlocking doors
            if unlock_doors and state.collected_keys:
                for key_color in state.collected_keys:
                    expanded_states = explore_door_unlocking(
                        deepcopy(state), key_color, doors, level, unlock_doors=True
                    )

                    for expanded_state in expanded_states:
                        expanded_result = can_satisfy_observation(
                            observation, level, expanded_state, doors, unlock_doors=True
                        )

                        if expanded_result['satisfiable']:
                            result['next_states'].update(expanded_result['next_states'])
                            result['satisfiable'] = True

    elif observation.type == ObservationType.CARRYING:

            obj_type, obj_color = observation.objects[0]

            if obj_type in [Tiles.KEY, Tiles.BALL, Tiles.SQUARE,]:
                # Check if a key of this color is reachable
                for obj_pos, color in state.reachable_objects.get(obj_type, []):
                    if color == obj_color or obj_color == 0:
                        # Key is reachable
                        new_state = deepcopy(state)
                        # new_state.collected_keys.add(obj_color)
                        result['next_states'].add(new_state)
                        result['satisfiable'] = True
                # If not satisfied and unlocking is allowed, try unlocking doors

                if unlock_doors and state.collected_keys:
                    for key_color in state.collected_keys:
                        # print("key color:", GET_COLOR.get(key_color, f"Unknown({key_color})"))
                        expanded_states = explore_door_unlocking(
                            deepcopy(state), key_color, doors, level, unlock_doors=True
                        )

                        for expanded_state in expanded_states:
                            expanded_result = can_satisfy_observation(
                                observation, level, expanded_state, doors, unlock_doors=True
                            )

                            if expanded_result['satisfiable']:
                                result['next_states'].update(expanded_result['next_states'])
                                result['satisfiable'] = True

    elif observation.type == ObservationType.NEXT:
            # Simply check if both object types are reachable independently
            obj1_type, obj1_color = observation.objects[0]
            obj2_type, obj2_color = observation.objects[1]

            obj1_reachable = False
            obj2_reachable = False

            obj1_pos = None

            obj1_reachable_objects = state.reachable_objects.get(obj1_type, [])
            if obj1_type == 9:
                obj1_reachable_objects += state.reachable_objects.get(10, [])
            elif obj1_type == 10:
                obj1_reachable_objects += state.reachable_objects.get(9, [])

            obj2_reachable_objects = state.reachable_objects.get(obj2_type, [])
            if obj2_type == 9:
                obj2_reachable_objects += state.reachable_objects.get(10, [])
            elif obj2_type == 10:
                obj2_reachable_objects += state.reachable_objects.get(9, [])

            for obj_pos, color in obj1_reachable_objects:
                if color == obj1_color or obj1_color == 0:
                    # There needs to be handling for either obj1 or obj2 being an open or unlocked door and there being a irreversible change from locked to unlocked to satisfy it.
                    if obj1_type == 8 and obj_pos in state.unlocked_doors:
                        continue  # Skip locked doors that are unlocked
                    obj1_reachable = True
                    obj1_pos = obj_pos
                    break
            if obj1_type in [9, 10]:
                # Check if any locked doors
                for obj_pos, color in state.reachable_objects.get(8, []):
                    if color == obj1_color or obj1_color == 0:
                        # unlocked already, good to modify.
                        if obj_pos in state.unlocked_doors:
                            obj1_reachable = True
                            break
            if obj1_type == 27: # even more special case which is just a door in any state
                for obj_pos, color in state.reachable_objects.get(8, []) + state.reachable_objects.get(9, []) + state.reachable_objects.get(10, []):
                    if color == obj1_color or obj1_color == 0:
                        obj1_reachable = True
                        break

            # Check if object 2 is reachable
            for obj_pos, color in obj2_reachable_objects:
                if obj_pos != obj1_pos and (color == obj2_color or obj2_color == 0):
                    if obj2_type == 8 and obj_pos in state.unlocked_doors:
                        continue  # Skip locked doors that are unlocked
                    obj2_reachable = True
                    break

            if obj2_type in [9, 10]:
                # Check if any locked doors
                for obj_pos, color in state.reachable_objects.get(8, []):
                    if color == obj2_color or obj2_color == 0:
                        # unlocked already, good to modify.
                        if obj_pos in state.unlocked_doors:
                            obj2_reachable = True
                            break

            if obj2_type == 27: # even more special case which is just a door in any state
                for obj_pos, color in state.reachable_objects.get(8, []) + state.reachable_objects.get(9, []) + state.reachable_objects.get(10, []):
                    if color == obj2_color or obj2_color == 0:
                        obj2_reachable = True
                        break

            # If both objects are reachable, the observation can be satisfied
            if obj1_reachable and obj2_reachable:
                new_state = deepcopy(state)
                result['next_states'].add(new_state)
                result['satisfiable'] = True

            # If not satisfied and unlocking is allowed, try unlocking doors
            elif not (obj1_reachable and obj2_reachable) and unlock_doors and state.collected_keys:
                for key_color in state.collected_keys:
                    expanded_states = explore_door_unlocking(
                        deepcopy(state), key_color, doors, level, unlock_doors=True
                    )

                    for expanded_state in expanded_states:
                        expanded_result = can_satisfy_observation(
                            observation, level, expanded_state, doors, unlock_doors=True
                        )

                        if expanded_result['satisfiable']:
                            result['next_states'].update(expanded_result['next_states'])
                            result['satisfiable'] = True

    return result


def explore_door_unlocking(
    state: ReachabilityState,
    key_color: int,
    doors: Dict[Tuple[int, int], Tuple[int, int]],
    level: "XMinigridLevel",
    unlock_doors: bool = True
) -> List[ReachabilityState]:
    """
    Explore states after unlocking doors with a specific key.
    Only unlocks doors when necessary.

    Args:
        state: The current reachability state
        key_color: The color of the key that was collected
        doors: Dictionary mapping positions to (door_type, door_color)
        level: The XMinigridLevel object
        unlock_doors: Whether to actually unlock doors

    Returns:
        List of possible next states
    """

    if not unlock_doors:
        return []  # Return current state without unlocking

    possible_to_unlock_more = False
    for door_pos, (door_type, door_color) in doors.items():
        if door_type == 8 and door_color == key_color and door_pos not in state.unlocked_doors:
            possible_to_unlock_more = True
            break

    if not possible_to_unlock_more:
        return []

    states = set()  # Start with the current state

    # Find all locked doors of matching color
    matching_locked_doors = []
    for door_pos, (door_type, door_color) in doors.items():
        if door_type == 8 and door_color == key_color:  # DOOR_LOCKED with matching color
            # Check if the door is reachable
            door_reachable = False
            y, x = door_pos

            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if ((ny, nx) in state.reachable_positions and
                    0 <= ny < level.height and 0 <= nx < level.width):
                    door_reachable = True
                    break

            if door_reachable and door_pos not in state.unlocked_doors:
                matching_locked_doors.append(door_pos)

    # Use a breadth-first approach to unlock minimal sets of doors
    # Start by trying to unlock just one door at a time (greedy approach)
    for door_pos in matching_locked_doors:
        new_state = deepcopy(state)
        new_state.unlocked_doors.add(door_pos)

        # Recompute reachability with this door unlocked
        new_reachability = compute_reachability(level, new_state.unlocked_doors)
        new_state.reachable_positions = new_reachability['reachable_positions']
        new_state.reachable_objects = new_reachability['reachable_objects']
        new_state.collected_keys = new_reachability['collected_keys']
        states.add(new_state)

    return states


def compute_reachability(level, unlocked_doors):
    """
    Compute reachability given the current set of unlocked doors.
    Locked doors are considered reachable but stop traversal beyond them.
    """
    # Initialize data structures
    grid = level.grid
    height, width = level.height, level.width
    start_pos = (int(level.agent_pos[0]), int(level.agent_pos[1]))

    # For BFS traversal
    queue = deque([start_pos])
    visited = {start_pos}

    # Track reachable objects by type
    reachable_objects = {tile_id: [] for tile_id in range(len(GET_TILE))}
    collected_keys = set()

    # Movement directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    # If the agent is carrying an object, add it to reachable objects
    pocket_obj, pocket_color = int(level.agent_pocket[0]), int(level.agent_pocket[1])
    if pocket_obj != Tiles.EMPTY and pocket_color != Colors.EMPTY:
            obj_type, obj_color = int(level.agent_pocket[0]), int(level.agent_pocket[1])
            if obj_type in reachable_objects:
                reachable_objects[obj_type].append((start_pos, obj_color))
            if obj_type == Tiles.KEY:
                collected_keys.add(obj_color)

    while queue:
        y, x = queue.popleft()

        # Check the current cell for objects
        obj_type, obj_color = int(grid[y, x, 0]), int(grid[y, x, 1])

        # Record this object (if it's not a floor or empty)
        if obj_type != 1 and obj_type != 0:  # Not FLOOR or EMPTY
            reachable_objects[obj_type].append(((y, x), obj_color))

        # If we found a key, add it to collected keys
        if obj_type == 7:  # KEY
            collected_keys.add(obj_color)

        # Try each direction
        for dy, dx in directions:
            ny, nx = y + dy, x + dx

            # Check bounds
            if 0 <= ny < height and 0 <= nx < width:
                # Skip if already visited
                if (ny, nx) in visited:
                    continue

                # Get the object at the new position
                new_obj_type = int(grid[ny, nx, 0])
                new_obj_color = int(grid[ny, nx, 1])

                # Handle different objects
                if new_obj_type == 2:  # WALL
                    # Walls are never passable
                    continue
                elif new_obj_type == 8:  # DOOR_LOCKED
                    # Mark the locked door as visited (reachable) but don't add to queue
                    # unless it's in unlocked_doors
                    visited.add((ny, nx))
                    reachable_objects[new_obj_type].append(((ny, nx), new_obj_color))

                    # Only continue BFS beyond this door if it's unlocked
                    if (ny, nx) in unlocked_doors:
                        queue.append((ny, nx))

                    # Otherwise stop BFS at this door
                    continue

                # Note: DOOR_CLOSED (9) and DOOR_OPEN (10) are always passable
                # Mark as visited and add to queue
                visited.add((ny, nx))
                queue.append((ny, nx))

    return {
        'reachable_positions': visited,
        'reachable_objects': reachable_objects,
        'collected_keys': collected_keys
    }


def print_satisfiability_results(results, level):
    """Print the results of the satisfiability check in a human-readable format."""
    print("\n=== OBSERVATION SEQUENCE SATISFIABILITY RESULTS ===")
    print(f"Satisfiable: {results['satisfiable']}")
    print(f"Explanation: {results['explanation']}")

    if results['unlocked_doors']:
        print("\nDoors that were unlocked:")
        for y, x in results['unlocked_doors']:
            color_id = int(level.grid[y, x, 1])
            color_name = GET_COLOR.get(color_id, f"Unknown({color_id})")
            print(f"  - {color_name} door at position ({y}, {x})")
    else:
        print("\nNo doors needed to be unlocked.")

    print("\nFinal reachability:")
    if results['possible_states']:
        state = results['possible_states']
        print(f"  - Reachable positions: {len(state.reachable_positions)}")
        print(f"  - Collected keys: {[GET_COLOR.get(k, f'Unknown({k})') for k in state.collected_keys]}")

        # Print reachable objects by category
        print("\nReachable objects:")
        for obj_type, objects in state.reachable_objects.items():
            if objects:
                obj_name = GET_TILE.get(obj_type, f"Unknown({obj_type})")
                print(f"  - {obj_name}: {len(objects)} instances")


def parse_object(obj_str: str, color: str = None) -> Optional[Tuple[Tiles, Any]]:
    """
    Parse object string into a tuple of (Tiles enum, color/attribute)

    Args:
        obj_str: Object string (e.g., "ball", "door_closed")
        color: Optional color string

    Returns:
        Tuple of (Tiles enum, color/attribute) or None if invalid
    """
    # Map object strings to Tiles enum
    obj_mapping = {
        "ball": Tiles.BALL,
        "key": Tiles.KEY,
        "square": Tiles.SQUARE,
        "pyramid": Tiles.PYRAMID,
        "goal": Tiles.GOAL,
        "door_closed": Tiles.DOOR_CLOSED,
        "door_locked": Tiles.DOOR_LOCKED,
        "door_open": Tiles.DOOR_OPEN,
        "door": Tiles.DOOR_CLOSED + Tiles.DOOR_OPEN + Tiles.DOOR_LOCKED,  # Default if state not specified
        "hex": Tiles.HEX,
        "star": Tiles.STAR,
        "wall": Tiles.WALL,
        "floor": Tiles.FLOOR,
        "empty": Tiles.EMPTY
    }

    # Map color strings to Colors enum
    color_mapping = {
        "red": Colors.RED,
        "green": Colors.GREEN,
        "blue": Colors.BLUE,
        "purple": Colors.PURPLE,
        "yellow": Colors.YELLOW,
        "grey": Colors.GREY,
        "black": Colors.BLACK,
        "orange": Colors.ORANGE,
        "white": Colors.WHITE,
        "brown": Colors.BROWN,
        "pink": Colors.PINK,
        None: Colors.EMPTY
    }

    if obj_str in obj_mapping:
        if color in color_mapping:
            return (obj_mapping[obj_str], color_mapping[color])
        else:
            return (obj_mapping[obj_str], Colors.EMPTY)

    return None


def parse_formula(formula_str: str) -> Tuple[bool, ObservationType, List[Tuple[Any, Any]]]:
    """
    Parse a formula string and convert it to observation type and objects.

    Handles various formula formats:
    - Simple with color: "front_ball_grey"
    - Simple without color: "front_ball"
    - Relational with colors: "next_square_blue_door_purple_closed"
    - Relational mixed: "next_ball_door_closed"
    - Door with state and color: "front_door_purple_closed"

    Now also handles negations with a leading "-".

    Examples from input:
    - next_ball_key -> (False, NEXT, [(BALL, 0), (KEY, 0)])
    - front_door_purple_closed -> (False, FRONT, [(DOOR_CLOSED, PURPLE)])
    - next_key_blue_door_locked -> (False, NEXT, [(KEY, BLUE), (DOOR_LOCKED, EMPTY)])
    - -front_ball -> (True, FRONT, [(BALL, EMPTY)])

    Args:
        formula_str: Formula string

    Returns:
        Tuple of (ObservationType, List of object tuples)
    """
    negated = False
    if formula_str.startswith("-"):
        negated = True
        formula_str = formula_str[1:]

    parts = formula_str.lower().split('_')
    if len(parts) < 2:
        return None, []

    # Determine observation type
    relation = parts[0]
    relation_mapping = {
        "next": ObservationType.NEXT,
        "front": ObservationType.FRONT,
        "carrying": ObservationType.CARRYING
    }

    if relation not in relation_mapping:
        return None, []

    obs_type = relation_mapping[relation]

    # Process objects based on the pattern
    objects = []

    valid_colors = {'red', 'green', 'blue', 'purple', 'yellow', 'grey', 'black', 'orange', 'white', 'brown', 'pink'}
    door_states = {'open', 'closed', 'locked'}

    # For front/carrying relations
    if relation in ["front", "carrying"]:
        # Door with color and state: "front_door_purple_closed"
        if len(parts) >= 4 and parts[1] == "door" and parts[2] in valid_colors and parts[3] in door_states:
            door_type = f"door_{parts[3]}"
            obj = parse_object(door_type, parts[2])
            if obj:
                objects.append(obj)

        # Door with state but no color: "front_door_closed"
        elif len(parts) == 3 and parts[1] == "door" and parts[2] in door_states:
            door_type = f"door_{parts[2]}"
            obj = parse_object(door_type)
            if obj:
                objects.append(obj)

        # Regular object with color: "front_ball_blue"
        elif len(parts) == 3 and parts[2] in valid_colors:
            obj = parse_object(parts[1], parts[2])
            if obj:
                objects.append(obj)

        # Object without color: "front_ball"
        elif len(parts) == 2:
            obj = parse_object(parts[1])
            if obj:
                objects.append(obj)

    # For next relation (can have multiple objects)
    elif relation == "next":
        i = 1
        while i < len(parts):
            # Object with color (ball_blue)
            if i + 1 < len(parts) and parts[i+1] in valid_colors and parts[i] != "door":
                obj = parse_object(parts[i], parts[i+1])
                if obj:
                    objects.append(obj)
                i += 2

            # Door with color and state (door_purple_closed)
            elif i + 2 < len(parts) and parts[i] == "door" and parts[i+1] in valid_colors and parts[i+2] in door_states:
                door_type = f"door_{parts[i+2]}"
                obj = parse_object(door_type, parts[i+1])
                if obj:
                    objects.append(obj)
                i += 3

            # Door with state but no color (door_closed)
            elif i + 1 < len(parts) and parts[i] == "door" and parts[i+1] in door_states:
                door_type = f"door_{parts[i+1]}"
                obj = parse_object(door_type)
                if obj:
                    objects.append(obj)
                i += 2

            # Door with color but no state (door_purple)
            elif i + 1 < len(parts) and parts[i] == "door" and parts[i+1] in valid_colors:
                obj = parse_object(parts[i], parts[i+1])
                if obj:
                    objects.append(obj)
                i += 2

            # Object without color (ball)
            else:
                obj = parse_object(parts[i])
                if obj:
                    objects.append(obj)
                i += 1

    return negated, obs_type, objects


def extract_sequences_from_yaml(yaml_dict: Dict) -> List[List[Tuple[str, str, List[Observation]]]]:
    """
    Extract all possible sequences from u0 to uA for graph structures.

    Returns:
        List of sequences, where each sequence is a list of tuples (from_node, to_node, reward)
    """
    # Get the transitions from the YAML
    machine = yaml_dict.get("transitions", {})
    transitions = machine.get("m0", {})

    # Find all paths from u0 to uA using DFS
    all_sequences = []

    def dfs(current_node: str, path: List[Tuple[str, str, List[Observation]]], visited: Set[str]):
        # If we've reached uA, add the path to our sequences
        if current_node == "uA":
            all_sequences.append(path.copy())
            return

        # Get all possible transitions from current node
        if current_node not in transitions:
            return

        for dest_node, data in transitions[current_node].items():
            if dest_node in visited:
                continue

            edges = data.get("edges", [])
            for edge in edges:
                if edge.get("call") == "leaf" and "formula" in edge:
                    formulas = edge.get("formula", [])
                    observations_required = []
                    for formula in formulas:
                        if formula:
                            negated, obs_type, obs_objects = parse_formula(formula)
                            observations_required.append(Observation(negated, obs_type, obs_objects))
                    path.append((current_node, dest_node, observations_required))

            # Add edge to path and continue DFS
            visited.add(dest_node)

            dfs(dest_node, path, visited)

            # Backtrack
            path.pop()
            visited.remove(dest_node)

    # Start DFS from u0
    dfs("u0", [], set())

    return all_sequences


def syntactic_subset(obs1: Observation, obs2: Observation) -> bool:
    """
    Check if obs1 is a subset of obs2.

    Args:
        obs1: First observation
        obs2: Second observation (must be a negated observation)

    Returns:
        True if obs1 is a subset of obs2, False otherwise
    """
    def get_encompassed_types(obj_type):
        """Get all object types that are encompassed by the given type."""
        if obj_type == 27:  # DOOR_ANY
            return {8, 9, 10, 27}  # DOOR_LOCKED, DOOR_UNLOCKED, DOOR_OPEN, and itself
        else:
            return {obj_type}  # Regular objects only encompass themselves
    # Check if the types match

    if obs1.type != obs2.type:
        return False

    obs_type = obs1.type
    match obs_type:
        case ObservationType.FRONT:
            obj1 = obs1.objects[0]
            obj2_encompasses = get_encompassed_types(obs2.objects[0])
            # Check if the object types match
            if not obj1[0] in obj2_encompasses:
                return False
            # Check if the colors match
            if obj1[1] != obj2_encompasses[1] and obj2_encompasses[1] != 0:
                return False
        case ObservationType.CARRYING:
            obj1 = obs1.objects[0]
            obj2 = obs2.objects[0]
            # Check if the object types match
            if obj1[0] != obj2[0]:
                return False
            # Check if the colors match
            if obj1[1] != obj2[1] and obj2[1] != 0:
                return False
        case ObservationType.NEXT:
            """Check if obs1 objects are encompassed by obs2 objects with proper color matching."""
            obj1a, obj1b = obs1.objects
            obj2a, obj2b = obs2.objects

            # Get all types that each obj2 object can encompass
            obj2a_encompasses = get_encompassed_types(obj2a[0])
            obj2b_encompasses = get_encompassed_types(obj2b[0])

            # Check if obj1 types can be satisfied by obj2 types
            obj1_types = [obj1a[0], obj1b[0]]

            # Create a list of what obj2 can provide (considering encompassing)
            obj2_available = []
            for encompassed_type in obj2a_encompasses:
                obj2_available.append(encompassed_type)
            for encompassed_type in obj2b_encompasses:
                obj2_available.append(encompassed_type)

            # Count what's needed vs what's available
            obj1_counts = pd.Series(obj1_types).value_counts()
            obj2_counts = pd.Series(obj2_available).value_counts()

            # Check if obj2 has enough of each type that obj1 needs
            for obj_type, needed_count in obj1_counts.items():
                available_count = obj2_counts.get(obj_type, 0)
                if available_count < needed_count:
                    return False

            # If obj2 has wildcard colors (color 0), then any color matching is valid
            if obj2a[1] == 0 and obj2b[1] == 0:
                return True

            # Now check if there exists a valid color matching
            # We need to try all possible ways to assign obj1 objects to obj2 objects
            # considering that obj2 objects might encompass obj1 object types

            obj2_objects = [obj2a, obj2b]

            # Generate all possible ways to match obj1 objects to obj2 objects
            for perm in permutations(obj2_objects, 2):
                candidate_obj2a, candidate_obj2b = perm

                # Check if this assignment works (considering encompassing)
                obj1a_compatible = obj1a[0] in get_encompassed_types(candidate_obj2a[0])
                obj1b_compatible = obj1b[0] in get_encompassed_types(candidate_obj2b[0])

                color1_compatible = (obj1a[1] == candidate_obj2a[1] or candidate_obj2a[1] == 0)
                color2_compatible = (obj1b[1] == candidate_obj2b[1] or candidate_obj2b[1] == 0)

                if (obj1a_compatible and obj1b_compatible and
                    color1_compatible and color2_compatible):
                    return True

            return False
        case _:
            raise ValueError(f"Unknown observation type: {obs_type}")
    return True


def check_seq_semantic_satisfiability(paths: List[List[Tuple[str, str, List[Observation]]]]):
    """
    Check if the sequence of observations is syntactically satisfiable.
    This function takes the list of possible paths through an HRM and filters out those that are not possible to solve based on the semantics of the propositions
    """
    semantically_satisfiable_paths = []
    for path in paths:
        sat = True
        # We need to determine that all possible observations are satisfiable in conjunction
        # This function does this by ensuring that the positive proposition is not a subset of any negative proposition
        # If this is true, we can ignore the negative propositions.
        new_path = []
        for a, b, observations in path:
            neg_obs = [obs for obs in observations if obs.negated]
            pos = [obs for obs in observations if not obs.negated][0] # this is a big assumption
            for neg in neg_obs:
                if syntactic_subset(pos, neg):
                    print(f"Found subset ({pos}, {neg}), path is unsatisfiable")
                    sat = False
                    # If a positive observation is a subset of a negative observation, the path is not satisfiable
                    break
            if not sat:
                break
            new_path.append(pos)
        if sat:
            semantically_satisfiable_paths.append(new_path)
    # print(f"Found {len(semantically_satisfiable_paths)} semantically satisfiable paths")
    # print("These are the paths:")
    # for path in semantically_satisfiable_paths:
    #     print(path)
    return semantically_satisfiable_paths


def check_satisfiability(level: 'XMinigridLevel', sequences: List[List[Observation]]) -> Dict[str, Any]:
    explanation = {}
    for sequence_id in range(len(sequences)):
        sequence = sequences[sequence_id]
        result = check_sequence_satisfiability(level, sequence)
        if result['satisfiable']:
            return {"satisfiable": True, "explanation": {sequence_id: {"sequence": [str(obs) for obs in sequence], "explanation": result['explanation']}}}
        else:
            explanation[sequence_id] = {"sequence": [str(obs) for obs in sequence], "explanation": result['explanation']}
    return {
        'satisfiable': False,
        'explanation': explanation
    }


def load_observations_from_yaml(file_path: str) -> List[Observation]:
    """Convert yaml file into sequence of observations"""
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    return check_seq_semantic_satisfiability(extract_sequences_from_yaml(yaml_data))


def is_solvable_problem(level: XMinigridLevel, hrm: HRM, alphabet: List[str]):
    """
    Returns whether the <level, HRM> pair is solvable, i.e. there is at least
    one rollout that results in the HRM task being realised in the level.
    """
    with tempfile.NamedTemporaryFile() as fp:
        dump(hrm, fp.name, alphabet)
        is_sat = check_satisfiability(level, load_observations_from_yaml(fp.name))
        return is_sat["satisfiable"]
