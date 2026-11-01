"""Search for a sequential climbing plan using the robot's full kinematic model."""

from __future__ import annotations

import heapq
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import placo
import microban_model


LIMBS = ("left_hand", "right_hand", "left_foot", "right_foot")
MOVE_CYCLE = ("left_foot", "right_foot", "left_hand", "right_hand")
LIMB_INDEX = {limb: index for index, limb in enumerate(LIMBS)}
LIMB_FRAMES = {
    "left_hand": "left_hand_ball",
    "right_hand": "right_hand_ball",
    "left_foot": "left_foot_ball",
    "right_foot": "right_foot_ball",
}

DEFAULT_WALL = "N2"
HOLD_JSON_BY_WALL = {
    "N1": Path(__file__).with_name("holds_N1.json"),
    "N2": Path(__file__).with_name("holds_N2.json"),
}
GRID_PITCH_M = 0.015
GRID_X0_M = 0.0175
GRID_Z0_M = 0.010
SOCKET_OUT_M = 0.026
SOCKET_BELOW_M = 0.014
LIMB_STEP_LIMIT_M = {
    "left_hand": 0.15,
    "right_hand": 0.15,
    "left_foot": 0.17,
    "right_foot": 0.17,
}
START_IK_ITERS = 120
MOVE_IK_ITERS = 30
IK_TOLERANCE_M = 0.006
CONTACT_TASK_WEIGHT = 300.0
POSTURE_TASK_WEIGHT = 0.01

# Standard opening stances. A hold is identified by the left bolt's grid cell.
START_CELLS_BY_WALL = {
    "N1": {
        "left_hand": (18, 22),
        "right_hand": (29, 22),
        "left_foot": (21, 2),
        "right_foot": (26, 2),
    },
    "N2": {
        "left_hand": (18, 23),
        "right_hand": (29, 23),
        "left_foot": (21, 2),
        "right_foot": (26, 2),
    },
}

# Backwards-compatible default path; use holds_json_for_wall() for a selected wall.
DEFAULT_HOLDS_JSON = HOLD_JSON_BY_WALL[DEFAULT_WALL]


def _normalize_wall(wall: str) -> str:
    normalized = str(wall).strip().upper()
    if normalized not in HOLD_JSON_BY_WALL:
        raise ValueError(f"Unknown wall {wall!r}; choose N1 or N2")
    return normalized


def holds_json_for_wall(wall: str = DEFAULT_WALL) -> Path:
    """Return the default holds file for a wall (N1 or N2)."""
    return HOLD_JSON_BY_WALL[_normalize_wall(wall)]


def load_holds(
    holds_json: str | Path | None = None,
    wall: str = DEFAULT_WALL,
) -> dict[str, dict[str, Any]]:
    """Load holds using only their grid columns and rows; ignore measured mm fields."""
    wall = _normalize_wall(wall)
    path = holds_json_for_wall(wall) if holds_json is None else Path(holds_json)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("holds"), list):
        raise ValueError(f"{path} must contain a JSON object with a 'holds' list")

    holds = {}
    for item in data["holds"]:
        try:
            col, row = int(item["col"]), int(item["row"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Each hold in {path} needs numeric col and row fields") from exc
        hold_id = f"c{col}r{row}"
        if hold_id in holds:
            raise ValueError(f"Duplicate hold at column {col}, row {row} in {path}")
        holds[hold_id] = {
            "id": hold_id,
            "col": col,
            "row": row,
            "position": hold_position(col, row),
        }
    if not holds:
        raise ValueError(f"No holds found in {path}")
    return holds


def hold_position(col: int, row: int) -> np.ndarray:
    """Socket centre in the simulator's world frame, derived from grid coordinates."""
    x = GRID_X0_M + (col + 1) * GRID_PITCH_M
    y = -SOCKET_OUT_M
    z = GRID_Z0_M + row * GRID_PITCH_M - SOCKET_BELOW_M
    return np.array([x, y, z], dtype=float)


def default_start_state(
    holds: dict[str, dict[str, Any]], wall: str = DEFAULT_WALL
) -> dict[str, str]:
    current_start_pos = START_CELLS_BY_WALL[_normalize_wall(wall)]
    by_cell = {(hold["col"], hold["row"]): hold_id for hold_id, hold in holds.items()}
    missing = [cell for cell in current_start_pos.values() if cell not in by_cell]
    if missing:
        raise ValueError(
            f"Cannot infer the standard {wall} start stance; missing hold cells "
            f"{missing}. Pass start_state for a different route."
        )
    return {limb: by_cell[cell] for limb, cell in current_start_pos.items()}


def climbing_aStar(
    holds_json: str | Path | None = None,
    start_state: dict[str, str] | None = None,
    goal_row: int | None = None,
    wall: str = DEFAULT_WALL,
) -> list[dict[str, Any]] | None:
    """Plan a route on N1 or N2 using that wall's holds and default start stance.

    Each search edge solves inverse kinematics for the entire robot: the three
    stationary ball frames remain on their current sockets while the scheduled
    limb moves. The search stops when either hand reaches ``goal_row`` (the
    highest hold row by default). Returned entries contain ``limb``, ``hold_id``,
    and the destination ``hold``'s ``col`` and ``row``.
    """
    wall = _normalize_wall(wall)
    holds = load_holds(holds_json, wall=wall)
    initial = _normalize_start_state(start_state, holds, wall)
    start_positions = tuple(initial[limb] for limb in LIMBS)
    highest_row = max(hold["row"] for hold in holds.values())
    target_row = highest_row if goal_row is None else int(goal_row)
    if target_row > highest_row:
        raise ValueError(f"goal_row {target_row} is above the highest hold row ({highest_row})")
    if any(holds[start_positions[LIMB_INDEX[h]]]["row"] >= target_row
           for h in ("left_hand", "right_hand")):
        return []

    robot = _make_climbing_robot()
    start_config = _solve_start_pose(robot, start_positions, holds)
    if start_config is None:
        return None

    counter = itertools.count()
    start_state_key = (start_positions, 0)
    frontier = [(_heuristic(start_positions, holds, target_row, turn=0), 0.0,
                 next(counter), start_state_key)]
    best_cost = {start_state_key: 0.0}
    configurations = {start_state_key: start_config}
    came_from = {}

    while frontier:
        _, current_cost, _, state_key = heapq.heappop(frontier)
        if current_cost > best_cost.get(state_key, math.inf):
            continue
        positions, turn = state_key
        if any(holds[positions[LIMB_INDEX[h]]]["row"] >= target_row
               for h in ("left_hand", "right_hand")):
            return _reconstruct_moves(came_from, state_key)

        limb = MOVE_CYCLE[turn]
        limb_index = LIMB_INDEX[limb]
        current_hold = holds[positions[limb_index]]
        occupied = set(positions)
        parent_config = configurations[state_key]
        next_rows = [hold["row"] for hold in holds.values() if hold["row"] > current_hold["row"]]
        if not next_rows:
            continue
        next_row = min(next_rows)

        for next_hold_id, next_hold in holds.items():
            if (next_hold_id in occupied or next_hold["row"] != next_row
                    or not is_in_reach(limb, current_hold, next_hold)):
                continue

            next_config = _solve_move(
                robot, parent_config, positions, limb, next_hold, holds
            )
            if next_config is None:
                continue

            next_positions_list = list(positions)
            next_positions_list[limb_index] = next_hold_id
            next_positions = tuple(next_positions_list)
            next_state_key = (next_positions, (turn + 1) % len(MOVE_CYCLE))
            move_cost = float(np.linalg.norm(next_hold["position"] - current_hold["position"]))
            tentative_cost = current_cost + move_cost
            if tentative_cost >= best_cost.get(next_state_key, math.inf):
                continue

            move = {
                "limb": limb,
                "hold_id": next_hold_id,
                "hold": {"col": next_hold["col"], "row": next_hold["row"]},
            }
            came_from[next_state_key] = (state_key, move)
            best_cost[next_state_key] = tentative_cost
            configurations[next_state_key] = next_config
            priority = tentative_cost + _heuristic(
                next_positions, holds, target_row, next_state_key[1]
            )
            heapq.heappush(frontier, (priority, tentative_cost, next(counter), next_state_key))

    return None


def _make_climbing_robot():
    robot = placo.RobotWrapper(str(microban_model.urdf(climb=True)))
    robot.set_joint_limits("left_knee", 0.01, 2.3)
    robot.set_joint_limits("right_knee", 0.01, 2.3)
    robot.update_kinematics()
    return robot


def _solve_start_pose(robot, positions, holds):
    """Find one whole-body pose with all four balls on the start sockets."""
    solver = placo.KinematicsSolver(robot)
    for limb in LIMBS:
        task = solver.add_position_task(LIMB_FRAMES[limb], holds[positions[LIMB_INDEX[limb]]]["position"])
        task.configure(f"start_{limb}", "soft", CONTACT_TASK_WEIGHT)
    com_task = solver.add_com_task(np.mean([holds[h]["position"] for h in positions], axis=0))
    com_task.configure("start_com", "soft", POSTURE_TASK_WEIGHT)
    body_orientation = robot.get_T_world_frame("body")[:3, :3].copy()
    orientation_task = solver.add_orientation_task("body", body_orientation)
    orientation_task.configure("start_body_orientation", "soft", POSTURE_TASK_WEIGHT)
    solver.add_regularization_task(1e-3)
    solver.enable_joint_limits(True)

    return _solve_and_capture(robot, solver, positions, holds, START_IK_ITERS)


def _solve_move(robot, parent_config, positions, moving_limb, target_hold, holds):
    """Solve one move with the other three end effectors fixed at their holds."""
    _restore_configuration(robot, parent_config)
    fixed_contacts = {
        limb: holds[positions[index]]["position"]
        for index, limb in enumerate(LIMBS)
        if limb != moving_limb
    }

    solver = placo.KinematicsSolver(robot)
    for limb, target in fixed_contacts.items():
        task = solver.add_position_task(LIMB_FRAMES[limb], target)
        task.configure(f"fixed_{limb}", "soft", CONTACT_TASK_WEIGHT)

    move_task = solver.add_position_task(LIMB_FRAMES[moving_limb], target_hold["position"])
    move_task.configure(f"move_{moving_limb}", "soft", CONTACT_TASK_WEIGHT)
    com_target = np.mean(list(fixed_contacts.values()), axis=0)
    com_task = solver.add_com_task(com_target)
    com_task.configure("support_com", "soft", POSTURE_TASK_WEIGHT)
    body_orientation = robot.get_T_world_frame("body")[:3, :3].copy()
    orientation_task = solver.add_orientation_task("body", body_orientation)
    orientation_task.configure("body_orientation", "soft", POSTURE_TASK_WEIGHT)
    solver.add_regularization_task(1e-3)
    solver.enable_joint_limits(True)

    next_positions = list(positions)
    next_positions[LIMB_INDEX[moving_limb]] = target_hold["id"]
    return _solve_and_capture(robot, solver, tuple(next_positions), holds, MOVE_IK_ITERS)


def _solve_and_capture(robot, solver, positions, holds, iterations):
    try:
        for _ in range(iterations):
            solver.solve(True)
            robot.update_kinematics()
    except RuntimeError:
        # An infeasible QP means this candidate pose cannot be reached under
        # the current joint limits and posture tasks.
        return None

    for limb in LIMBS:
        target = holds[positions[LIMB_INDEX[limb]]]["position"]
        actual = robot.get_T_world_frame(LIMB_FRAMES[limb])[:3, 3]
        if np.linalg.norm(actual - target) > IK_TOLERANCE_M:
            return None

    return {
        "base": robot.get_T_world_fbase().copy(),
        "joints": {name: robot.get_joint(name) for name in robot.joint_names()},
    }


def _restore_configuration(robot, config):
    robot.set_T_world_fbase(config["base"])
    for name, angle in config["joints"].items():
        robot.set_joint(name, angle)
    robot.update_kinematics()


def _normalize_start_state(start_state, holds, wall=DEFAULT_WALL):
    raw = default_start_state(holds, wall=wall) if start_state is None else start_state
    if set(raw) != set(LIMBS):
        raise ValueError(f"start_state must specify exactly these limbs: {', '.join(LIMBS)}")

    normalized = {}
    for limb in LIMBS:
        value = raw[limb]
        hold_id = f"c{int(value[0])}r{int(value[1])}" if isinstance(value, (tuple, list)) else str(value)
        if hold_id not in holds:
            raise ValueError(f"start_state for {limb} refers to unknown hold {value!r}")
        normalized[limb] = hold_id
    if len(set(normalized.values())) != len(LIMBS):
        raise ValueError("Each limb must start on a different hold")
    return normalized


def is_in_reach(limb, from_hold, to_hold):
    """Cheap limb-length bound before running whole-body IK for a candidate."""
    row_delta = to_hold["row"] - from_hold["row"]
    if row_delta <= 0:
        return False
    distance = float(np.linalg.norm(to_hold["position"] - from_hold["position"]))
    return distance <= LIMB_STEP_LIMIT_M[limb]


def _heuristic(positions, holds, goal_row, turn):
    """Lower bound from the required hand rise and mandatory cyclic moves."""
    # Either hand can reach the goal. Each hand move can rise at most its
    # limb-length bound, and every turn before that hand moves must also advance
    # by at least one grid row. Each move costs at least one row of vertical travel.
    costs = []
    for hand in ("left_hand", "right_hand"):
        row = holds[positions[LIMB_INDEX[hand]]]["row"]
        gap = max(0, goal_row - row)
        if gap == 0:
            return 0.0
        max_rise = max(1, int(LIMB_STEP_LIMIT_M[hand] // GRID_PITCH_M))
        hand_moves = math.ceil(gap / max_rise)
        hand_turn = MOVE_CYCLE.index(hand)
        first_hand_step = (hand_turn - turn) % len(MOVE_CYCLE) + 1
        total_steps = first_hand_step + (hand_moves - 1) * len(MOVE_CYCLE)
        costs.append(total_steps * GRID_PITCH_M)
    return min(costs)


def _reconstruct_moves(came_from, current):
    moves = []
    while current in came_from:
        previous, move = came_from[current]
        moves.append(move)
        current = previous
    moves.reverse()
    return moves


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plan a climbing route on N1 or N2.")
    parser.add_argument("wall", nargs="?", choices=tuple(HOLD_JSON_BY_WALL), default=DEFAULT_WALL)
    args = parser.parse_args()

    plan = climbing_aStar(wall=args.wall)
    if plan is None:
        print("No feasible route found")
    else:
        for step, move in enumerate(plan, start=1):
            hold = move["hold"]
            print(f"{step:02d}. {move['limb']:>10} -> {move['hold_id']} "
                  f"(col={hold['col']}, row={hold['row']})")
