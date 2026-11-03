import json
from pathlib import Path
import time

import numpy as np
import placo
import microban_model
from placo_utils.visualization import robot_viz, points_viz
from simulator.climbing_sim import JOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAP_PATH = PROJECT_ROOT / "simulator/climbing_sim/walls" / "N1.json"


hole_dist = 0.015  # 15mm as cm / 100
ball_rad = 0.0125  # cm / 100
ball_center_wall_dist = 0.026  # cm
first_hole_floor_dist = hole_dist / 2  # ?
wall_start_leg_dist = 5 * hole_dist
wall_start_arm_dist = 11 * hole_dist
wall_start_arm_leg_dist = 3 * hole_dist  # (left-left or right-right)
with open(MAP_PATH, "r") as f:
    _HOLDS_MAP = json.load(f)

_START_HOLDS = {
    "right_foot": _HOLDS_MAP["holds"][0],
    "left_foot": _HOLDS_MAP["holds"][1],
    "right_hand": _HOLDS_MAP["holds"][2],
    "left_hand": _HOLDS_MAP["holds"][3],
}

MJCF_PATH = str(microban_model.mjcf(climb=True))


class ClimbMove:
    start_pose = "current"
    options = {"y_spacing": wall_start_leg_dist, "z_spacing": wall_start_leg_dist}
    option_docs = {
        "y_spacing": "Distance left foot (to right foot)",
        "z_spacing": "hopefully hold_distance (rows)",
    }
    variants = ("climber",)

    def __init__(self) -> None:
        self._robot = placo.RobotWrapper(
            MJCF_PATH, placo.Flags.mjcf + placo.Flags.ignore_collisions
        )
        self._solver = placo.KinematicsSolver(self._robot)

        # left_foot_start coordinates
        # x = towards the wall?, y left right, z hight
        left_foot_start = [0.0] * 3
        right_foot_start = [0.0] * 3
        right_hand_start = [0.0] * 3
        left_hand_start = [0.0] * 3
        left_foot_start[0] = -ball_center_wall_dist
        left_foot_start[1] = _START_HOLDS["left_foot"]["col"] * hole_dist
        left_foot_start[2] = _START_HOLDS["left_foot"]["row"] * hole_dist
        right_foot_start[0] = -ball_center_wall_dist
        right_foot_start[1] = _START_HOLDS["right_foot"]["col"] * hole_dist
        right_foot_start[2] = _START_HOLDS["right_foot"]["row"] * hole_dist
        right_hand_start[0] = -ball_center_wall_dist
        right_hand_start[1] = _START_HOLDS["right_hand"]["col"] * hole_dist
        right_hand_start[2] = _START_HOLDS["right_hand"]["row"] * hole_dist
        left_hand_start[0] = -ball_center_wall_dist
        left_hand_start[1] = _START_HOLDS["left_hand"]["col"] * hole_dist
        left_hand_start[2] = _START_HOLDS["left_hand"]["row"] * hole_dist

        # Limit knee angles
        self._robot.set_joint_limits("left_knee", 0.01, 2.3)
        self._robot.set_joint_limits("right_knee", 0.01, 2.3)

        # Ball tasks
        right_foot_task = self._solver.add_position_task(
            "right_foot_ball", np.array(right_foot_start)
        )
        left_foot_task = self._solver.add_position_task(
            "left_foot_ball", np.array(left_foot_start)
        )
        right_hand_task = self._solver.add_position_task(
            "right_hand_ball", np.array(right_hand_start)
        )
        left_hand_task = self._solver.add_position_task(
            "left_hand_ball", np.array(left_hand_start)
        )
        self._ball_frames = [
            "right_foot_ball",
            "left_foot_ball",
            "right_hand_ball",
            "left_hand_ball",
        ]
        self._tasks = [right_foot_task, left_foot_task, right_hand_task, left_hand_task]
        self._limb_starts = [
            right_foot_start,
            left_foot_start,
            right_hand_start,
            left_hand_start,
        ]
        for i, ball_task in enumerate(self._tasks):
            ball_task.configure(self._ball_frames[i], "soft", 100)
            # ball_task.target_world = self._robot.get_T_world_frame(self._ball_frames[i])[:3, 3].copy()

        # Keep trunk parallel to the world
        body_orientation_task = self._solver.add_orientation_task("body", np.eye(3))
        body_orientation_task.configure("body_orientation", "soft", 1e-3)

        # Joints -> keep them close to their zero position
        joints_task = self._solver.add_joints_task()
        joints_task.configure("joints", "soft", 1e-6)

        # TODO: do i need this?
        joints_task.set_joints({joint: 0 for joint in JOINTS})

        # I think this is this for not moving the body much in general?
        self._solver.add_regularization_task(1e-3)
        self._solver.enable_joint_limits(True)

        # "Holds" position (just for visualization)
        # x = into wall (red), y = green = left/right, z (blue) = height
        print("left_foot_start")
        print(left_foot_start)
        self._vis_holds = np.array(
            [
                [left_foot_start[0], left_foot_start[1] + y, left_foot_start[2] + z]
                for y in [0, self.options["y_spacing"]]
                for z in np.linspace(start=0, stop=5 * self.options["z_spacing"], num=5)
            ]
        )
        print("self._vis_holds")
        print(self._vis_holds)
        points_viz("holds", self._vis_holds)

        # leg_dist_task = self._solver.add_distance_task(
        #     "left_foot_ball",
        #     "right_foot_ball",
        #     float(wall_start_leg_dist),
        # )
        # leg_dist_task.configure("leg distance", "soft", 1e-3)

    def start(self, obs) -> None:
        print("\nSTART: CLIMB MOVE\n")

        for _ in range(50):
            self._solver.solve(True)
            self._robot.update_kinematics()
            
        self._task_rotation_index = 0
        self._limb_move_status = "ready"
        self._viz = robot_viz(self._robot)

    def calc_target_update_waypoints(self, task, part_duration):
        prev_target = task.target_world  # Start trajectory at the old target

        # x = rouge dans le mur; y gauche vert, bleue z,  (frame viz with identity (origin))

        x = [
            0,  # 0
            0,  # 1
            (-2 * ball_center_wall_dist) - 0.005,  # 2
            (-2 * ball_center_wall_dist) - 0.005,  # 3
            0.005,  # 4
            0,  # 5
            # 6
        ]  # x: into wall (- is away from wall)

        z = [
            0,
            3 * hole_dist,
            3 * hole_dist,
            3 * hole_dist,
            3 * hole_dist,
            3 * hole_dist,
        ]  # height
        n_steps = len(x)
        way_pts = [[]] * n_steps
        for i in range(n_steps):
            way_pts[i] = [
                (i / (n_steps - 1)) * part_duration,
                prev_target + [x[i], 0, z[i]],  # y = [0, 0, 0, 0, 0, 0]  # y: gauche
            ]
        return way_pts

    def step(self, obs):
        if self._limb_move_status == "finished":
            return {}
        return self._move_limb(obs, self._tasks[self._task_rotation_index])

    part_duration = 2.0  # [s]

    def _move_limb(self, obs, task):
        match self._limb_move_status:
            case "ready":
                print(f"ready: {self._ball_frames[self._task_rotation_index]}")
                self._waypoints = self.calc_target_update_waypoints(
                    task, self.part_duration
                )
                self._trajectory = placo.CubicSpline3D()
                for t, pos in self._waypoints:
                    self._trajectory.add_point(t, pos, np.zeros(3))
                # print(self._trajectory.duration())
                self._limb_move_status = "running"
                self._limb_move_t0 = obs["time"]
                print(f"last waypoint: {self._waypoints[-1][-1]}")
                print(f"first_target: {task.target_world}")
                return {}
            case "running":
                print(f"running")
                time_for_pos = obs["time"] - self._limb_move_t0
                print(f"time_for_pos: {time_for_pos}")
                if time_for_pos < self.part_duration:
                    # s = min(time_for_pos / max(self.part_duration, 0.01), 1.0)
                    # if s > self.part_duration:
                    #                    print(s)
                    pos_delta = self._trajectory.pos(time_for_pos)
                    limb_start = self._limb_starts[
                        self._task_rotation_index
                    ]  # right_foot: [-0.026, 0.39, 0.03]
                    print(pos_delta)
                    task.target_world = pos_delta

                    self._solver.solve(True)
                    self._robot.update_kinematics()

                    self._viz.display(self._robot.state.q)
                else:
                    self._limb_move_status = "finished"

                print(f"target_world: {task.target_world}")
                # print(f"ball position: {self._robot.get_joint("left_foot_ball")}")

                return {name: self._robot.get_joint(name) for name in JOINTS}
            case "finished":
                print("finished")
                self._task_rotation_index = (self._task_rotation_index + 1) % 4
                self._limb_move_status = "ready"
                return {}

    def _cycle_limbs(self, obs):
        for task in self._tasks:
            self._move_limb(obs, task)
        return {name: self._robot.get_joint(name) for name in JOINTS}
