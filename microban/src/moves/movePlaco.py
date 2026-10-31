import math
from pathlib import Path

import numpy as np
import placo
import microban_model
from microban_model import JOINTS


def find_urdf():
    """robot_climb.urdf à côté de ce fichier, sinon dans cad/, sinon l'URDF climb de microban_model."""
    for path in (Path(__file__).with_name("robot_climb.urdf"), Path("cad/robot_climb.urdf")):
        if path.is_file():
            return str(path)
    return str(microban_model.urdf(climb=True))

LIMB_FRAMES = {                     # membre -> repère URDF qui touche le mur
    "left_hand": "left_hand_ball",
    "right_hand": "right_hand_ball",
    "left_foot": "left_foot_ball",
    "right_foot": "right_foot_ball",
}
Z_SPACING = 0.075                   # distance entre deux barreaux (m)

# Position de départ des 4 boules dans le repère Placo (x = vers le mur, y = gauche, z = haut)
START_POS = {
    "left_hand": np.array([0.08417194, 0.08258263, 0.16097954]),
    "right_hand": np.array([0.08419343, -0.08261053, 0.16094472]),
    "left_foot": np.array([0.09276341, 0.03737545, -0.14360264]),
    "right_foot": np.array([0.09277957, -0.03763053, -0.14360031]),
}

# Décalage ajouté au point "juste avant de se poser" (réglé à la main en simulation)
COMPENSATION = {
    "left_foot": [0.015, 0.01, 0.025],
    "right_foot": [0.015, -0.005, 0.005],
    "left_hand": [0.04, -0.02, 0.05],
    "right_hand": [0.12, 0.02, 0.05],
}

TOE_IN_DEG = 10.0          # de combien les pointes de pied tournent vers l'intérieur
TOE_IN_SIGN = -1.0         # inverser si les pieds tournent du mauvais côté
FOOT_REST_WEIGHT = 1e-3    # garde aussi l'angle du pied en appui (0.0 = ancien comportement)

# Angle de repos par articulation (rad) : la tâche articulaire tire doucement vers ces valeurs
DEFAULT_FALLBACK = 0.0
DEFAULT_JOINT_ANGLES = {
    "left_hip_pitch": -2.0,
    "right_hip_pitch": -2.0,
    "left_knee": -1.0,
    "right_knee": -1.0,
    "left_shoulder_pitch": -2.0,
    "right_shoulder_pitch": -2.0,
    "left_elbow": 2.0,
    "right_elbow": 2.0,
    "left_ankle_pitch": 1.0,
    "right_ankle_pitch": 1.0,
}


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


class BasePlacoMove:
    options = {"duration_s": 2.0}
    option_docs = {"duration_s": "durée du mouvement (s)"}
    variants = ("climber",)

    LIMB = None  # défini par les classes enfants

    # PARTAGÉ par tous les mouvements
    robot = None
    solver = None
    tasks = {}       # membre -> tâche de position
    foot_ori = {}    # pied -> tâche d'orientation
    last_time = 0.0  # temps du dernier pas (partagé : le temps continue d'un membre à l'autre)

    def __init__(self):
        self.options = dict(self.options)
        self.done = False

    @staticmethod
    def sync(obs):
        """Recopie les angles mesurés dans le modèle Placo."""
        B = BasePlacoMove
        for name in B.robot.joint_names():
            B.robot.set_joint(name, obs["q"][JOINTS.index(name)])
        B.robot.update_kinematics()

    @staticmethod
    def setup(obs):
        """Crée le robot Placo et toutes les tâches (les boules visent leur position de départ)."""
        B = BasePlacoMove
        urdf = find_urdf()
        print(f"[movePlaco] URDF : {urdf}")
        B.robot = placo.RobotWrapper(urdf, placo.Flags.collision_as_visual)
        B.solver = placo.KinematicsSolver(B.robot)
        B.solver.mask_fbase(False)
        B.sync(obs)
        B.last_time = obs["time"]

        # Une tâche de position par boule
        B.tasks = {}
        for limb, frame in LIMB_FRAMES.items():
            B.tasks[limb] = B.solver.add_position_task(frame, START_POS[limb].copy())
            B.tasks[limb].configure(f"{limb}_pos", "soft", 100.0)

        # Pieds : à plat sur le mur, pointes légèrement vers l'intérieur
        toe_in = TOE_IN_SIGN * math.radians(TOE_IN_DEG)
        B.foot_ori = {
            "right_foot": B.solver.add_orientation_task(LIMB_FRAMES["right_foot"], rot_x(-toe_in)),
            "left_foot": B.solver.add_orientation_task(LIMB_FRAMES["left_foot"], rot_x(toe_in)),
        }
        for foot, task in B.foot_ori.items():
            task.configure(f"{foot}_ori", "soft", FOOT_REST_WEIGHT)

        body_orientation_task = B.solver.add_orientation_task("body", np.eye(3))
        body_orientation_task.configure("body_orientation", "soft", 1e-4)

        # Limites et régularisation
        B.robot.set_joint_limits("left_knee", 0.01, 2.3)
        B.robot.set_joint_limits("right_knee", 0.01, 2.3)
        B.robot.set_joint_limits("left_ankle_pitch", -0.7, 0.55)
        B.robot.set_joint_limits("right_ankle_pitch", -0.7, 0.55)
        B.solver.enable_joint_limits(True)

        joints = B.solver.add_joints_task()
        joints.configure("joints", "soft", 1e-5)
        joints.set_joints({n: DEFAULT_JOINT_ANGLES.get(n, DEFAULT_FALLBACK) for n in B.robot.joint_names()})
        B.solver.add_regularization_task(1e-3)

        unknown = set(DEFAULT_JOINT_ANGLES) - set(B.robot.joint_names())
        if unknown:
            print(f"WARNING: unknown joints in DEFAULT_JOINT_ANGLES: {unknown}")

    def begin(self):
        """Prépare le mouvement de ce membre : orientation des pieds et trajectoire."""
        B = BasePlacoMove
        print(f"------------\nMoving limb: {self.LIMB}")

        # Le pied actif reste à plat, l'autre garde juste un peu son angle
        for foot, task in B.foot_ori.items():
            task.configure(f"{foot}_ori", "soft", 1.0 if foot == self.LIMB else FOOT_REST_WEIGHT)

        # Trajectoire : se lève, recule, monte, revient au-dessus de la prise, se pose
        T, z = self.options["duration_s"], Z_SPACING
        p = B.tasks[self.LIMB].target_world.copy()  # part de la dernière cible (pas de dérive)
        self.start_position = p.copy()
        waypoints = [
            [0.0, p],
            [T * 0.2, p + np.array([0, 0, 0.04])],
            [T * 0.4, p + np.array([-0.03, 0, z / 2])],
            [T * 0.6, p + np.array([-0.03, 0, z * 1.6])],
            [T * 0.8, p + np.array([0, 0, z]) + COMPENSATION[self.LIMB]],
            [T, p + np.array([0, 0, z])],
        ]
        self.spline = placo.CubicSpline3D()
        for t, pos in waypoints:
            self.spline.add_point(t, pos, np.zeros(3))
        self.movement_time = 0.0
        self.done = False

    def set_target(self, t):
        """La boule de ce membre vise le point de sa trajectoire à l'instant t."""
        BasePlacoMove.tasks[self.LIMB].target_world = self.spline.pos(t)

    def retry(self):
        """Rejoue toute la trajectoire depuis sa position de départ."""
        BasePlacoMove.tasks[self.LIMB].target_world = self.start_position.copy()
        self.begin()

    @staticmethod
    def time_step(obs):
        """Recopie les angles mesurés et renvoie le dt depuis le dernier pas."""
        B = BasePlacoMove
        B.sync(obs)
        now = obs["time"]
        dt = now - B.last_time
        B.last_time = now
        if dt <= 0.0:
            dt = 1e-3
        return dt

    @staticmethod
    def solve(dt):
        """Résout la cinématique, intègre les vitesses articulaires et renvoie les consignes."""
        B = BasePlacoMove
        B.solver.dt = dt
        try:
            B.solver.solve(True)
        except Exception:
            B.solver.solve(False)
        for name in B.robot.joint_names():
            B.robot.set_joint(name, B.robot.get_joint(name) + B.robot.get_joint_velocity(name) * dt)
        B.robot.update_kinematics()
        return {name: B.robot.get_joint(name) for name in B.robot.joint_names()}

    # --- Lancer un seul membre (touche du membre) ---

    def start(self, obs):
        B = BasePlacoMove
        if B.robot is None:
            B.setup(obs)
        B.last_time = obs["time"]
        self.begin()

    def step(self, obs):
        dt = self.time_step(obs)
        self.movement_time += dt
        if self.movement_time <= self.options["duration_s"]:
            self.set_target(self.movement_time)
        else:
            self.done = True  # la boule reste sur sa dernière cible
        return self.solve(dt)


# --- Déclinaisons pour chaque membre ---

class MoveRightArm(BasePlacoMove):
    LIMB = "right_hand"


class MoveLeftArm(BasePlacoMove):
    LIMB = "left_hand"


class MoveRightLeg(BasePlacoMove):
    LIMB = "right_foot"


class MoveLeftLeg(BasePlacoMove):
    LIMB = "left_foot"
