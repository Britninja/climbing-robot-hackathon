"""MoveClimb : enchaîne pied droit, pied gauche, main droite, main gauche (touche c)."""
import math

from microban_model import JOINTS
from moves.movePlaco import BasePlacoMove, MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm

ORDER = (MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm)


class MoveClimb:
    options = {"duration_s": 2.0, "cycles": 13, "support_current_A": 0.2}
    option_docs = {
        "duration_s": "durée d'un mouvement de membre (s)",
        "cycles": "nombre de cycles (4 membres par cycle)",
        "support_current_A": "courant minimal du membre en appui (A)",
    }
    variants = ("climber",)

    def __init__(self):
        self.options = dict(self.options)
        self.moves = [cls() for cls in ORDER]

    def _next_limb(self):
        """Passe au membre suivant et prépare sa trajectoire."""
        self.current = self.moves[self.limb_index]
        self.current.options["duration_s"] = self.options["duration_s"]
        self.current.begin()
        self.limb_index = (self.limb_index + 1) % len(self.moves)
        self.count += 1

    def _has_support(self, obs):
        """Vérifie que les moteurs du membre qui vient de bouger fournissent un effort."""
        currents = obs.get("current")
        if currents is None:
            return True  # certaines sources d'observation ne fournissent pas les courants

        side, limb = self.current.LIMB.split("_")
        joints = (
            ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")
            if limb == "foot"
            else ("shoulder_pitch", "shoulder_roll", "elbow")
        )
        try:
            readings = [
                abs(float(currents[JOINTS.index(f"{side}_{joint}")]))
                for joint in joints
                if math.isfinite(float(currents[JOINTS.index(f"{side}_{joint}")]))
            ]
        except (IndexError, KeyError, TypeError, ValueError):
            return True  # une lecture indisponible ne doit pas bloquer le mouvement
        if not readings:
            return True
        return sum(readings) >= self.options["support_current_A"]

    def start(self, obs):
        BasePlacoMove.setup(obs)  # repart de la pose de départ à chaque appui sur c
        self.limb_index = 0
        self.count = 0
        self.movement_time = 0.0
        self.checking_support = False
        self._next_limb()

    def step(self, obs):
        dt = BasePlacoMove.time_step(obs)
        self.movement_time += dt

        if not self.checking_support:
            self.current.set_target(min(self.movement_time, self.options["duration_s"]))
            targets = BasePlacoMove.solve(dt)
            if self.movement_time >= self.options["duration_s"]:
                self.checking_support = True
            return targets

        # Vérifier au tick suivant, après avoir envoyé la cible finale du membre.
        if not self._has_support(obs):
            self.current.retry()
            self.movement_time = 0.0
            self.checking_support = False
        elif self.count >= 4 * self.options["cycles"]:
            return  # tous les membres ont fini et leur appui a été vérifié
        else:
            self._next_limb()
            self.movement_time = 0.0
            self.checking_support = False

        self.current.set_target(0.0)
        return BasePlacoMove.solve(dt)
