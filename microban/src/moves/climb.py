"""MoveClimb : enchaîne pied droit, pied gauche, main droite, main gauche (touche c)."""
from moves.movePlaco import (
    DEFAULT_LIMB_DURATION_S,
    BasePlacoMove,
    MoveRightLeg,
    MoveLeftLeg,
    MoveRightArm,
    MoveLeftArm,
)

ORDER = (MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm)
DEFAULT_MOVE_DURATION_S = DEFAULT_LIMB_DURATION_S
DEFAULT_CYCLE_COUNT = 13
DEFAULT_FEET_LEVEL_DURATION_S = 0.8
CYCLE_SIZE = len(ORDER)


class MoveClimb:
    options = {
        "duration_s": DEFAULT_MOVE_DURATION_S,
        "cycles": DEFAULT_CYCLE_COUNT,
        "feet_level_duration_s": DEFAULT_FEET_LEVEL_DURATION_S,
    }
    option_docs = {
        "duration_s": "durée d'un mouvement de membre (s)",
        "cycles": f"nombre de cycles ({CYCLE_SIZE} membres par cycle)",
        "feet_level_duration_s": "durée pour redresser les deux pieds avant chaque cycle (s)",
    }
    variants = ("climber",)

    def __init__(self):
        self.options = dict(self.options)
        self.moves = [cls() for cls in ORDER]

    def _next_limb(self):
        """Passe au membre suivant et prépare sa trajectoire."""
        self.current = self.moves[self.limb_index]
        self.current.options["duration_s"] = self.options["duration_s"]
        first_left_foot_takeoff = self.count == 1 and self.current.LIMB == "left_foot"
        if first_left_foot_takeoff:
            start_position = BasePlacoMove.frame_position_world("left_foot")
            self.current.begin(start_position=start_position, initial_ground_takeoff=True)
        else:
            self.current.begin()
        self.limb_index = (self.limb_index + 1) % len(self.moves)
        self.count += 1

    def start(self, obs):
        BasePlacoMove.setup(obs)  # repart de la pose de départ à chaque appui sur c
        self.limb_index = 0
        self.count = 0
        self.movement_time = 0.0
        self.preparing_feet = True
        self.preparation_time = 0.0
        self.finished = False
        BasePlacoMove.prepare_feet_for_cycle()

    def step(self, obs):
        if self.finished:
            return

        dt = BasePlacoMove.time_step(obs)
        if self.preparing_feet:
            self.preparation_time += dt
            if self.preparation_time >= self.options["feet_level_duration_s"]:
                self.preparing_feet = False
                self._next_limb()
                self.movement_time = 0.0
                self.current.set_target(0.0)
        else:
            self.movement_time += dt
            if self.movement_time <= self.options["duration_s"]:
                self.current.set_target(self.movement_time)
            elif self.count >= CYCLE_SIZE * self.options["cycles"]:
                self.finished = True
            elif self.count % CYCLE_SIZE == 0:
                self.preparing_feet = True
                self.preparation_time = 0.0
                BasePlacoMove.prepare_feet_for_cycle()
            else:
                self._next_limb()
                self.movement_time = 0.0
                self.current.set_target(0.0)

        return BasePlacoMove.solve(dt)
