"""MoveClimb : enchaîne pied droit, pied gauche, main droite, main gauche (touche c)."""
from moves.movePlaco import BasePlacoMove, MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm

ORDER = (MoveRightLeg, MoveLeftLeg, MoveRightArm, MoveLeftArm)


class MoveClimb:
    options = {"dz_cm": 7.5, "duration_s": 2.0, "cycles": 13}
    option_docs = {
        "dz_cm": "montée par mouvement (cm)",
        "duration_s": "durée d'un mouvement (s)",
        "cycles": "nombre de cycles (4 membres par cycle)",
    }
    variants = ("climber",)

    def __init__(self):
        self.moves = [cls() for cls in ORDER]

    def start(self, obs):
        BasePlacoMove.initialized = False  # les cibles repartent de la pose actuelle
        self.count = 0
        self.current = None

    def step(self, obs):
        # Mouvement fini (ou pas encore commencé) : on passe au membre suivant
        if self.current is None or self.current.done:
            if self.count >= 4 * self.options["cycles"]:
                return  # escalade terminée
            self.current = self.moves[self.count % 4]
            self.current.options.update(dz_cm=self.options["dz_cm"], duration_s=self.options["duration_s"])
            self.current.start(obs)
            self.count += 1
        return self.current.step(obs)