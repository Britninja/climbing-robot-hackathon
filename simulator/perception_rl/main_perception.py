"""Whackathon : RealSense + repere ArUco existant + YOLO11 Small -> holds.json.

Ne modifie pas wall_detector.py : on importe ses fonctions, sans lancer son main.

Execution :
    python main_perception.py
    python main_perception.py --poids /chemin/vers/best.pt

Touches : Q quitter | G grille | S sauver holds.json + apercu | C sauver calibration
          clic gauche : lire les coordonnees du mur en millimetres.

IMPORTANT : col/row sont des indices ESTIMES par arrondi vers le trou le plus proche.
            Le centre de la bounding box YOLO peut etre decale du trou de fixation.
            Les indices sont calcules avec la conversion millimetres -> trous de
            perception_rgb, comme dans live_cameras.py.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE.parent))
sys.path.insert(0, str(BASE.parent.parent))
import perception_rgb as P

try:
    import simulator.perception_rl.wall_detector as mur
except ModuleNotFoundError as exc:
    if exc.name != "wall_detector":
        raise
    import wall_detector_corrige as mur


SORTIE_JSON = BASE / "holds.json"
SORTIE_IMAGE = BASE / "apercu_holds.jpg"
FICHIER_CALIBRATION = BASE / "wall_calibration.json"

GRILLE_COLS = 52
GRILLE_ROWS = 93
PAS_GRILLE_MM = 15.0
FENETRE = "Whackathon - ArUco + YOLO Small"
# YOLO ne distingue pas les couleurs. On utilise la moyenne des offsets mesures
# pour les prises reelles, comme correction du centre visuel vers les boulons.
OFFSETS_PRISE_MM = np.mean([c["offset_mm"] for c in P.COLORS.values()], axis=0)


def trouver_poids(chemin):
    """Retrouve les poids entraines, ou utilise --poids si besoin."""
    if chemin:
        p = Path(chemin).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Poids YOLO introuvables : {p}")
        return p

    candidats = [
        BASE.parent.parent / "runs/detect/runs/detect/prises_small_v2/weights/best.pt",
    ]
    for p in candidats:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Impossible de trouver best.pt. Lance avec :\n"
        "python main_perception.py --poids /chemin/vers/best.pt"
    )


def intrinseques(profil):
    """Retrouve EXACTEMENT le traitement optique du wall_detector existant."""
    intr = profil.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ], dtype=np.float64)
    carte_x = carte_y = None
    if intr.model == rs.distortion.none:
        dist = np.zeros((5, 1), dtype=np.float64)
    elif intr.model == rs.distortion.brown_conrady:
        dist = np.asarray(intr.coeffs, dtype=np.float64).reshape(5, 1)
    elif intr.model == rs.distortion.inverse_brown_conrady:
        # Ne PAS passer les coefficients inverse Brown a cv2.undistortPoints.
        carte_x, carte_y = mur.carte_rectification_inverse_brown(intr)
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        raise RuntimeError(f"Distorsion RealSense non prise en charge : {intr.model}")
    return intr, K, dist, carte_x, carte_y


def lire_calibration_enregistree(K, dist, intr):
    """Autorise la reutilisation uniquement avec memes intrinseques et resolution."""
    if not FICHIER_CALIBRATION.is_file():
        raise FileNotFoundError(f"Calibration absente : {FICHIER_CALIBRATION}")
    data = json.loads(FICHIER_CALIBRATION.read_text(encoding="utf-8"))
    attendus = (mur.LARGEUR_IMAGE, mur.HAUTEUR_IMAGE)
    if tuple(data["resolution"]) != attendus:
        raise RuntimeError("Resolution incompatible avec wall_calibration.json")
    K_enregistre = np.asarray(data["camera_matrix"], dtype=np.float64)
    if not np.allclose(K, K_enregistre, rtol=1e-3, atol=0.5):
        raise RuntimeError("Intrinseques de la camera differents : refaire la calibration.")
    dist_enregistree = np.asarray(data["distortion"], dtype=np.float64).reshape(-1)
    if not np.allclose(dist.reshape(-1), dist_enregistree, atol=1e-5):
        raise RuntimeError("Distorsion differente : refaire la calibration.")
    if data.get("camera_distortion_model_original") != str(intr.model):
        raise RuntimeError("Modele RealSense different : refaire la calibration.")
    coeffs = data.get("camera_distortion_coefficients_original")
    if coeffs is not None and not np.allclose(intr.coeffs, coeffs, atol=1e-5):
        raise RuntimeError("Coefficients RealSense differents : refaire la calibration.")
    if not np.isclose(data["largeur_mur_mm"], mur.LARGEUR_MUR_MM) or not np.isclose(
        data["hauteur_mur_mm"], mur.HAUTEUR_MUR_MM
    ):
        raise RuntimeError("Dimensions du mur differentes dans la calibration.")
    H = np.asarray(data["H_pixel_undistordu_vers_mur"], dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise RuntimeError("Homographie de calibration invalide.")
    return {
        "H": H,
        "rvec": np.asarray(data["rvec"], dtype=np.float64).reshape(3, 1),
        "tvec": np.asarray(data["tvec"], dtype=np.float64).reshape(3, 1),
        "erreur_px": float(data["erreur_reprojection_px"]),
    }


def trouver_prises(modele, image, confiance, appareil):
    prediction = modele.predict(
        source=image,
        conf=confiance,
        iou=0.35,
        agnostic_nms=True,
        imgsz=640,
        device=appareil,
        verbose=False
    )[0]

    if prediction.boxes is None or len(prediction.boxes) == 0:
        return []

    return prediction.boxes.xyxy.cpu().numpy().tolist()


def coordonnees_prise(x_mm, y_mm):
    """Convertit le centre YOLO en coordonnees de la grille comme live_cameras.

    wall_detector exprime les coordonnees depuis le coin bas-gauche du panneau
    entier. perception_rgb attend l'origine de la zone percee, decalee de
    BOARD_ORIGIN_X/Y_MM.
    """
    x_zone = x_mm - P.BOARD_ORIGIN_X_MM
    y_zone = y_mm - P.BOARD_ORIGIN_Y_MM
    return P.mm_to_hole_index(x_zone, y_zone, OFFSETS_PRISE_MM)


def convertir_prises(boites, resultat_mur, K, dist):
    """Position projetee sur le plan du mur, PAS le trou de fixation prouve."""
    prises = []
    if resultat_mur is None:
        return prises
    for x1, y1, x2, y2 in boites:
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x_mm, y_mm = mur.pixel_vers_mur(u, v, resultat_mur["H"], K, dist)
        if not np.isfinite([x_mm, y_mm]).all():
            continue
        if not (0 <= x_mm <= mur.LARGEUR_MUR_MM and 0 <= y_mm <= mur.HAUTEUR_MUR_MM):
            continue
        col, row = coordonnees_prise(x_mm, y_mm)
        if not (0 <= col < GRILLE_COLS and 0 <= row < GRILLE_ROWS):
            continue
        x_zone = x_mm - P.BOARD_ORIGIN_X_MM
        y_zone = y_mm - P.BOARD_ORIGIN_Y_MM
        prises.append({
            "col": col,
            "row": row,
            "x_mm": round(x_zone, 1),
            "y_mm": round(y_zone, 1),
            "color": "Prise",  # Pas de reconnaissance de couleur a ce stade.
            "n_cams": 1,
        })
    prises.sort(key=lambda p: (p["y_mm"], p["x_mm"]))
    return prises


def dessiner_prises(image, boites, resultat_mur, K, dist):
    for x1, y1, x2, y2 in boites:
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 220, 0), 2)
        cv2.circle(image, (round(u), round(v)), 4, (0, 0, 255), -1)
        if resultat_mur is None:
            etiquette = "Prise (repere indisponible)"
        else:
            x_mm, y_mm = mur.pixel_vers_mur(u, v, resultat_mur["H"], K, dist)
            col, row = coordonnees_prise(x_mm, y_mm)
            etiquette = f"C{col} R{row}  X={x_mm:.0f} Y={y_mm:.0f}"
        cv2.putText(image, etiquette, (max(3, int(x1)), max(16, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 235, 0), 1)


def sauvegarder(image, boites, repere, K, dist):
    prises = convertir_prises(boites, repere, K, dist)
    if repere is None or not prises:
        print("Sauvegarde refusee : repere absent ou aucune prise detectee sur le mur.")
        return
    document = {
        "grid": {"cols": GRILLE_COLS, "rows": GRILLE_ROWS, "pitch_mm": PAS_GRILLE_MM},
        "holds": prises,
    }
    SORTIE_JSON.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    if not cv2.imwrite(str(SORTIE_IMAGE), image):
        print("Attention : impossible de sauvegarder l'apercu image.")
    print(f"{len(prises)} prise(s) -> {SORTIE_JSON}")
    print(f"Image annotee -> {SORTIE_IMAGE}")
    print(f"Offset centre visuel -> fixation : {OFFSETS_PRISE_MM[0]:.1f}, {OFFSETS_PRISE_MM[1]:.1f} mm")
    cellules = {}
    for prise in prises:
        cell = (prise["col"], prise["row"])
        cellules[cell] = cellules.get(cell, 0) + 1
    doublons = sum(n - 1 for n in cellules.values() if n > 1)
    print(f"Indices col/row renseignes : {len(prises)} ; collisions de cellules : {doublons}.")


def analyser_arguments():
    p = argparse.ArgumentParser(description="Detection live ArUco + prises YOLO11 Small")
    p.add_argument("--poids", default=None, help="Chemin vers best.pt")
    p.add_argument("--conf", type=float, default=0.35, help="Seuil YOLO (defaut: 0.35)")
    p.add_argument("--pas-yolo", type=int, default=3,
                   help="Lancer YOLO toutes les N images (defaut: 3)")
    p.add_argument("--device", default="cpu", help="cpu, 0 (GPU), etc.")
    p.add_argument("--charger-calibration", action="store_true",
                   help="Utiliser H sauvegardee sans voir les 4 ArUco, CAMERA STRICTEMENT FIXE")
    args = p.parse_args()
    if not 0 <= args.conf <= 1:
        p.error("--conf doit etre entre 0 et 1")
    if args.pas_yolo < 1:
        p.error("--pas-yolo doit etre >= 1")
    return args


def main():
    args = analyser_arguments()
    poids = trouver_poids(args.poids)
    print("Poids YOLO :", poids)
    modele = YOLO(str(poids))

    dictionnaire = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detecteur = cv2.aruco.ArucoDetector(dictionnaire, params)

    clic = {"point": None}
    def souris(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clic["point"] = (x, y)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, mur.LARGEUR_IMAGE,
                         mur.HAUTEUR_IMAGE, rs.format.bgr8, mur.FPS)
    actif = False
    try:
        profil = pipeline.start(config)
        actif = True
        intr, K, dist, carte_x, carte_y = intrinseques(profil)
        print("Modele optique :", intr.model)
        repere_memorise = None
        mode_memorise = None
        if args.charger_calibration:
            repere_memorise = lire_calibration_enregistree(K, dist, intr)
            mode_memorise = "FICHIER (camera fixe)"
            print("Calibration precedente chargee. Ne pas deplacer la camera.")

        cv2.namedWindow(FENETRE)
        cv2.setMouseCallback(FENETRE, souris)
        grille = False
        compteur = 0
        boites = []
        print("Q quitter | G grille | S sauver holds.json | C sauver calibration | clic X,Y")

        while True:
            frames = pipeline.wait_for_frames()
            couleur = frames.get_color_frame()
            if not couleur:
                continue
            image = np.asanyarray(couleur.get_data()).copy()
            if carte_x is not None:
                image = cv2.remap(image, carte_x, carte_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)

            # Les 4 ArUco permettent de recalculer un repere sur cette image.
            repere_actuel, coins, presents = mur.analyse_mur(image, detecteur, K, dist)
            if repere_actuel is not None:
                repere_memorise = repere_actuel
                mode_memorise = "SESSION (camera fixe)"
            repere = repere_actuel if repere_actuel is not None else repere_memorise

            # YOLO et ArUco utilisent la MEME image (eventuellement rectifiee).
            if compteur % args.pas_yolo == 0:
                boites = trouver_prises(modele, image, args.conf, args.device)
            compteur += 1

            affichage = image.copy()
            if coins:
                cv2.aruco.drawDetectedMarkers(
                    affichage, coins, np.array(presents, dtype=np.int32).reshape(-1, 1)
                )
            if repere is not None:
                mur.dessiner_repere(affichage, repere, K, dist, grille)
            dessiner_prises(affichage, boites, repere, K, dist)

            if repere_actuel is not None:
                etat, couleur_etat = f"ArUco OK | {repere['erreur_px']:.2f} px", (0, 255, 0)
            elif repere is not None:
                etat, couleur_etat = f"Repere memorise : {mode_memorise}", (0, 190, 255)
            else:
                etat, couleur_etat = f"ArUco incomplets : {presents}", (0, 0, 255)
            cv2.putText(affichage, etat, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.53, couleur_etat, 2)
            cv2.putText(affichage, f"YOLO : {len(boites)} prise(s)", (8, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if clic["point"] is not None and repere is not None:
                cx, cy = clic["point"]
                mx, my = mur.pixel_vers_mur(cx, cy, repere["H"], K, dist)
                cv2.circle(affichage, (cx, cy), 5, (255, 0, 255), -1)
                cv2.putText(affichage, f"Clic : X={mx:.1f} Y={my:.1f} mm",
                            (8, 68), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, (255, 0, 255), 2)

            cv2.imshow(FENETRE, affichage)
            touche = cv2.waitKey(1) & 0xFF
            if touche == ord("q"):
                break
            if touche == ord("g"):
                grille = not grille
            if touche == ord("c"):
                if repere_actuel is not None:
                    mur.enregistrer(repere_actuel, K, dist, intr)
                else:
                    print("Calibration non sauvee : 4 ArUco requis a cet instant.")
            if touche == ord("s"):
                # Detection YOLO du MEME frame que l'homographie affichee.
                boites = trouver_prises(modele, image, args.conf, args.device)
                sauvegarde_image = image.copy()
                if repere is not None:
                    mur.dessiner_repere(sauvegarde_image, repere, K, dist, grille)
                dessiner_prises(sauvegarde_image, boites, repere, K, dist)
                sauvegarder(sauvegarde_image, boites, repere, K, dist)
    finally:
        if actif:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
