"""Tâche 8.5 : voltige — looping, tonneau, Immelmann, Split-S.

Chaque figure est une suite de **segments** de référence, décrits dans le repère de la
manœuvre (figé au départ : e₁ = route initiale à l'horizontale, e₂ = à droite, e₃ = vers
le bas) :

* ``Pitch(Δθ)`` : le vecteur vitesse tourne de Δθ **dans le plan vertical** (e₁, e₃),
  portance dans ce plan, vers le centre de la boucle (Δθ > 0 : vers le haut au départ) ;
* ``Roll(Δφ)`` : l'avion tourne de Δφ **autour de son vecteur vitesse**, trajectoire
  inchangée.

| Figure     | Segments                          | Sortie                          |
|------------|-----------------------------------|---------------------------------|
| looping    | Pitch(+360°)                      | même cap, ailes à plat          |
| tonneau    | Roll(±360°)                       | même cap, ailes à plat          |
| Immelmann  | Pitch(+180°), Roll(±180°)         | cap inversé, plus haut          |
| Split-S    | Roll(±180°), Pitch(−180°)         | cap inversé, plus bas           |

Les angles sont suivis **sans singularité** à partir du repère vent (``FlightGeometry``) :
un looping passe par la verticale, où l'inclinaison μ et la route χ ne sont plus définies.

Récompense : progression le long des segments (**nouveaux records** seulement, en
fraction de l'angle total : 40 points pour la figure complète + 20 à la sortie), écarts à
la référence (sortie du plan de la boucle, portance mal orientée, trajectoire qui dévie pendant un
tonneau), petit coût par pas, puis, une fois les segments parcourus, retour ailes à plat.
La figure est réussie (``done``) quand tous les segments sont parcourus et que l'avion est
stabilisé (|μ| < 15°, |γ| < 10°) au bon cap (± 30° du cap initial, ou de son inverse
pour l'Immelmann et le Split-S). Le tangage n'est compté que si la vitesse reste près du
plan de la figure (|v·e₂| < sin 30°).

Pièges rencontrés à l'entraînement : avec une progression de 10 points seulement, l'agent
préférait ne pas faire la figure (voler droit ne coûte rien pendant un segment de
tangage) ; avec une progression réversible (ΔΦ), une tentative ratée (monter puis
retomber) ne rapportait rien et l'agent n'essayait pas ; avec des angles d'Euler dans
l'observation, un agent montait à la verticale et y restait pour garder sa progression ;
avec des pénalités de suivi de 0.2 par pas, une tentative ratée laissait l'avion hors du
plan pour le reste de l'épisode (jusqu'à −60 points) et l'agent cessait d'essayer.

Pilote de référence (``baseline_command``) : suivi scripté des segments (facteur de
charge constant et roulis pour garder la portance dans le plan, ou roulis à taux fixe
en compensant la pesanteur), puis pilote automatique en palier.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from jetfighter.aircraft.instruments import Instruments
from jetfighter.control.autopilot import Autopilot, AutopilotTargets
from jetfighter.control.fbw import HighLevelCommand
from jetfighter.envs.rewards import RewardTerms, action_smoothness, angle_error
from jetfighter.envs.tasks.base import DEG, FlightGeometry, InitialCondition, Task, Vec, clip

PITCH, ROLL = "pitch", "roll"
MANEUVERS: tuple[str, ...] = ("loop", "roll", "immelmann", "split_s")

# Conditions d'entrée : (altitude min, max), (vitesse min, max) [m, m/s]
ENTRY: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "loop": ((3000, 7000), (240, 300)),
    "roll": ((3000, 8000), (180, 280)),
    "immelmann": ((3000, 7000), (250, 300)),
    "split_s": ((5500, 8500), (150, 190)),
}


def segments_for(maneuver: str, roll_sign: float) -> list[tuple[str, float]]:
    if maneuver == "loop":
        return [(PITCH, 2 * math.pi)]
    if maneuver == "roll":
        return [(ROLL, roll_sign * 2 * math.pi)]
    if maneuver == "immelmann":
        return [(PITCH, math.pi), (ROLL, roll_sign * math.pi)]
    if maneuver == "split_s":
        return [(ROLL, roll_sign * math.pi), (PITCH, -math.pi)]
    raise ValueError(f"Figure inconnue : {maneuver!r} (disponibles : {MANEUVERS})")


def _signed_angle(a: Vec, b: Vec, axis: Vec) -> float:
    """Angle signé de ``a`` vers ``b`` autour de ``axis`` (vecteurs unitaires)."""
    return math.atan2(float(np.cross(a, b) @ axis), float(a @ b))


@dataclass
class AerobaticsTask(Task):
    name: str = "aerobatics"
    maneuvers: tuple[str, ...] = MANEUVERS  # figures tirées au hasard à chaque épisode
    episode_time: float = 45.0
    segment_tolerance: float = 5 * DEG
    exit_bank: float = 15 * DEG
    exit_gamma: float = 10 * DEG
    exit_heading: float = 30 * DEG  # écart de cap toléré à la sortie (cap initial ou inverse)
    max_out_of_plane: float = math.sin(30 * DEG)  # |v·e₂| au-delà duquel le tangage ne compte plus
    w_progress: float = 40.0  # réparti sur toute la figure
    w_plane: float = 0.05  # faibles : pénalités cumulées sur 45 s -> l'agent n'essayait plus
    w_lift: float = 0.05
    w_track: float = 0.1
    w_exit: float = 0.2
    w_time: float = 0.01
    w_smooth: float = 0.05
    completion_bonus: float = 20.0
    pull_load_factor: float = 5.0  # pilote de référence
    n_features: int = 17
    event_terms: frozenset[str] = frozenset({"progress", "success"})
    maneuver: str = field(default="loop", init=False)
    segments: list[tuple[str, float]] = field(default_factory=list, init=False)
    index: int = field(default=0, init=False)
    seg_progress: float = field(default=0.0, init=False)  # angle parcouru (signé)
    frame: Vec = field(default_factory=lambda: np.eye(3), init=False)  # colonnes e1, e2, e3
    origin: Vec = field(default_factory=lambda: np.zeros(3), init=False)
    _geo: FlightGeometry | None = field(default=None, init=False)
    _prev_theta: float = field(default=0.0, init=False)
    _prev_lift: Vec = field(default_factory=lambda: np.zeros(3), init=False)
    _seg_start_velocity: Vec = field(default_factory=lambda: np.zeros(3), init=False)
    _last_delta: float = field(default=0.0, init=False)
    _seg_best: float = field(default=0.0, init=False)  # meilleure progression du segment
    _completed: bool = field(default=False, init=False)
    _finished_time: float = field(default=math.inf, init=False)
    _exit_state: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0), init=False)
    _pending_maneuver: str | None = field(default=None, init=False)
    _ap_engaged: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.maneuvers = tuple(self.maneuvers)
        for m in self.maneuvers:
            segments_for(m, 1.0)  # validation

    @property
    def max_step_cost(self) -> float:
        shaping = max(self.w_plane + self.w_lift, self.w_track, 2 * self.w_exit)
        # reculer coûte au plus ≈ 0.05 rad par pas, sur un angle total ≥ π
        return shaping + self.w_time + self.w_smooth + 0.016 * self.w_progress

    # ------------------------------------------------------------------
    def sample_initial(self, rng: np.random.Generator, six_dof: bool) -> InitialCondition:
        m = self.maneuvers[int(rng.integers(len(self.maneuvers)))]
        self._pending_maneuver = m
        (h_lo, h_hi), (v_lo, v_hi) = ENTRY[m]
        v_hi = min(v_hi, 260.0) if six_dof else v_hi
        return InitialCondition(
            altitude=float(rng.uniform(h_lo, h_hi)),
            airspeed=float(rng.uniform(min(v_lo, v_hi - 10), v_hi)),
            heading=float(rng.uniform(-math.pi, math.pi)),
        )

    def reset(
        self, ins: Instruments, geo: FlightGeometry, rng: np.random.Generator, six_dof: bool
    ) -> None:
        m = self._pending_maneuver or self.maneuvers[int(rng.integers(len(self.maneuvers)))]
        self._pending_maneuver = None
        self.maneuver = m
        self.segments = segments_for(m, 1.0 if rng.random() < 0.5 else -1.0)
        v = geo.velocity_dir
        e1 = np.array([v[0], v[1], 0.0])
        e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
        e3 = np.array([0.0, 0.0, 1.0])
        self.frame = np.column_stack([e1, np.cross(e3, e1), e3])
        self.origin = geo.position.copy()
        self.index = 0
        self._completed = False
        self._finished_time = math.inf
        self._ap_engaged = False
        self._start_segment(geo)
        self._geo = geo

    def _start_segment(self, geo: FlightGeometry) -> None:
        self.seg_progress = 0.0
        self._last_delta = 0.0
        self._seg_best = 0.0
        self._prev_theta = self._plane_angle(geo)
        self._prev_lift = geo.lift_dir.copy()
        self._seg_start_velocity = geo.velocity_dir.copy()

    def _plane_angle(self, geo: FlightGeometry) -> float:
        v = geo.velocity_dir
        return math.atan2(-float(v @ self.frame[:, 2]), float(v @ self.frame[:, 0]))

    # ------------------------------------------------------------------
    @property
    def total_angle(self) -> float:
        return sum(abs(a) for _, a in self.segments)

    @property
    def segment(self) -> tuple[str, float] | None:
        return self.segments[self.index] if self.index < len(self.segments) else None

    def progress_fraction(self) -> float:
        done = sum(abs(a) for _, a in self.segments[: self.index])
        seg = self.segment
        if seg is not None:
            done += clip(math.copysign(1.0, seg[1]) * self.seg_progress, 0.0, abs(seg[1]))
        return done / self.total_angle

    def desired_lift(self, geo: FlightGeometry, sign: float) -> Vec:
        """Portance voulue dans un segment de tangage : dans le plan, vers le centre."""
        d: Vec = np.asarray(sign * np.cross(self.frame[:, 1], geo.velocity_dir), dtype=np.float64)
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-9 else geo.lift_dir

    def lift_error(self, geo: FlightGeometry) -> float:
        """Angle signé portance -> portance voulue autour de la vitesse (0 hors tangage)."""
        seg = self.segment
        if seg is None or seg[0] != PITCH:
            return 0.0
        d = self.desired_lift(geo, math.copysign(1.0, seg[1]))
        return _signed_angle(geo.lift_dir, d, geo.velocity_dir)

    def update(self, ins: Instruments, geo: FlightGeometry, t: float) -> None:
        self._geo = geo
        self._last_delta = 0.0
        seg = self.segment
        if seg is not None:
            kind, angle = seg
            if kind == PITCH:
                theta = self._plane_angle(geo)
                delta = angle_error(theta - self._prev_theta)
                self._prev_theta = theta
                # hors du plan de la figure, l'angle projeté n'a plus de sens : sans cette
                # garde, un agent tournait de 90° puis « enroulait » l'angle en quelques
                # secondes par de petits mouvements autour de l'axe latéral (looping en 6 s)
                if abs(float(geo.velocity_dir @ self.frame[:, 1])) > self.max_out_of_plane:
                    delta = 0.0
            else:
                delta = _signed_angle(self._prev_lift, geo.lift_dir, geo.velocity_dir)
            self._prev_lift = geo.lift_dir.copy()
            self.seg_progress += delta
            sign = math.copysign(1.0, angle)
            after = clip(sign * self.seg_progress, 0.0, abs(angle))
            # seule une progression nouvelle rapporte (record du segment) ; reculer ne
            # coûte rien : une tentative partielle reste payante, ce qui encourage l'essai
            self._last_delta = max(after - self._seg_best, 0.0)
            self._seg_best = max(self._seg_best, after)
            if sign * self.seg_progress >= abs(angle) - self.segment_tolerance:
                self.index += 1
                if self.segment is not None:
                    self._start_segment(geo)
        if self.segment is None and not self._completed and self._exit_ok(ins, geo):
            self._completed = True
            self._finished_time = t
            self._exit_state = self._exit_errors(ins, geo)

    def _exit_ok(self, ins: Instruments, geo: FlightGeometry) -> bool:
        heading_error = abs(self._exit_errors(ins, geo)[0])
        return (abs(angle_error(ins.bank)) < self.exit_bank and abs(ins.gamma) < self.exit_gamma
                and heading_error < self.exit_heading)  # fmt: skip

    def _exit_errors(self, ins: Instruments, geo: FlightGeometry) -> tuple[float, float, float]:
        """(écart de cap à la sortie attendue [rad], écart latéral [m], Δh [m])."""
        reverse = self.maneuver in ("immelmann", "split_s")
        e1 = self.frame[:, 0]
        expected = math.atan2(e1[1], e1[0]) + (math.pi if reverse else 0.0)
        offset = geo.position - self.origin
        return (
            angle_error(ins.course - expected),
            float(offset @ self.frame[:, 1]),
            float(-offset @ self.frame[:, 2]),
        )

    # ------------------------------------------------------------------
    def features(self, ins: Instruments) -> Vec:
        geo = self._geo
        assert geo is not None
        one_hot = [1.0 if self.maneuver == m else 0.0 for m in MANEUVERS]
        seg = self.segment
        if seg is not None:
            remaining = (abs(seg[1]) - math.copysign(1.0, seg[1]) * self.seg_progress) / math.pi
            seg_feats = [
                1.0 if seg[0] == PITCH else -1.0,
                math.copysign(1.0, seg[1]),
                clip(remaining, -1.0, 2.0),
            ]
        else:
            seg_feats = [0.0, 0.0, 0.0]
        offset = geo.position - self.origin
        return np.array(
            [
                *one_hot,
                *seg_feats,
                self.progress_fraction(),
                *(self.frame.T @ geo.velocity_dir),
                *(self.frame.T @ geo.lift_dir),
                self.lift_error(geo) / math.pi,
                clip(float(offset @ self.frame[:, 1]) / 500.0, -3.0, 3.0),
                clip(-float(offset @ self.frame[:, 2]) / 1000.0, -3.0, 3.0),
            ]
        )

    def reward(self, ins: Instruments, action: Vec, previous_action: Vec) -> RewardTerms:
        geo = self._geo
        assert geo is not None
        terms: RewardTerms = {
            "progress": self.w_progress * self._last_delta / self.total_angle,
            "plane": 0.0,
            "lift": 0.0,
            "track": 0.0,
            "exit": 0.0,
            "time": -self.w_time,
            "smoothness": -self.w_smooth * action_smoothness(action, previous_action),
        }
        seg = self.segment
        if seg is not None and seg[0] == PITCH:
            lateral = abs(float(geo.velocity_dir @ self.frame[:, 1]))
            terms["plane"] = -self.w_plane * min(lateral / math.sin(20 * DEG), 1.0)
            terms["lift"] = -self.w_lift * abs(self.lift_error(geo)) / math.pi
        elif seg is not None:
            cos_dev = clip(float(geo.velocity_dir @ self._seg_start_velocity), -1.0, 1.0)
            terms["track"] = -self.w_track * min(math.acos(cos_dev) / (20 * DEG), 1.0)
        else:
            terms["exit"] = -self.w_exit * (
                abs(angle_error(ins.bank)) / math.pi + min(abs(ins.gamma) / (30 * DEG), 1.0)
            )
        terms["success"] = self.completion_bonus if self._completed else 0.0
        return terms

    def done(self) -> bool:
        return self._completed

    def success(self, ins: Instruments) -> bool:
        return self._completed

    def metrics(self) -> dict[str, float]:
        heading, lateral, dh = self._exit_state if self._completed else (math.pi, 0.0, 0.0)
        return {
            "progress": self.progress_fraction() if self.segment else 1.0,
            "completion_time": self._finished_time if self._completed else self.episode_time,
            "exit_heading_error_deg": abs(math.degrees(heading)),
            "exit_lateral_offset": abs(lateral),
            "exit_altitude_change": dh,
        }

    # ------------------------------------------------------------------
    def autopilot_targets(self, ins: Instruments) -> AutopilotTargets:
        return AutopilotTargets(altitude=ins.altitude, bank=0.0)

    def baseline_command(
        self, ins: Instruments, autopilot: Autopilot, dt: float
    ) -> HighLevelCommand:
        """Pilote scripté : suit les segments, puis remet l'avion en palier."""
        geo = self._geo
        assert geo is not None
        seg = self.segment
        if seg is None:
            if not self._ap_engaged:
                autopilot.gamma_pid.reset(0.0)
                autopilot.bank_pid.reset(0.0)
                autopilot.speed_pid.reset(1.0)
                self._ap_engaged = True
            autopilot.targets = AutopilotTargets(altitude=ins.altitude, bank=0.0, throttle=1.0)
            return autopilot.high_level(ins, dt)
        kind, angle = seg
        sign = math.copysign(1.0, angle)
        remaining = abs(angle) - sign * self.seg_progress
        if kind == PITCH:
            roll_rate = clip(3.0 * self.lift_error(geo), -90 * DEG, 90 * DEG)
            return HighLevelCommand(nz=self.pull_load_factor, roll_rate=roll_rate, throttle=1.0)
        # tonneau : taux de roulis fixe, ralenti en fin de segment ; le facteur de charge
        # compense la composante de la pesanteur portée par la portance (trajectoire droite)
        rate = sign * clip(2.5 * remaining + 20 * DEG, 0.0, 150 * DEG)
        nz = -float(geo.lift_dir[2]) * 1.0
        return HighLevelCommand(nz=nz, roll_rate=rate, throttle=1.0)

    def describe(self) -> dict[str, float]:
        return {
            "maneuver_index": float(MANEUVERS.index(self.maneuver)),
            "roll_sign": next((math.copysign(1.0, a) for k, a in self.segments if k == ROLL), 0.0),
        }
