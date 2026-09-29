"""Environnement Gymnasium ``JetEnv`` : le F-16 (3-DOF ou 6-DOF) piloté par un agent RL.

Deux **modes d'action** (décision du 27/09/2026 : hiérarchique d'abord, bas niveau ensuite),
toujours dans ``Box(−1, 1)`` :

* ``"hierarchical"`` (défaut) — [n_z, taux de roulis, manette], exécutés par la boucle interne
  (commandes de vol électriques en 6-DOF) ; même espace d'action pour les deux modèles.
  0 sur l'axe de tangage = n_z qui **maintient la pente** (cos γ / cos μ : 1 g en palier,
  2 g incliné à 60°) ; +1 = +9 g ; −1 = −3 g. Sans cette compensation, la politique
  devait produire elle-même 1/cos μ au centième près pour ne pas monter ou descendre en
  virage : PPO n'apprenait pas la tâche cap/altitude (``nz_neutral="one_g"`` pour
  l'ancien comportement) ;
* ``"low_level"`` — 6-DOF : [manette, δe, δa, δr] (gouvernes directes, sans limiteur) ;
  3-DOF : [manette, α commandée, taux de roulis].

Fréquences : physique 100 Hz, boucle interne et surveillance de l'enveloppe 50 Hz,
décision de l'agent 10 Hz (réglables dans ``EnvConfig``).

Observations (``Box(−10, 10)``, float32) : état propre normalisé (vitesse, altitude, pente,
inclinaison et gîte en sin/cos, α, β, p, q, r, n_z, Ps, puissance), dernière action, puis
les grandeurs propres à la tâche (erreurs relatives à la consigne). Elles sont calculées à
partir des **mesures** (capteurs éventuellement bruités) ; la récompense et les fins
d'épisode utilisent les valeurs **vraies**.

Fin d'épisode : ``terminated`` si l'enveloppe est violée (crash, surcharge, décrochage…,
pénalité ``task.crash_penalty`` plus le coût maximal des pas restants, dans la limite de
l'horizon 1/(1−γ) de l'agent) ou si la tâche est
accomplie (``task.done()`` : dernier point de passage franchi, figure de voltige
terminée) ; ``truncated`` à la durée maximale de la tâche. En fin d'épisode, ``info``
contient ``is_success`` (critère de la roadmap), ``metrics`` (temps de rétablissement,
dépassements…) et ``episode_terms`` (récompense cumulée par terme).

Usage ::

    env = JetEnv(EnvConfig(task="heading_altitude", model="3dof"))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from jetfighter.aircraft import dynamics_3dof as d3
from jetfighter.aircraft import dynamics_6dof as d6
from jetfighter.aircraft.envelope import EnvelopeMonitor
from jetfighter.aircraft.instruments import (
    Instruments,
    position_ned,
    read_instruments,
    wind_axes,
)
from jetfighter.aircraft.params import CONFIG_DIR, load_aircraft, load_envelope
from jetfighter.aircraft.sensors import SensorSuite
from jetfighter.control.fbw import HighLevelCommand, make_inner_loop
from jetfighter.core.frames import dcm_from_euler, dcm_wind_to_body, quat_from_dcm
from jetfighter.envs.rewards import accumulate, total
from jetfighter.envs.tasks import FlightGeometry, InitialCondition, Task, make_task
from jetfighter.viz.recorder import FlightRecorder, FlightRecording

Vec = npt.NDArray[np.float64]
DEG = math.pi / 180
OBS_LIMIT = 10.0
N_OWN_SHIP = 19  # grandeurs propres à l'avion dans l'observation
REFERENCE_DT = 0.1  # [s] cadence de référence des récompenses par pas


@dataclass
class EnvConfig:
    task: str = "level"
    model: str = "3dof"  # "3dof" | "6dof"
    action_mode: str = "hierarchical"  # "hierarchical" | "low_level"
    agent_dt: float = 0.1  # période de décision de l'agent [s]
    control_dt: float = 0.02  # période de la boucle interne [s]
    physics_dt: float = 0.01  # pas d'intégration [s]
    episode_time: float | None = None  # None : durée de la tâche
    sensors: str | None = None  # None (parfaits), "realistic" ou chemin d'un YAML
    xcg: float | None = None  # centrage 6-DOF (None : valeur du YAML)
    max_roll_rate: float = 180 * DEG  # taux de roulis max demandable (mode hiérarchique)
    action_exponent: float = 1.0  # courbe « expo » de n_z et du roulis (1 : linéaire)
    attitude_features: str = "gravity"  # "gravity" (continu) | "euler" (singulier à la verticale)
    nz_neutral: str = "compensated"  # n_z à action nulle : "compensated" (cos γ/cos μ) | "one_g"
    record: bool = False  # enregistre les vols (cf. ``JetEnv.recording``)
    discount: float = 0.99  # γ de l'agent : fixe l'horizon couvert par la pénalité de crash
    task_kwargs: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Correspondance action normalisée <-> commande physique
# --------------------------------------------------------------------------
NZ_MAX, NZ_MIN = 9.0, -3.0


# --------------------------------------------------------------------------
# Bas niveau (étape 8.6) : plus de commandes de vol pour protéger la structure
# --------------------------------------------------------------------------
LIMIT_WEIGHTS = {"overload": 0.5, "sideslip": 0.2}
NZ_WARN_HIGH, NZ_WARN_LOW = 8.0, -2.0  # début de la pénalité (limites structurales : +10/−4 g)


def limit_terms(ins: Instruments) -> dict[str, float]:
    """Pénalités propres au mode bas niveau : approche des limites structurales et
    dérapage (plus aucun limiteur ni amortisseur de lacet ne s'en charge)."""
    over = max(ins.nz - NZ_WARN_HIGH, NZ_WARN_LOW - ins.nz, 0.0)
    return {
        "overload": -LIMIT_WEIGHTS["overload"] * min(over / 2.0, 1.0),
        "sideslip": -LIMIT_WEIGHTS["sideslip"] * min((ins.beta / (5 * DEG)) ** 2, 1.0),
    }


def _expo(x: float, exponent: float) -> float:
    return math.copysign(abs(x) ** exponent, x)


def gravity_compensation(gamma: float, bank: float) -> float:
    """Facteur de charge qui maintient la pente en virage : cos γ / cos μ.

    |cos μ| est borné à 0.25 (compensation d'au plus 4 g) ; sur le dos (|μ| > 90°), la
    compensation est négative (on « pousse » pour tenir la trajectoire). Le résultat est
    borné à [−2, 4] g, strictement dans la plage des commandes.
    """
    c = math.cos(bank)
    c = math.copysign(max(abs(c), 0.25), c)
    return min(max(math.cos(gamma) / c, -2.0), 4.0)


def action_to_command(
    a: Vec, max_roll_rate: float, exponent: float = 1.0, nz_bias: float = 1.0
) -> HighLevelCommand:
    """Action hiérarchique [−1, 1]³ -> (n_z, taux de roulis, manette).

    ``nz_bias`` est le facteur de charge commandé quand a₀ = 0 : 1 g (mode ``"one_g"``) ou
    la compensation de pesanteur cos γ / cos μ (mode ``"compensated"``, par défaut dans
    l'environnement : action nulle = trajectoire maintenue, même incliné). a₀ = ±1 donne
    toujours +9 / −3 g.

    Option ``exponent`` > 1 : courbe « expo » (|a|^exposant, comme sur les radiocommandes)
    sur le facteur de charge et le roulis, plus fine autour du neutre. Essayée avec 3 : pas
    de gain mesurable pour PPO, et l'imitation du pilote automatique devient bien moins
    précise (la réciproque, en racine cubique, est très raide près de zéro) ; linéaire par
    défaut.
    """
    a0 = _expo(float(a[0]), exponent)
    nz = nz_bias + a0 * ((NZ_MAX - nz_bias) if a0 >= 0 else (nz_bias - NZ_MIN))
    return HighLevelCommand(nz=nz, roll_rate=_expo(float(a[1]), exponent) * max_roll_rate,
                            throttle=0.5 * (float(a[2]) + 1.0))  # fmt: skip


def command_to_action(
    cmd: HighLevelCommand, max_roll_rate: float, exponent: float = 1.0, nz_bias: float = 1.0
) -> Vec:
    """Inverse de ``action_to_command`` (saturée dans [−1, 1])."""
    dn = cmd.nz - nz_bias
    a0 = dn / (NZ_MAX - nz_bias) if dn >= 0 else dn / (nz_bias - NZ_MIN)
    a1 = cmd.roll_rate / max_roll_rate
    inv = 1.0 / exponent
    a = np.array([_expo(min(max(a0, -1.0), 1.0), inv), _expo(min(max(a1, -1.0), 1.0), inv),
                  2.0 * cmd.throttle - 1.0])  # fmt: skip
    return np.clip(a, -1.0, 1.0)


class JetEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Environnement d'apprentissage (voir le module)."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}  # noqa: RUF012  (API Gymnasium)

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        **overrides: Any,
    ) -> None:
        super().__init__()
        cfg = replace(config or EnvConfig(), **overrides)
        if cfg.model not in ("3dof", "6dof"):
            raise ValueError("model doit valoir '3dof' ou '6dof'.")
        if cfg.attitude_features not in ("gravity", "euler"):
            raise ValueError("attitude_features doit valoir 'gravity' ou 'euler'.")
        if cfg.nz_neutral not in ("compensated", "one_g"):
            raise ValueError("nz_neutral doit valoir 'compensated' ou 'one_g'.")
        if cfg.action_mode not in ("hierarchical", "low_level"):
            raise ValueError("action_mode doit valoir 'hierarchical' ou 'low_level'.")
        self.cfg = cfg
        self.render_mode = render_mode
        self.six_dof = cfg.model == "6dof"
        self.hierarchical = cfg.action_mode == "hierarchical"

        params = load_aircraft("f16")
        self.model: d3.PointMassAircraft | d6.F16SixDof = (
            d6.F16SixDof(params, xcg=cfg.xcg) if self.six_dof else d3.PointMassAircraft(params)
        )
        self.params = params
        self.inner = make_inner_loop(self.model)
        self.monitor = EnvelopeMonitor(load_envelope("f16", six_dof=self.six_dof))
        self.task: Task = make_task(cfg.task, **cfg.task_kwargs)
        self.sensors = self._make_sensors(cfg.sensors)

        self.n_control = round(cfg.agent_dt / cfg.control_dt)
        self.n_physics = round(cfg.control_dt / cfg.physics_dt)
        if not (math.isclose(self.n_control * cfg.control_dt, cfg.agent_dt)
                and math.isclose(self.n_physics * cfg.physics_dt, cfg.control_dt)):  # fmt: skip
            raise ValueError("agent_dt, control_dt et physics_dt doivent être multiples.")
        self.episode_time = cfg.episode_time or self.task.episode_time

        n_act = 3 if (self.hierarchical or not self.six_dof) else 4
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        # bas niveau 6-DOF : position réelle des gouvernes (δe, δa, δr) dans l'observation
        self.n_surfaces = 3 if (self.six_dof and not self.hierarchical) else 0
        # coûts par pas mis à l'échelle de la cadence : les rendements restent comparables
        # entre un agent à 10 Hz (hiérarchique) et un agent à 50 Hz (gouvernes directes)
        self.reward_scale = cfg.agent_dt / REFERENCE_DT
        self.limit_cost = LIMIT_WEIGHTS["overload"] + LIMIT_WEIGHTS["sideslip"]
        if self.hierarchical:
            self.limit_cost = 0.0
        n_obs = N_OWN_SHIP + n_act + self.n_surfaces + self.task.n_features
        self.n_act, self.n_obs = n_act, n_obs
        self.observation_space = spaces.Box(-OBS_LIMIT, OBS_LIMIT, shape=(n_obs,),
                                            dtype=np.float32)  # fmt: skip

        self.x: Vec = np.zeros(0)
        self.t = 0.0
        self.instruments: Instruments | None = None
        self.previous_action: Vec = np.zeros(n_act)
        self.episode_terms: dict[str, float] = {}
        self.last_violation: str = ""
        self._recorder: FlightRecorder | None = None
        self._last_recording: FlightRecording | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @staticmethod
    def _make_sensors(spec: str | None) -> SensorSuite:
        if spec is None:
            return SensorSuite.perfect()
        path = CONFIG_DIR.parent / "sensors" / f"{spec}.yaml" if "/" not in spec else spec
        return SensorSuite.from_yaml(path)

    def _initial_state(self, ic: InitialCondition) -> tuple[Vec, Vec]:
        """État initial à partir de conditions initiales.

        On part de l'équilibre à la pente γ demandée ; s'il n'existe pas (piqué trop raide :
        l'avion accélère même au ralenti), de l'équilibre en palier, auquel on impose ensuite
        la pente, l'inclinaison et le taux de roulis. L'état de départ n'est alors pas un
        équilibre, ce qui est voulu pour les tâches de rattrapage.
        Lève ``TrimError`` si même le palier est impossible (vitesse hors enveloppe).
        """
        if isinstance(self.model, d6.F16SixDof):
            try:
                x, u = self.model.trim(ic.altitude, ic.airspeed, gamma=ic.gamma)
            except d6.TrimError:
                x, u = self.model.trim(ic.altitude, ic.airspeed)
            alpha = math.atan2(x[d6.W], x[d6.U])
            c_bw = dcm_wind_to_body(alpha, 0.0)
            c_nw = dcm_from_euler(ic.bank, ic.gamma, ic.heading)
            x[d6.QUAT] = quat_from_dcm(c_nw @ c_bw.T)
            x[d6.RATES] = c_bw @ np.array([ic.roll_rate, 0.0, 0.0])
            return x, u
        try:
            alpha, power = self.model.trim(ic.altitude, ic.airspeed, ic.gamma)
        except d3.TrimError:
            alpha, power = self.model.trim(ic.altitude, ic.airspeed)
        x = self.model.make_state(altitude=ic.altitude, airspeed=ic.airspeed, gamma=ic.gamma,
                                  heading=ic.heading, bank=ic.bank, alpha=alpha, power=power,
                                  roll_rate=ic.roll_rate)  # fmt: skip
        return x, np.array([power, alpha, 0.0])

    # ------------------------------------------------------------------
    # API Gymnasium
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        for _ in range(50):
            ic = options.get("initial_condition") or self.task.sample_initial(rng, self.six_dof)
            try:
                self.x, u0 = self._initial_state(ic)
                break
            except (d3.TrimError, d6.TrimError):
                if "initial_condition" in options:
                    raise
        else:  # pragma: no cover - pratiquement impossible avec les plages des tâches
            raise RuntimeError("Impossible de trouver un état initial équilibré.")

        self.t = 0.0
        self.inner.reset(self.x)
        self.monitor.reset()
        self.sensors.reset(np.random.default_rng(rng.integers(2**32)))
        self.instruments = read_instruments(self.model, self.x)
        self.task.reset(self.instruments, self.geometry(), rng, self.six_dof)
        self.previous_action = self._neutral_action(u0)
        self.episode_terms = {}
        self.last_violation = ""
        if self.cfg.record:
            self._recorder = FlightRecorder(
                self.model, physics_dt=self.cfg.physics_dt, substeps=self.n_physics,
                metadata={"task": self.task.name, **self.task.describe()},
            )  # fmt: skip
        info = {"initial_condition": ic.__dict__, "targets": self.task.describe()}
        return self._observation(), info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.instruments is None:
            raise RuntimeError("Appeler reset() avant step().")
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        u_direct = np.zeros(0) if self.hierarchical else self._low_level_controls(a)

        violation = None
        ins = self.instruments
        u: Vec = np.zeros(0)
        for _ in range(self.n_control):
            if self.hierarchical:
                command = action_to_command(a, self.cfg.max_roll_rate, self.cfg.action_exponent,
                                            self.nz_bias(ins))  # fmt: skip
                u_next = self.inner(ins, command, self.cfg.control_dt)
            else:
                u_next = u_direct
            assert u_next is not None
            u = u_next
            if self._recorder is not None:
                self._recorder.record(self.t, self.x, u)
            for _ in range(self.n_physics):
                self.x = self.model.step(self.x, u, self.cfg.physics_dt)
            self.t = round(self.t + self.cfg.control_dt, 9)
            ins = read_instruments(self.model, self.x)
            violation = self.monitor.check(ins, self.cfg.control_dt, state=self.x)
            if violation is not None:
                break
        self.instruments = ins
        if violation is None:
            self.task.update(ins, self.geometry(), self.t)

        terms = self.task.reward(ins, a, self.previous_action)
        if not self.hierarchical:
            terms |= limit_terms(ins)
        terms = {k: (v if k in self.task.event_terms else v * self.reward_scale)
                 for k, v in terms.items()}  # fmt: skip
        crashed = violation is not None
        terminated = crashed or self.task.done()
        if crashed:
            # pénalité fixe + coût maximal des pas restants (limités à l'horizon 1/(1−γ) de
            # l'agent) : s'écraser est toujours pire que de continuer à voler, même mal
            # (sinon, loin de la consigne, l'agent apprendrait à abréger l'épisode)
            remaining = max(self.episode_time - self.t, 0.0) / self.cfg.agent_dt
            horizon = 1.0 / max(1.0 - self.cfg.discount, 1e-3)
            step_cost = (self.task.max_step_cost + self.limit_cost) * self.reward_scale
            terms["crash"] = self.task.crash_penalty - step_cost * min(remaining, horizon)
            self.last_violation = self.monitor.message
        truncated = not terminated and self.t >= self.episode_time - 1e-9
        reward = total(terms)
        accumulate(self.episode_terms, terms)
        self.previous_action = a

        info: dict[str, Any] = {"reward_terms": terms, "t": self.t}
        if terminated or truncated:
            info["episode_terms"] = dict(self.episode_terms)
            info["is_success"] = bool(not crashed and self.task.episode_success(ins))
            info["violation"] = self.last_violation
            info["metrics"] = self.task.metrics()
            if self._recorder is not None:
                if self.last_violation:
                    self._recorder.event(self.t, self.last_violation)
                self._recorder.record(self.t, self.x, u)
                self._last_recording = self._recorder.finish()
        obs = self._observation() if not crashed or np.all(np.isfinite(self.x)) else \
            np.zeros(self.n_obs, dtype=np.float32)  # fmt: skip
        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self.instruments is None:
            return None
        i = self.instruments
        return (f"t={self.t:6.1f}s h={i.altitude:6.0f}m V={i.tas:5.0f}m/s "
                f"μ={math.degrees(i.bank):+6.1f}° γ={math.degrees(i.gamma):+5.1f}° "
                f"n={i.nz:+5.2f}g α={math.degrees(i.alpha):+5.1f}°")  # fmt: skip

    # ------------------------------------------------------------------
    # Aides
    # ------------------------------------------------------------------
    def _low_level_controls(self, a: Vec) -> Vec:
        throttle = 0.5 * (a[0] + 1.0)
        if isinstance(self.model, d6.F16SixDof):
            cs = self.params.control_surfaces
            return np.array([throttle, a[1] * cs["elevator"].max, a[2] * cs["aileron"].max,
                             a[3] * cs["rudder"].max])  # fmt: skip
        lim = self.params.limits
        alpha = a[1] * (lim.alpha_max if a[1] >= 0 else -lim.alpha_min)
        return np.array([throttle, alpha, a[2] * lim.roll_rate_max])

    @property
    def has_reference(self) -> bool:
        """Une politique de référence existe (hiérarchique, ou bas niveau 6-DOF)."""
        return self.hierarchical or self.six_dof

    def controls_to_action(self, u: Vec) -> Vec:
        """Inverse de ``_low_level_controls`` (6-DOF) : commande du modèle -> action."""
        cs = self.params.control_surfaces
        a = np.array([2.0 * u[0] - 1.0, u[1] / cs["elevator"].max, u[2] / cs["aileron"].max,
                      u[3] / cs["rudder"].max])  # fmt: skip
        return np.clip(a, -1.0, 1.0)

    def _neutral_action(self, u0: Vec) -> Vec:
        """Action qui correspond à l'équilibre initial (évite une pénalité de lissage
        artificielle au premier pas)."""
        if self.hierarchical:
            thr = float(u0[0])
            if isinstance(self.model, d6.F16SixDof):
                thr = self.model.engine.throttle_for_power(float(self.x[d6.POWER]))
            assert self.instruments is not None
            bias = self.nz_bias(self.instruments)
            return command_to_action(HighLevelCommand(bias, 0.0, min(max(thr, 0.0), 1.0)),
                                     self.cfg.max_roll_rate, self.cfg.action_exponent,
                                     bias)  # fmt: skip
        if self.six_dof:
            return self.controls_to_action(u0)
        return np.zeros(self.n_act)

    def nz_bias(self, ins: Instruments) -> float:
        """Facteur de charge commandé à action nulle (voir ``action_to_command``)."""
        if self.cfg.nz_neutral == "compensated":
            return gravity_compensation(ins.gamma, ins.bank)
        return 1.0

    def _observation(self) -> npt.NDArray[np.float32]:
        assert self.instruments is not None
        m = self.sensors.measure(self.instruments)
        if self.cfg.attitude_features == "gravity":
            # direction de la pesanteur en axes vent puis corps : continue à toute attitude
            # (les angles d'Euler basculent de 180° au passage de la verticale)
            cg, cp = math.cos(m.gamma), math.cos(m.pitch)
            attitude = [-math.sin(m.gamma), math.sin(m.bank) * cg, math.cos(m.bank) * cg,
                        -math.sin(m.pitch), math.sin(m.roll) * cp, math.cos(m.roll) * cp,
                        ]  # fmt: skip
            last = []
        else:
            attitude = [math.sin(m.gamma), math.sin(m.bank), math.cos(m.bank),
                        math.sin(m.roll), math.cos(m.roll), math.sin(m.pitch)]  # fmt: skip
            last = [math.cos(m.pitch)]
        own = np.array([
            (m.tas - 200.0) / 100.0,
            (m.altitude - 6000.0) / 4000.0,
            (m.mach - 0.7) / 0.5,
            m.vertical_speed / 100.0,
            *attitude,
            m.alpha / 0.4,
            m.beta / 0.2,
            m.p / 3.0, m.q / 1.0, m.r / 1.0,
            (m.nz - 1.0) / 4.0,
            m.specific_excess_power / 100.0,
            2.0 * m.power - 1.0,
            *last,
        ])  # fmt: skip
        if len(own) < N_OWN_SHIP:  # même taille dans les deux modes
            own = np.concatenate([own, np.zeros(N_OWN_SHIP - len(own))])
        parts = [own, self.previous_action]
        if self.n_surfaces:
            cs = self.params.control_surfaces
            parts.append(np.array([self.x[d6.DE] / cs["elevator"].max,
                                   self.x[d6.DA] / cs["aileron"].max,
                                   self.x[d6.DR] / cs["rudder"].max]))  # fmt: skip
        obs = np.concatenate([*parts, self.task.features(m)])
        return np.clip(np.nan_to_num(obs), -OBS_LIMIT, OBS_LIMIT).astype(np.float32)

    def geometry(self) -> FlightGeometry:
        """Position et repère vent vrais (pour les tâches de navigation et de voltige)."""
        return FlightGeometry(position_ned(self.model, self.x), wind_axes(self.model, self.x))

    def recording(self) -> FlightRecording | None:
        """Enregistrement du dernier épisode terminé (si ``record=True``)."""
        return self._last_recording
