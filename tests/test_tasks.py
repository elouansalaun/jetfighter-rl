"""Phase 8 : tâches d'apprentissage (critères, suivi, géométrie, politiques de référence)."""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env
from jetsim.aircraft.instruments import wind_axes
from jetsim.core.frames import dcm_from_euler

from jetfighter_rl.envs.baselines import AutopilotPolicy, evaluate
from jetfighter_rl.envs.jet_env import N_OWN_SHIP, EnvConfig, JetEnv
from jetfighter_rl.envs.tasks import (
    MANEUVERS,
    TASKS,
    FlightGeometry,
    InitialCondition,
    make_task,
    sustained_turn_reference,
)
from jetfighter_rl.envs.tasks.aerobatics import segments_for

DEG = math.pi / 180


def flown(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
    return info


@pytest.mark.parametrize("task", ["sustained_turn", "waypoints", "aerobatics"])
@pytest.mark.parametrize("model", ["3dof", "6dof"])
def test_check_env_new_tasks(task, model):
    env = JetEnv(EnvConfig(task=task, model=model))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (N_OWN_SHIP + 3 + env.task.n_features,)


def test_registry_and_task_kwargs():
    assert set(TASKS) == {"level", "heading_altitude", "sustained_turn", "waypoints", "aerobatics"}
    task = make_task("aerobatics", maneuvers="loop")
    assert task.maneuvers == ("loop",)
    with pytest.raises(ValueError):
        make_task("aerobatics", maneuvers=("barrel",))


# --------------------------------------------------------------------------
# Géométrie
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["3dof", "6dof"])
def test_wind_axes_match_flight_path_angles(model):
    env = JetEnv(EnvConfig(task="level", model=model))
    ic = InitialCondition(altitude=5000, airspeed=230, heading=0.7, gamma=12 * DEG, bank=40 * DEG)
    env.reset(seed=0, options={"initial_condition": ic})
    c_nw = wind_axes(env.model, env.x)
    np.testing.assert_allclose(c_nw, dcm_from_euler(40 * DEG, 12 * DEG, 0.7), atol=1e-6)
    geo = env.geometry()
    np.testing.assert_allclose(geo.position, [0.0, 0.0, -5000.0], atol=1e-6)
    assert geo.lift_dir[2] < 0  # portance vers le haut


# --------------------------------------------------------------------------
# 8.1 et 8.2 : critères de la roadmap
# --------------------------------------------------------------------------
def test_level_recovery_time_criterion():
    env = JetEnv(EnvConfig(task="level"))
    info = flown(env, AutopilotPolicy(env), seed=4)
    assert info["is_success"] and 0 < info["metrics"]["recovery_time"] < 10
    # sans action corrective depuis une forte inclinaison : jamais rétabli
    env = JetEnv(EnvConfig(task="level", episode_time=5.0))
    env.reset(seed=0, options={"initial_condition": InitialCondition(5000, 220, bank=60 * DEG)})
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0], np.float32))
        done = terminated or truncated
    assert not info["is_success"] and info["metrics"]["recovery_time"] == env.task.episode_time


def test_heading_altitude_overshoot_is_measured():
    env = JetEnv(EnvConfig(task="heading_altitude"))
    env.reset(seed=1)
    task, ins = env.task, env.instruments
    geo = env.geometry()
    climb = task.target_altitude > ins.altitude
    past = replace(ins, altitude=task.target_altitude + (120.0 if climb else -120.0))
    task.update(past, geo, 1.0)
    assert task.metrics()["altitude_overshoot"] == pytest.approx(120.0)
    on_target = replace(
        ins, altitude=task.target_altitude, course=task.target_heading, tas=task.target_speed
    )
    task.update(on_target, geo, 2.0)
    assert task.success(on_target)
    limit = max(50.0, 0.1 * abs(task._initial_errors[0]))
    assert task.episode_success(on_target) == (120.0 <= limit)


# --------------------------------------------------------------------------
# 8.3 : virage soutenu
# --------------------------------------------------------------------------
def test_sustained_turn_reference_is_the_optimum():
    from jetsim.aircraft.dynamics_3dof import PointMassAircraft
    from jetsim.aircraft.params import load_aircraft
    from jetsim.aircraft.performance import sustained_turn

    ref = sustained_turn_reference(5000.0)
    assert 200 < ref.speed < 300 and 12 * DEG < ref.turn_rate < 18 * DEG
    ac = PointMassAircraft(load_aircraft("f16"))
    for v in (ref.speed - 30, ref.speed + 30):
        assert sustained_turn(ac, 5000.0, v).turn_rate < ref.turn_rate
    assert sustained_turn_reference(5010.0) is ref  # cache par tranche de 250 m
    assert sustained_turn_reference(9000.0).turn_rate < ref.turn_rate  # air plus rare


def test_sustained_turn_baseline_meets_criterion():
    env = JetEnv(EnvConfig(task="sustained_turn"))
    ev = evaluate(env, AutopilotPolicy(env), episodes=2, seed=100)
    assert ev.success_rate == 1.0 and ev.crash_rate == 0.0
    assert ev.metrics["turn_rate_ratio"] == pytest.approx(1.0, abs=0.08)


# --------------------------------------------------------------------------
# 8.4 : points de passage
# --------------------------------------------------------------------------
def test_waypoint_generation_and_capture():
    env = JetEnv(EnvConfig(task="waypoints"))
    env.reset(seed=3)
    task = env.task
    ins, geo = env.instruments, env.geometry()
    pts = [(ins.north, ins.east)] + [(n, e) for n, e, _ in task.waypoints]
    legs = [math.dist(a, b) for a, b in itertools.pairwise(pts)]
    assert len(task.waypoints) == 4 and all(4999 < d < 9001 for d in legs)
    a = np.zeros(3)
    # se rapprocher du point visé rapporte, le franchir donne le bonus
    wp = task.waypoints[0]
    closer = replace(
        ins, north=ins.north + 0.1 * (wp[0] - ins.north), east=ins.east + 0.1 * (wp[1] - ins.east)
    )
    task.update(closer, geo, 0.1)
    assert task.reward(closer, a, a)["progress"] > 0
    for k, (n, e, h) in enumerate(task.waypoints):
        at_wp = replace(ins, north=n + 100.0, east=e, altitude=h - 50.0)
        task.update(at_wp, geo, 1.0 + k)
        terms = task.reward(at_wp, a, a)
        assert task.index == k + 1 and terms["waypoint"] >= task.waypoint_bonus
    assert task.done() and terms["waypoint"] == task.waypoint_bonus + task.final_bonus
    assert task.metrics()["waypoints_reached"] == 4


def test_waypoints_baseline_completes_the_course():
    env = JetEnv(EnvConfig(task="waypoints"))
    info = flown(env, AutopilotPolicy(env), seed=101)
    assert info["is_success"] and info["metrics"]["waypoints_reached"] == 4
    assert info["t"] < env.episode_time and not info["violation"]


# --------------------------------------------------------------------------
# 8.5 : voltige
# --------------------------------------------------------------------------
def test_maneuver_definitions():
    assert segments_for("loop", 1.0) == [("pitch", 2 * math.pi)]
    assert segments_for("split_s", -1.0) == [("roll", -math.pi), ("pitch", -math.pi)]
    assert sum(abs(a) for _, a in segments_for("immelmann", 1.0)) == pytest.approx(2 * math.pi)


def _geometry(gamma: float, bank: float, heading: float = 0.0) -> FlightGeometry:
    return FlightGeometry(np.zeros(3), dcm_from_euler(bank, gamma, heading))


def test_loop_progress_is_tracked_through_the_vertical():
    env = JetEnv(EnvConfig(task="aerobatics", task_kwargs={"maneuvers": "loop"}))
    env.reset(seed=0)
    task, ins = env.task, env.instruments
    heading = ins.course
    # vecteur vitesse qui tourne dans le plan vertical : montée, dos, descente, palier
    angles = np.linspace(0, 2 * math.pi, 73)
    for k, th in enumerate(angles[1:], start=1):
        # au-delà de 90° la route s'inverse et l'avion est sur le dos (repère continu)
        c_nw = dcm_from_euler(0.0, 0.0, heading) @ np.array(
            [[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]
        )
        task.update(ins, FlightGeometry(np.zeros(3), c_nw), 0.1 * k)
        if k == 36:
            assert task.progress_fraction() == pytest.approx(0.5, abs=0.02)
            assert abs(task.lift_error(FlightGeometry(np.zeros(3), c_nw))) < 1e-6
    assert task.segment is None  # boucle complète
    assert task.done()  # sortie ailes à plat (instruments de départ)


def test_roll_progress_about_velocity():
    env = JetEnv(EnvConfig(task="aerobatics", task_kwargs={"maneuvers": "roll"}))
    env.reset(seed=0)
    task, ins = env.task, env.instruments
    sign = task.segments[0][1] / abs(task.segments[0][1])
    for k, mu in enumerate(np.linspace(0, 2 * math.pi, 37)[1:], start=1):
        task.update(ins, _geometry(0.0, sign * mu, ins.course), 0.1 * k)
    assert task.segment is None and task.done()


@pytest.mark.parametrize("maneuver", MANEUVERS)
def test_scripted_pilot_flies_every_maneuver(maneuver):
    env = JetEnv(EnvConfig(task="aerobatics", task_kwargs={"maneuvers": maneuver}))
    info = flown(env, AutopilotPolicy(env), seed=7)
    m = info["metrics"]
    assert info["is_success"] and not info["violation"]
    assert m["exit_heading_error_deg"] < 5 and m["exit_lateral_offset"] < 100
    expected_climb = {"loop": None, "roll": None, "immelmann": 1, "split_s": -1}[maneuver]
    if expected_climb is not None:
        assert expected_climb * m["exit_altitude_change"] > 800


def test_scripted_loop_in_6dof():
    env = JetEnv(EnvConfig(task="aerobatics", model="6dof", task_kwargs={"maneuvers": "loop"}))
    info = flown(env, AutopilotPolicy(env), seed=3)
    assert info["is_success"] and info["metrics"]["exit_heading_error_deg"] < 5


def test_coning_out_of_plane_does_not_count_as_a_loop():
    """Reward hacking observé : virer de 90° puis tourner autour de l'axe latéral faisait
    « tourner » l'angle projeté dans le plan de la boucle sans faire de looping."""
    env = JetEnv(EnvConfig(task="aerobatics", task_kwargs={"maneuvers": "loop"}))
    env.reset(seed=0)
    task, ins = env.task, env.instruments
    e1, e2, e3 = task.frame.T
    for k in range(1, 200):
        phi = 0.3 * k  # petit cône autour de e2 (vitesse presque latérale)
        v = 0.95 * e2 + 0.31 * (math.cos(phi) * e1 + math.sin(phi) * e3)
        v /= np.linalg.norm(v)
        w = np.cross(e3, v)
        w /= np.linalg.norm(w)
        c_nw = np.column_stack([v, w, np.cross(v, w)])
        task.update(ins, FlightGeometry(np.zeros(3), c_nw), 0.1 * k)
    assert task.progress_fraction() < 0.05 and not task.done()


def test_loop_exit_requires_the_initial_heading():
    env = JetEnv(EnvConfig(task="aerobatics", task_kwargs={"maneuvers": "loop"}))
    env.reset(seed=0)
    task, ins = env.task, env.instruments
    task.index = len(task.segments)  # figure parcourue
    wrong = replace(ins, course=ins.course + math.radians(90))
    task.update(wrong, env.geometry(), 1.0)
    assert not task.done()
    task.update(ins, env.geometry(), 1.1)
    assert task.done()
