import math

from orbitforge.core.state import TimeWindow
from orbitforge.core.vector import Vec3
from orbitforge.attitude.pointing import quaternion_between
from orbitforge.attitude.slew import bang_bang_slew_time
from orbitforge.mission.imaging_windows import (
    ChainLimits,
    ImagingTarget,
    REASON_ACCESS,
    REASON_EARTH,
    REASON_PREDECESSOR,
    REASON_SLEW_SUN,
    REASON_SUN,
    REASON_TRANSITION_TIME,
    chain_imaging_windows,
    evaluate_transition,
    slew_directions,
)

# Geometry shared across tests: spacecraft on a circular ~7000 km radius,
# static sun and inertial target directions. The general sun direction is +z,
# ~90 deg from every pointing used in the x-y plane, so it never confounds the
# timing/earth tests; the sun-on-slew-path test uses its own sun direction.
POS = Vec3(7000.0, 0.0, 0.0)
SUN_X = Vec3(1.0, 0.0, 0.0)
SUN_Z = Vec3(0.0, 0.0, 1.0)
DIR_A = Vec3(1.0, 0.0, 0.0).unit()           # away from Earth nadir (-x)
DIR_B = Vec3(0.0, 1.0, 0.0).unit()           # 90 deg from A, clears earth limb
DIR_NADIR = Vec3(-1.0, 0.0, 0.0).unit()      # straight into the Earth

HORIZON = TimeWindow(0.0, 3600.0)
FULL_ACCESS = (TimeWindow(0.0, 3600.0),)


def position_at(t):
    return POS


def sun_at_factory(sun):
    def _sun(t):
        return sun

    return _sun


def const_dir(d):
    return lambda t: d.unit()


def make_limits(**overrides):
    base = dict(
        max_rate_rad_s=math.radians(2.0),
        max_accel_rad_s2=math.radians(0.5),
        settle_s=20.0,
        min_sun_angle_rad=math.radians(45.0),
        earth_limb_margin_rad=0.0,
        slew_path_samples=12,
    )
    base.update(overrides)
    return ChainLimits(**base)


def target(name, direction, access=FULL_ACCESS):
    return ImagingTarget(name, tuple(access), const_dir(direction))


def all_reasons(result):
    return {r for b in result.blocked for r in b.reasons}


def test_two_target_chain_includes_slew_and_settle():
    limits = make_limits()
    slew_t = bang_bang_slew_time(
        math.pi / 2, limits.max_rate_rad_s, limits.max_accel_rad_s2
    )
    targets = [target('A', DIR_A), target('B', DIR_B)]
    chain = chain_imaging_windows(
        targets,
        HORIZON,
        exposure_s=10.0,
        dt_s=1.0,
        position_at=position_at,
        sun_at=sun_at_factory(SUN_Z),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), DIR_A),
    )

    assert chain.feasible
    opp_a = chain.opportunity('A')
    opp_b = chain.opportunity('B')
    # A is reachable immediately; B must wait for the 90 deg slew plus settle
    # after A's exposure ended at t=10.
    assert opp_a.window.start_tai_s == 0.0
    expected_earliest = 10.0 + slew_t + limits.settle_s
    assert abs(opp_b.window.start_tai_s - expected_earliest) < 1.0
    assert abs(opp_b.slew_angle_rad - math.pi / 2) < 1e-9
    assert abs(opp_b.ready_tai_s - opp_b.window.start_tai_s) < 1e-9

    # The blocked prefix on B is attributed to the transition time, not the
    # geometric access (B is geometrically available the whole horizon).
    res_b = chain.results[1]
    prefix = [b for b in res_b.blocked if b.window.start_tai_s < expected_earliest]
    assert prefix
    assert REASON_TRANSITION_TIME in prefix[0].reasons
    assert REASON_ACCESS not in prefix[0].reasons


def test_slew_through_sun_is_blocked_with_specific_reason():
    # A and B are 120 deg apart with the sun on the bisector of the slew path.
    # Each rest pointing clears the 45 deg sun cone (60 deg separation), but
    # the great-circle midpoint points straight at the sun.
    a = Vec3(1.0, math.tan(math.radians(60)), 0.0).unit()
    b = Vec3(1.0, -math.tan(math.radians(60)), 0.0).unit()
    assert abs(math.degrees(a.angle(SUN_X)) - 60.0) < 1e-9
    assert abs(math.degrees(a.angle(b)) - 120.0) < 1e-9

    limits = make_limits(min_sun_angle_rad=math.radians(45.0))
    chain = chain_imaging_windows(
        [target('A', a), target('B', b)],
        HORIZON,
        exposure_s=0.0,
        dt_s=10.0,
        position_at=position_at,
        sun_at=sun_at_factory(SUN_X),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), a),
    )

    res_b = chain.results[1]
    assert res_b.opportunity is None
    # The midpoint of the slew points at the sun: blocked on the slew-phase
    # sun constraint specifically, while exposure/access stay clean.
    assert REASON_SLEW_SUN in all_reasons(res_b)
    assert REASON_SUN not in all_reasons(res_b)
    assert REASON_ACCESS not in all_reasons(res_b)
    worst = {
        r: v
        for blk in res_b.blocked
        for r, v in blk.worst_margin_rad.items()
        if r == REASON_SLEW_SUN
    }
    assert worst
    assert worst[REASON_SLEW_SUN] <= -math.radians(44.0)


def test_slew_path_helper_is_great_circle():
    dirs = slew_directions(DIR_A, DIR_B, [0.0, 0.5, 1.0])
    assert (dirs[0] - DIR_A).norm() < 1e-12
    assert (dirs[-1] - DIR_B).norm() < 1e-12
    mid = dirs[1]
    assert abs(math.degrees(mid.angle(DIR_A)) - 45.0) < 1e-9
    assert abs(mid.angle(DIR_A) - mid.angle(DIR_B)) < 1e-9


def test_earth_limb_blocks_nadir_target():
    limits = make_limits()
    limb = math.degrees(math.asin(6378.137 / POS.norm()))
    assert limb > 65.0  # nadir is well inside the limb cone
    # Sun along +y: the (antiparallel) +x -> -x slew turns in the x-z plane,
    # staying 90 deg from the sun, so only the Earth limb can block it.
    sun_y = Vec3(0.0, 1.0, 0.0)
    chain = chain_imaging_windows(
        [target('N', DIR_NADIR)],
        HORIZON,
        exposure_s=5.0,
        dt_s=30.0,
        position_at=position_at,
        sun_at=sun_at_factory(sun_y),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), DIR_A),
    )
    res = chain.results[0]
    assert res.opportunity is None
    assert REASON_EARTH in all_reasons(res)
    # No sun-phase reason either: the blocking is attributed to Earth alone.
    assert not {r for r in all_reasons(res) if r.endswith('sun_keepout')}


def test_coupled_target_is_not_simple_union():
    # Both A and B clear the sun cone on their own and B is geometrically
    # available for the full horizon; nevertheless B cannot start until the
    # slew/settle from A completes. A per-target-union implementation would
    # mark B available at t=0.
    limits = make_limits()
    chain = chain_imaging_windows(
        [target('A', DIR_A), target('B', DIR_B)],
        HORIZON,
        exposure_s=10.0,
        dt_s=1.0,
        position_at=position_at,
        sun_at=sun_at_factory(SUN_Z),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), DIR_A),
    )
    res_b = chain.results[1]
    early = [
        b
        for b in res_b.blocked
        if b.window.start_tai_s == 0.0 and b.window.end_tai_s >= 25.0
    ]
    assert early
    assert REASON_TRANSITION_TIME in early[0].reasons


def test_access_gap_blocks_exposure_overrun():
    limits = make_limits()
    access = (TimeWindow(0.0, 100.0),)
    chain = chain_imaging_windows(
        [target('A', DIR_A, access=access)],
        TimeWindow(0.0, 200.0),
        exposure_s=10.0,
        dt_s=1.0,
        position_at=position_at,
        sun_at=sun_at_factory(SUN_Z),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), DIR_A),
    )
    res = chain.results[0]
    assert res.opportunity is not None
    # Starts after t=90 cannot fit the 10 s exposure inside the access window.
    tail = [b for b in res.blocked if b.window.start_tai_s >= 90.0]
    assert tail
    assert REASON_ACCESS in tail[0].reasons


def test_predecessor_failure_propagates():
    limits = make_limits()
    chain = chain_imaging_windows(
        [target('N', DIR_NADIR), target('B', DIR_B)],
        HORIZON,
        exposure_s=5.0,
        dt_s=30.0,
        position_at=position_at,
        sun_at=sun_at_factory(SUN_Z),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), DIR_A),
    )
    assert not chain.feasible
    assert chain.results[0].opportunity is None
    res_b = chain.results[1]
    assert res_b.opportunity is None
    assert REASON_PREDECESSOR in all_reasons(res_b)


def test_simultaneous_constraints_are_all_reported():
    # A target pointing straight at the sun through the Earth violates both
    # keepouts at once; the blocked interval must name both, not collapse to
    # a single generic "blocked" flag.
    sun_toward_nadir = Vec3(-1.0, 0.0, 0.0)
    limits = make_limits()
    chain = chain_imaging_windows(
        [target('X', DIR_NADIR)],
        HORIZON,
        exposure_s=5.0,
        dt_s=60.0,
        position_at=position_at,
        sun_at=sun_at_factory(sun_toward_nadir),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), DIR_A),
    )
    res = chain.results[0]
    assert res.opportunity is None
    reasons = all_reasons(res)
    assert REASON_SUN in reasons
    assert REASON_EARTH in reasons
    # At least one blocked interval carries both reasons simultaneously.
    assert any(
        REASON_SUN in b.reasons and REASON_EARTH in b.reasons
        for b in res.blocked
    )


def test_same_target_twice_needs_no_slew_but_waits_for_release():
    # Repointing at the same direction costs no slew/settle, but the second
    # exposure still cannot begin before the first one ends.
    limits = make_limits()
    chain = chain_imaging_windows(
        [target('A', DIR_A), target('A2', DIR_A)],
        HORIZON,
        exposure_s=30.0,
        dt_s=1.0,
        position_at=position_at,
        sun_at=sun_at_factory(SUN_Z),
        limits=limits,
        initial_quaternion=quaternion_between(Vec3(0, 0, 1), DIR_A),
    )
    assert chain.feasible
    opp_b = chain.opportunity('A2')
    assert opp_b.slew_duration_s < 1e-9
    assert opp_b.settle_s == 0.0
    assert abs(opp_b.window.start_tai_s - 30.0) < 1.0


def test_evaluate_transition_reports_phase_reasons_directly():
    limits = make_limits()
    # Starting a B exposure 1 s after release cannot fit slew + settle.
    verdict = evaluate_transition(
        DIR_A,
        0.0,
        target('B', DIR_B),
        1.0,
        position_at,
        sun_at_factory(SUN_Z),
        limits,
    )
    assert not verdict.feasible
    assert REASON_TRANSITION_TIME in verdict.reasons
    assert verdict.slew_angle_rad == math.pi / 2

    # Given enough time, the same pair is feasible (sun at +x is ~135 deg from
    # every station of the A->B path in the x-y plane).
    slew_t = bang_bang_slew_time(
        math.pi / 2, limits.max_rate_rad_s, limits.max_accel_rad_s2
    )
    verdict_ok = evaluate_transition(
        DIR_A,
        0.0,
        target('B', DIR_B),
        slew_t + limits.settle_s,
        position_at,
        sun_at_factory(SUN_Z),
        limits,
    )
    assert verdict_ok.feasible
