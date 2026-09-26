import math
from orbitforge.core.vector import Vec3
from orbitforge.core.state import TimeWindow
from orbitforge.environment.sun import sun_eci
from orbitforge.mission.imaging_windows import (
    EARTH_LIMB,
    MIN_DURATION,
    SLEW_PATH_SUN,
    SUN_KEEPOUT,
    TRANSITION_TIME,
    ImagingTarget,
    KeepoutSpec,
    SlewModel,
    blocker_summary,
    imaging_windows,
    screen_target,
)

DEG = math.pi / 180.0
SLEW = SlewModel(max_rate_rad_s=0.01, max_accel_rad_s2=0.001, settle_s=30.0)
# bang-bang time for a 90 deg slew with the model above
SLEW90 = 2 * 0.01 / 0.001 + (math.pi / 2 - 0.1) / 0.01
NO_KEEPOUT = KeepoutSpec(sun_min_angle_rad=0.0, earth_limb_margin_rad=-math.pi)
POS = lambda t: Vec3(7000, 0, 0)
ZENITH_SUN = lambda t: Vec3(0, 0, 1)

def fixed(direction):
    return lambda t: direction

def test_slew_and_settle_gate_second_target_not_union():
    a = ImagingTarget('A', fixed(Vec3(1, 0, 0)), min_dwell_s=60)
    b = ImagingTarget('B', fixed(Vec3(0, 1, 0)), min_dwell_s=60)
    out = imaging_windows([a, b], POS, SLEW, NO_KEEPOUT, 0, 1000, sun_direction_at=ZENITH_SUN, step=10)
    wa, wb = out
    # per-target screening alone would report B clear for the whole span
    screened_b = screen_target(b, POS, NO_KEEPOUT, 0, 1000, sun_direction_at=ZENITH_SUN, step=10)
    assert screened_b.windows == (TimeWindow(0, 1000),)
    # joint chain: B cannot start before dwell(A) + slew + settle
    assert len(wb.windows) == 1
    assert abs(wb.windows[0].start_tai_s - (60 + SLEW90 + 30)) < 1e-06
    assert wb.windows[0].end_tai_s == 1000
    # and A must release the camera early enough to hand off to B
    assert len(wa.windows) == 1
    assert abs(wa.windows[0].end_tai_s - (1000 - 60 - 30 - SLEW90)) < 1e-06
    # both trimmed edges are attributed to the transition, not the keepouts
    assert any((b_.constraint == TRANSITION_TIME and b_.end_tai_s == wb.windows[0].start_tai_s for b_ in wb.blockers))
    assert any((b_.constraint == TRANSITION_TIME and b_.start_tai_s == wa.windows[0].end_tai_s for b_ in wa.blockers))

def test_sun_keepout_blocks_and_is_named():
    t1 = ImagingTarget('T', fixed(Vec3(1, 0, 0)), min_dwell_s=60)
    keepout = KeepoutSpec(sun_min_angle_rad=30 * DEG, earth_limb_margin_rad=-math.pi)
    out = imaging_windows([t1], POS, SLEW, keepout, 0, 500, sun_direction_at=fixed(Vec3(1, 0, 0)), step=10)
    assert out[0].windows == ()
    blockers = [b for b in out[0].blockers if b.constraint == SUN_KEEPOUT]
    assert blockers and blockers[0].start_tai_s == 0 and (blockers[0].end_tai_s == 500)

def test_earth_limb_blocks_and_is_named():
    t1 = ImagingTarget('T', fixed(Vec3(-1, 0, 0)), min_dwell_s=60)
    keepout = KeepoutSpec(sun_min_angle_rad=0.0, earth_limb_margin_rad=0.0)
    out = imaging_windows([t1], POS, SLEW, keepout, 0, 500, sun_direction_at=ZENITH_SUN, step=10)
    assert out[0].windows == ()
    assert any((b.constraint == EARTH_LIMB for b in out[0].blockers))

def test_slew_path_sun_violation_blocks_transition():
    # endpoints are 45.9 deg from the sun, but the great-circle slew
    # path between them passes within ~10 deg of it
    sun = Vec3(1, 1, 0.25)
    keepout = KeepoutSpec(sun_min_angle_rad=30 * DEG, earth_limb_margin_rad=-math.pi)
    a = ImagingTarget('A', fixed(Vec3(1, 0, 0)), min_dwell_s=60)
    b = ImagingTarget('B', fixed(Vec3(0, 1, 0)), min_dwell_s=60)
    out = imaging_windows([a, b], POS, SLEW, keepout, 0, 1000, sun_direction_at=fixed(sun), step=10)
    wa, wb = out
    assert wb.windows == ()
    assert any((blk.constraint == SLEW_PATH_SUN for blk in wb.blockers))
    # the broken chain also explains why A cannot be used in the sequence
    assert wa.windows == ()
    assert any((blk.constraint == SLEW_PATH_SUN for blk in wa.blockers))

def test_three_target_chain_forward_and_backward_trim():
    def sun(t):
        return Vec3(0, 0, 1) if t <= 600 else Vec3(-1, 0, 0)
    keepout = KeepoutSpec(sun_min_angle_rad=30 * DEG, earth_limb_margin_rad=-math.pi)
    a = ImagingTarget('A', fixed(Vec3(1, 0, 0)), min_dwell_s=60)
    b = ImagingTarget('B', fixed(Vec3(0, 1, 0)), min_dwell_s=60)
    c = ImagingTarget('C', fixed(Vec3(-1, 0, 0)), min_dwell_s=60)
    wa, wb, wc = imaging_windows([a, b, c], POS, SLEW, keepout, 0, 1000, sun_direction_at=sun, step=10)
    # C is sun-blocked after t=600, so the whole chain must finish in time
    assert any((blk.constraint == SUN_KEEPOUT for blk in wc.blockers))
    assert len(wc.windows) == 1
    assert abs(wc.windows[0].start_tai_s - (180 + 2 * SLEW90)) < 1e-06
    assert wc.windows[0].end_tai_s == 600
    # B is squeezed from both sides by the two slews
    assert len(wb.windows) == 1
    assert abs(wb.windows[0].start_tai_s - (60 + SLEW90 + 30)) < 1e-06
    assert abs(wb.windows[0].end_tai_s - (600 - 60 - 30 - SLEW90)) < 1e-06
    # A must finish early enough for the whole chain to complete
    assert len(wa.windows) == 1
    assert wa.windows[0].start_tai_s == 0
    assert abs(wa.windows[0].end_tai_s - (600 - 2 * (60 + 30 + SLEW90))) < 1e-06
    summary = blocker_summary([wa, wb, wc])
    assert summary[TRANSITION_TIME] > 0 and summary[SUN_KEEPOUT] > 0

def test_delayed_slew_waits_for_window_open():
    # sun sits on B's boresight until t=500, so the slew from A can only
    # start once the window opens; arrival is one slew time later
    def sun(t):
        return Vec3(0, 1, 0) if t < 500 else Vec3(0, 0, 1)
    keepout = KeepoutSpec(sun_min_angle_rad=30 * DEG, earth_limb_margin_rad=-math.pi)
    a = ImagingTarget('A', fixed(Vec3(1, 0, 0)), min_dwell_s=60)
    b = ImagingTarget('B', fixed(Vec3(0, 1, 0)), min_dwell_s=60)
    wa, wb = imaging_windows([a, b], POS, SLEW, keepout, 0, 1000, sun_direction_at=sun, step=10)
    assert len(wb.windows) == 1
    assert abs(wb.windows[0].start_tai_s - (500 + SLEW90 + 30)) < 1e-06
    assert any((blk.constraint == SUN_KEEPOUT for blk in wb.blockers))
    assert any((blk.constraint == TRANSITION_TIME for blk in wb.blockers))
    assert len(wa.windows) == 1
    assert abs(wa.windows[0].end_tai_s - (1000 - 60 - 30 - SLEW90)) < 1e-06

def test_short_clear_span_flagged_min_duration():
    def sun(t):
        return Vec3(0, 0, 1) if t <= 30 else Vec3(1, 0, 0)
    keepout = KeepoutSpec(sun_min_angle_rad=30 * DEG, earth_limb_margin_rad=-math.pi)
    t1 = ImagingTarget('T', fixed(Vec3(1, 0, 0)), min_dwell_s=60)
    out = imaging_windows([t1], POS, SLEW, keepout, 0, 500, sun_direction_at=sun, step=10)
    assert out[0].windows == ()
    assert any((b.constraint == MIN_DURATION for b in out[0].blockers))
    assert any((b.constraint == SUN_KEEPOUT for b in out[0].blockers))

def test_default_sun_ephemeris_smoke():
    pos = lambda t: Vec3(0, 0, 7200)
    anti_sun = ImagingTarget('T', lambda t: pos(t) - sun_eci(t), min_dwell_s=60)
    keepout = KeepoutSpec(sun_min_angle_rad=30 * DEG, earth_limb_margin_rad=0.0)
    out = imaging_windows([anti_sun], pos, SLEW, keepout, 1_700_000_000, 1_700_001_800, step=60)
    assert out[0].windows == (TimeWindow(1_700_000_000, 1_700_001_800),)
