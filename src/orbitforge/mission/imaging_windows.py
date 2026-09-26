from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable
from orbitforge.core.state import TimeWindow
from orbitforge.core.vector import Vec3
from orbitforge.attitude.pointing import quaternion_between
from orbitforge.attitude.quaternion import Quaternion, slerp
from orbitforge.attitude.slew import bang_bang_slew_time
from orbitforge.attitude.sun_constraint import keepout_ok, limb_keepout
from orbitforge.environment.sun import sun_eci

SUN_KEEPOUT = 'sun_keepout'
EARTH_LIMB = 'earth_limb'
SLEW_PATH_SUN = 'slew_path_sun'
SLEW_PATH_LIMB = 'slew_path_earth_limb'
TRANSITION_TIME = 'transition_time'
MIN_DURATION = 'min_duration'

_IDENTITY = Quaternion(1.0, 0.0, 0.0, 0.0)

@dataclass(frozen=True)
class ImagingTarget:
    name: str
    boresight_at: Callable[[float], Vec3]
    min_dwell_s: float = 60.0

@dataclass(frozen=True)
class SlewModel:
    max_rate_rad_s: float
    max_accel_rad_s2: float
    settle_s: float = 0.0

@dataclass(frozen=True)
class KeepoutSpec:
    sun_min_angle_rad: float = 0.0
    earth_limb_margin_rad: float = 0.0
    earth_radius_km: float = 6378.137

@dataclass(frozen=True)
class WindowBlocker:
    constraint: str
    start_tai_s: float
    end_tai_s: float
    detail: str

@dataclass(frozen=True)
class TargetWindows:
    target: str
    windows: tuple[TimeWindow, ...]
    blockers: tuple[WindowBlocker, ...]

    @property
    def feasible(self):
        return bool(self.windows)

def sun_direction_from_ephemeris(position_at):
    def direction(t):
        return (sun_eci(t) - position_at(t)).unit()
    return direction

def screen_target(target: ImagingTarget, position_at, keepout: KeepoutSpec, start, end, sun_direction_at=None, step=30.0):
    """Intervals where one target's boresight satisfies both keepouts.

    Every excluded sample run is recorded as a WindowBlocker naming the
    constraint (SUN_KEEPOUT or EARTH_LIMB) and its worst margin.
    """
    if sun_direction_at is None:
        sun_direction_at = sun_direction_from_ephemeris(position_at)
    samples = []
    t = start
    while t <= end + 1e-09:
        pos = position_at(t)
        d = target.boresight_at(t)
        sun_ok = keepout_ok(d, sun_direction_at(t), keepout.sun_min_angle_rad)
        limb_ok = limb_keepout(pos, d, keepout.earth_radius_km, keepout.earth_limb_margin_rad)
        sun_ang = None if sun_ok else d.unit().angle(sun_direction_at(t).unit())
        nadir_ang = None if limb_ok else (pos * -1).unit().angle(d.unit())
        samples.append((t, sun_ok, limb_ok, sun_ang, nadir_ang))
        t += step
    windows = []
    run_start = None
    prev_t = None
    for s in samples:
        if s[1] and s[2]:
            if run_start is None:
                run_start = s[0]
        elif run_start is not None:
            windows.append(TimeWindow(run_start, prev_t))
            run_start = None
        prev_t = s[0]
    if run_start is not None:
        windows.append(TimeWindow(run_start, prev_t))
    blockers = []
    for s, e, worst in _violation_runs(samples, 1, 3):
        blockers.append(WindowBlocker(SUN_KEEPOUT, s, e, f'sun angle down to {math.degrees(worst):.1f} deg, limit {math.degrees(keepout.sun_min_angle_rad):.1f} deg'))
    for s, e, worst in _violation_runs(samples, 2, 4):
        blockers.append(WindowBlocker(EARTH_LIMB, s, e, f'nadir angle down to {math.degrees(worst):.1f} deg, inside earth limb + margin'))
    kept = []
    for w in windows:
        if w.duration_s + 1e-09 >= target.min_dwell_s:
            kept.append(w)
        else:
            blockers.append(WindowBlocker(MIN_DURATION, w.start_tai_s, w.end_tai_s, f'clear span {w.duration_s:.1f}s shorter than dwell {target.min_dwell_s:.1f}s'))
    blockers.sort(key=lambda b: (b.start_tai_s, b.constraint))
    return TargetWindows(target.name, tuple(kept), tuple(blockers))

def _violation_runs(samples, flag_idx, angle_idx):
    runs = []
    run_start = None
    worst = None
    prev_t = None
    for s in samples:
        if not s[flag_idx]:
            if run_start is None:
                run_start = s[0]
                worst = s[angle_idx]
            else:
                worst = min(worst, s[angle_idx])
        elif run_start is not None:
            runs.append((run_start, prev_t, worst))
            run_start = None
        prev_t = s[0]
    if run_start is not None:
        runs.append((run_start, prev_t, worst))
    return runs

def transition_clear(a: ImagingTarget, b: ImagingTarget, t_slew_start, slew: SlewModel, keepout: KeepoutSpec, position_at, sun_direction_at, path_samples=9):
    """Feasibility of slewing from a to b starting at t_slew_start.

    Returns (ok, slew_seconds, blockers). The boresight path is interpolated
    with the quaternion chain and screened against both keepouts, so a
    transition can be blocked even when both endpoints are clear.
    """
    da = a.boresight_at(t_slew_start).unit()
    db = b.boresight_at(t_slew_start).unit()
    slew_s = 0.0
    for _ in range(2):
        slew_s = bang_bang_slew_time(da.angle(db), slew.max_rate_rad_s, slew.max_accel_rad_s2)
        db = b.boresight_at(t_slew_start + slew_s).unit()
    if da.angle(db) < 1e-12:
        return (True, slew_s, [])
    q = quaternion_between(da, db)
    worst_sun = None
    worst_limb = None
    for k in range(path_samples + 1):
        f = k / path_samples
        d = slerp(_IDENTITY, q, f).rotate(da)
        t = t_slew_start + f * slew_s
        if not keepout_ok(d, sun_direction_at(t), keepout.sun_min_angle_rad):
            ang = d.angle(sun_direction_at(t).unit())
            worst_sun = ang if worst_sun is None else min(worst_sun, ang)
        pos = position_at(t)
        if not limb_keepout(pos, d, keepout.earth_radius_km, keepout.earth_limb_margin_rad):
            ang = (pos * -1).unit().angle(d)
            worst_limb = ang if worst_limb is None else min(worst_limb, ang)
    blockers = []
    if worst_sun is not None:
        blockers.append(WindowBlocker(SLEW_PATH_SUN, t_slew_start, t_slew_start + slew_s, f'slew path {a.name}->{b.name} dips to {math.degrees(worst_sun):.1f} deg from sun'))
    if worst_limb is not None:
        blockers.append(WindowBlocker(SLEW_PATH_LIMB, t_slew_start, t_slew_start + slew_s, f'slew path {a.name}->{b.name} dips to {math.degrees(worst_limb):.1f} deg from nadir, inside earth limb + margin'))
    return (not blockers, slew_s, blockers)

def _earliest_arrival(a: ImagingTarget, b: ImagingTarget, t_hand, w_start, slew: SlewModel, keepout: KeepoutSpec, position_at, sun_direction_at, path_samples):
    """Earliest epoch the boresight can arrive on b at/after w_start.

    The immediate slew is not allowed to arrive before the clear window
    opens, so delayed slews are screened at their own epochs: arrive at
    window open, or start the slew inside the window. Returns
    (arrival_epoch, blockers); arrival is None when no candidate path is
    clear, and blockers then names the constraint behind every attempt.
    """
    ok, slew_s, blockers = transition_clear(a, b, t_hand, slew, keepout, position_at, sun_direction_at, path_samples)
    if t_hand + slew_s >= w_start - 1e-09:
        return (max(w_start, t_hand + slew_s), []) if ok else (None, blockers)
    all_blockers = []
    for t_s in (max(t_hand, w_start - slew_s), w_start):
        ok, slew_s, blockers = transition_clear(a, b, t_s, slew, keepout, position_at, sun_direction_at, path_samples)
        if ok:
            return (max(w_start, t_s + slew_s), [])
        all_blockers.extend(blockers)
    return (None, all_blockers)

def imaging_windows(targets, position_at, slew: SlewModel, keepout: KeepoutSpec, start, end, sun_direction_at=None, step=30.0, path_samples=9):
    """Jointly feasible imaging windows for an ordered target sequence.

    Each target is first screened against the sun and earth-limb keepouts,
    then consecutive targets are coupled: imaging of target i+1 cannot start
    until the slew from target i finishes and the settle time elapses inside
    a clear window, and the slew path itself must stay clear of both
    keepouts. A forward pass propagates earliest feasible starts, a
    backward pass propagates latest feasible ends, and every trimmed or
    dropped interval keeps a WindowBlocker naming the constraint responsible
    (never a plain per-target union). The first target is not charged settle
    time; forward transitions are evaluated at the earliest handoff epoch,
    delayed so acquisition lands inside the clear window when needed.
    """
    if sun_direction_at is None:
        sun_direction_at = sun_direction_from_ephemeris(position_at)
    if not targets:
        return []
    screened = [screen_target(t, position_at, keepout, start, end, sun_direction_at, step) for t in targets]
    n = len(targets)
    clear = [list(tw.windows) for tw in screened]
    blockers = [list(tw.blockers) for tw in screened]
    dwell = [t.min_dwell_s for t in targets]
    tol = 1e-09
    earliest = [[None] * len(clear[i]) for i in range(n)]
    latest = [[None] * len(clear[i]) for i in range(n)]
    for j, w in enumerate(clear[0]):
        earliest[0][j] = w.start_tai_s
    for i in range(1, n):
        for j, w in enumerate(clear[i]):
            best = None
            reasons = []
            for k in range(len(clear[i - 1])):
                es = earliest[i - 1][k]
                if es is None:
                    continue
                t_hand = es + dwell[i - 1]
                arrival, path_blockers = _earliest_arrival(targets[i - 1], targets[i], t_hand, w.start_tai_s, slew, keepout, position_at, sun_direction_at, path_samples)
                if arrival is None:
                    reasons.extend(path_blockers)
                    continue
                est = arrival + slew.settle_s
                if est + dwell[i] <= w.end_tai_s + tol:
                    best = est if best is None else min(best, est)
                else:
                    reasons.append(WindowBlocker(TRANSITION_TIME, w.start_tai_s, w.end_tai_s, f'slew+settle from {targets[i - 1].name} ready at {est:.1f}, dwell overruns window end by {est + dwell[i] - w.end_tai_s:.1f}s'))
            if best is None:
                if not reasons:
                    reasons.append(WindowBlocker(TRANSITION_TIME, w.start_tai_s, w.end_tai_s, f'no feasible handoff from {targets[i - 1].name}'))
                blockers[i].extend(reasons)
            else:
                earliest[i][j] = best
                if best > w.start_tai_s + tol:
                    blockers[i].append(WindowBlocker(TRANSITION_TIME, w.start_tai_s, best, f'leading edge held by slew+settle from {targets[i - 1].name}'))
    for j, w in enumerate(clear[-1]):
        latest[n - 1][j] = w.end_tai_s
    for i in range(n - 2, -1, -1):
        for j, w in enumerate(clear[i]):
            best = None
            reasons = []
            for k in range(len(clear[i + 1])):
                le = latest[i + 1][k]
                if le is None:
                    continue
                succ_latest_start = le - dwell[i + 1]
                slew_s = 0.0
                ok = True
                path_blockers = []
                for _ in range(2):
                    t_guess = max(w.start_tai_s, succ_latest_start - slew.settle_s - slew_s)
                    ok, slew_s, path_blockers = transition_clear(targets[i], targets[i + 1], t_guess, slew, keepout, position_at, sun_direction_at, path_samples)
                if not ok:
                    reasons.extend(path_blockers)
                    continue
                end_limit = succ_latest_start - slew.settle_s - slew_s
                hi = min(w.end_tai_s, end_limit)
                if hi - dwell[i] >= w.start_tai_s - tol:
                    best = hi if best is None else max(best, hi)
                else:
                    reasons.append(WindowBlocker(TRANSITION_TIME, w.start_tai_s, w.end_tai_s, f'must finish by {end_limit:.1f} to hand off to {targets[i + 1].name}, dwell does not fit'))
            if best is None:
                if not reasons:
                    reasons.append(WindowBlocker(TRANSITION_TIME, w.start_tai_s, w.end_tai_s, f'no feasible handoff to {targets[i + 1].name}'))
                blockers[i].extend(reasons)
            else:
                latest[i][j] = best
                if best < w.end_tai_s - tol:
                    blockers[i].append(WindowBlocker(TRANSITION_TIME, best, w.end_tai_s, f'trailing edge released for slew+settle to {targets[i + 1].name}'))
    out = []
    for i in range(n):
        final = []
        for j, w in enumerate(clear[i]):
            lo = earliest[i][j]
            hi = latest[i][j]
            if lo is None or hi is None:
                continue
            if hi - lo + tol >= dwell[i]:
                final.append(TimeWindow(lo, hi))
            else:
                blockers[i].append(WindowBlocker(TRANSITION_TIME, lo, hi, f'jointly feasible span {hi - lo:.1f}s shorter than dwell {dwell[i]:.1f}s'))
        blockers[i].sort(key=lambda b: (b.start_tai_s, b.constraint))
        out.append(TargetWindows(targets[i].name, tuple(final), tuple(blockers[i])))
    return out

def blocker_summary(target_windows):
    counts = {}
    for tw in target_windows:
        for b in tw.blockers:
            counts[b.constraint] = counts.get(b.constraint, 0) + 1
    return counts
