"""Imaging opportunity windows for an attitude chain.

The attitude chain already owns quaternions, the bang-bang slew model and the
sun/earth keepout tools. This module wires them into the imaging timeline:

* the slew *and* settle between two consecutive locked targets consume real
  time, so a target is only usable if enough time has elapsed since the end of
  the previous target's exposure;
* the sun keepout and earth-limb keepout are checked on three phases:
  along the eigenaxis slew path, during the on-target settle dwell and during
  the exposure itself;
* every rejected start time carries the exact reason(s) it was rejected, so a
  conflict can be attributed to a specific constraint instead of merging
  independently computed per-target windows.

Slews are modelled as the minimum rotation between the two boresight
directions (eigenaxis = d_prev x d_next). Slerping two attitude quaternions is
not correct for the timing: quaternions built to align a boresight carry an
arbitrary roll about it, so the quaternion path is longer than the geometric
minimum and does not trace the great circle of boresight directions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

from orbitforge.core.constants import R_EARTH_EQUATOR_KM
from orbitforge.core.state import TimeWindow
from orbitforge.core.vector import Vec3
from orbitforge.attitude.pointing import nadir_direction
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.attitude.slew import bang_bang_slew_time

# Blocking reason tags. The slew_/settle_ prefixes say which phase failed so a
# conflict can be attributed to a specific constraint and phase.
REASON_SUN = 'sun_keepout'
REASON_EARTH = 'earth_limb'
REASON_SLEW_SUN = 'slew_sun_keepout'
REASON_SLEW_EARTH = 'slew_earth_limb'
REASON_SETTLE_SUN = 'settle_sun_keepout'
REASON_SETTLE_EARTH = 'settle_earth_limb'
REASON_TRANSITION_TIME = 'slew_settle_time'
REASON_ACCESS = 'geometry_access'
REASON_PREDECESSOR = 'predecessor_unavailable'


@dataclass(frozen=True)
class ImagingTarget:
    """One imaging target.

    ``access_windows`` are the geometric visibility intervals (e.g. passes).
    ``direction_at(t)`` returns the required ECI boresight unit direction at
    time t, allowing a tracking target whose direction moves with time.
    """

    name: str
    access_windows: tuple[TimeWindow, ...]
    direction_at: Callable[[float], Vec3]


@dataclass(frozen=True)
class ChainLimits:
    max_rate_rad_s: float
    max_accel_rad_s2: float
    settle_s: float
    min_sun_angle_rad: float
    earth_limb_margin_rad: float = 0.0
    earth_radius_km: float = R_EARTH_EQUATOR_KM
    body_boresight: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 1.0))
    slew_path_samples: int = 8


@dataclass(frozen=True)
class BlockedInterval:
    """A time interval on which an exposure may not start.

    ``reasons`` is the set of every constraint violated on the interval (not a
    flat union over per-target evaluations). ``worst_margin_rad`` gives the most
    negative clearance per geometric reason, for reporting how hard the
    conflict bites.
    """

    window: TimeWindow
    reasons: frozenset[str]
    worst_margin_rad: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ImagingOpportunity:
    target: str
    window: TimeWindow
    slew_release_tai_s: float
    slew_start_tai_s: float
    slew_angle_rad: float
    slew_duration_s: float
    settle_s: float

    @property
    def ready_tai_s(self):
        return self.slew_start_tai_s + self.slew_duration_s + self.settle_s


@dataclass(frozen=True)
class TargetWindowResult:
    target: str
    feasible_windows: tuple[TimeWindow, ...]
    opportunity: ImagingOpportunity | None
    blocked: tuple[BlockedInterval, ...]


@dataclass(frozen=True)
class ChainResult:
    results: tuple[TargetWindowResult, ...]
    feasible: bool

    def opportunity(self, name: str):
        for r in self.results:
            if r.target == name:
                return r.opportunity
        raise KeyError(name)


@dataclass(frozen=True)
class TransitionVerdict:
    """Pair-level verdict for the transition into one candidate start time."""

    feasible: bool
    slew_angle_rad: float
    slew_duration_s: float
    ready_tai_s: float
    reasons: frozenset[str]
    worst_margin_rad: Mapping[str, float]


def slew_directions(prev_dir: Vec3, next_dir: Vec3, fractions):
    """Boresight directions along the minimum eigenaxis slew path."""
    a = prev_dir.unit()
    b = next_dir.unit()
    angle = a.angle(b)
    if angle < 1e-12:
        return [a for _ in fractions]
    axis = a.cross(b)
    if axis.norm() < 1e-12:
        # Antiparallel pointings: the eigenaxis is undetermined, pick any
        # unit vector perpendicular to a.
        helper = Vec3(1.0, 0.0, 0.0) if abs(a.x) < 0.9 else Vec3(0.0, 1.0, 0.0)
        axis = helper.cross(a).unit()
    else:
        axis = axis.unit()
    return [Quaternion.from_axis_angle(axis, f * angle).rotate(a) for f in fractions]


def sun_clearance(boresight: Vec3, sun_dir: Vec3, min_angle_rad: float) -> float:
    """Positive when the boresight clears the sun keepout cone."""
    return boresight.angle(sun_dir.unit()) - min_angle_rad


def earth_clearance(position: Vec3, boresight: Vec3, limits: ChainLimits) -> float:
    """Positive when the boresight stays off the earth limb plus margin."""
    r = position.norm()
    limb = math.asin(min(1.0, limits.earth_radius_km / r))
    return nadir_direction(position).angle(boresight.unit()) - (
        limb + limits.earth_limb_margin_rad
    )


def evaluate_transition(
    prev_boresight: Vec3,
    release_tai_s: float,
    target: ImagingTarget,
    start_tai_s: float,
    position_at: Callable[[float], Vec3],
    sun_at: Callable[[float], Vec3],
    limits: ChainLimits,
) -> TransitionVerdict:
    """Check slew + settle from a released attitude into a target at ``start``.

    Access and exposure-duration constraints are deliberately not checked
    here; this is the primitive that attributes a conflict to the transition
    between two pointings.
    """
    reasons: set[str] = set()
    worst: dict[str, float] = {}

    def record(reason_sun, reason_earth, margin_sun, margin_earth):
        if margin_sun < 0.0:
            reasons.add(reason_sun)
            worst[reason_sun] = min(worst.get(reason_sun, margin_sun), margin_sun)
        if margin_earth < 0.0:
            reasons.add(reason_earth)
            worst[reason_earth] = min(worst.get(reason_earth, margin_earth), margin_earth)

    end_dir = target.direction_at(start_tai_s).unit()
    angle = prev_boresight.angle(end_dir)
    slew_t = bang_bang_slew_time(
        angle, limits.max_rate_rad_s, limits.max_accel_rad_s2
    )
    # Already on target: no slew and nothing to settle.
    settle_t = 0.0 if angle < 1e-12 else limits.settle_s
    # Schedule the slew as late as possible so the settle dwell ends exactly at
    # exposure start: this leaves no unchecked idle hold. The transition only
    # needs the slew to fit after the release time.
    slew_start = start_tai_s - settle_t - slew_t
    ready = slew_start + slew_t + settle_t

    # Along the slew path: sun/earth geometry evaluated at the actual time of
    # each path station.
    fracs = [k / limits.slew_path_samples for k in range(limits.slew_path_samples + 1)]
    for f, d in zip(fracs, slew_directions(prev_boresight, end_dir, fracs)):
        t = slew_start + f * slew_t
        record(
            REASON_SLEW_SUN,
            REASON_SLEW_EARTH,
            sun_clearance(d, sun_at(t), limits.min_sun_angle_rad),
            earth_clearance(position_at(t), d, limits),
        )

    # Settle dwell: holding the final set-point until exposure start.
    if settle_t > 0.0:
        for t in (slew_start + slew_t + settle_t / 2.0, start_tai_s):
            d = target.direction_at(t).unit()
            record(
                REASON_SETTLE_SUN,
                REASON_SETTLE_EARTH,
                sun_clearance(d, sun_at(t), limits.min_sun_angle_rad),
                earth_clearance(position_at(t), d, limits),
            )

    if slew_start + 1e-9 < release_tai_s:
        reasons.add(REASON_TRANSITION_TIME)

    return TransitionVerdict(
        feasible=not reasons,
        slew_angle_rad=angle,
        slew_duration_s=slew_t,
        ready_tai_s=ready,
        reasons=frozenset(reasons),
        worst_margin_rad=worst,
    )


def _exposure_clear(target, start_tai_s, exposure_s, sun_at, limits, position_at):
    """Keepout over the exposure itself; returns (reasons, worst margins)."""
    reasons: set[str] = set()
    worst: dict[str, float] = {}
    end_tai_s = start_tai_s + exposure_s
    check_ts = (start_tai_s, end_tai_s) if exposure_s > 0 else (start_tai_s,)
    for t in check_ts:
        d = target.direction_at(t).unit()
        m_sun = sun_clearance(d, sun_at(t), limits.min_sun_angle_rad)
        m_earth = earth_clearance(position_at(t), d, limits)
        if m_sun < 0.0:
            reasons.add(REASON_SUN)
            worst[REASON_SUN] = min(worst.get(REASON_SUN, m_sun), m_sun)
        if m_earth < 0.0:
            reasons.add(REASON_EARTH)
            worst[REASON_EARTH] = min(worst.get(REASON_EARTH, m_earth), m_earth)
    return reasons, worst


def _access_contains(windows, start_tai_s, end_tai_s):
    for w in windows:
        if w.start_tai_s <= start_tai_s and end_tai_s <= w.end_tai_s:
            return True
    return False


def _blocked_runs(marked, dt, horizon_end):
    """Group consecutive samples sharing a reason set into BlockedIntervals.

    Worst margins are aggregated per run, so two disjoint intervals with the
    same reason set keep independent clearance numbers.
    """
    out = []
    run_start = run_key = None
    run_margins: dict[str, float] = {}
    last_t = None

    def close(end_t):
        out.append(
            BlockedInterval(
                TimeWindow(run_start, min(end_t, horizon_end)),
                run_key,
                dict(run_margins),
            )
        )

    for t, key, payload in marked:
        if key is None:
            if run_start is not None:
                close(last_t + dt)
                run_start = run_key = None
                run_margins = {}
            last_t = t
            continue
        if run_start is None or key != run_key:
            if run_start is not None:
                close(last_t + dt)
            run_start = t
            run_key = key
            run_margins = dict(payload)
        else:
            for reason, margin in payload.items():
                run_margins[reason] = min(run_margins.get(reason, margin), margin)
        last_t = t
    if run_start is not None:
        close(last_t + dt)
    return out


def chain_imaging_windows(
    targets,
    horizon: TimeWindow,
    exposure_s: float,
    dt_s: float,
    position_at: Callable[[float], Vec3],
    sun_at: Callable[[float], Vec3],
    limits: ChainLimits,
    initial_quaternion: Quaternion | None = None,
    initial_release_tai_s: float | None = None,
) -> ChainResult:
    """Compute locked imaging opportunities for an ordered chain of targets.

    Targets are processed in order and each is placed at its earliest feasible
    exposure start, conditional on the previous target's *locked* placement.
    Candidate start times are sampled every ``dt_s``; feasibility of a start
    time requires, simultaneously:

    1. the whole exposure lies inside one geometric access window;
    2. slew + settle since the previous exposure end fit before the start;
    3. the slew path clears the sun cone and earth limb;
    4. the settle dwell clears both keepouts;
    5. the exposure itself clears both keepouts.

    Every infeasible sample is tagged with the exact reason(s), so the blocked
    intervals on each target explain which constraint (and which phase) is in
    the way. This is not a union of per-target judgements: criteria 2-4 couple
    consecutive targets.
    """
    if initial_quaternion is None:
        initial_quaternion = Quaternion(1.0, 0.0, 0.0, 0.0)
    release_t = (
        horizon.start_tai_s
        if initial_release_tai_s is None
        else initial_release_tai_s
    )
    prev_boresight = initial_quaternion.rotate(limits.body_boresight).unit()

    results = []
    chain_ok = True
    last_start = horizon.start_tai_s
    span_end = horizon.end_tai_s - exposure_s

    for target in targets:
        if not chain_ok:
            results.append(
                TargetWindowResult(
                    target.name,
                    (),
                    None,
                    (
                        BlockedInterval(
                            horizon,
                            frozenset({REASON_PREDECESSOR}),
                        ),
                    ),
                )
            )
            continue

        sample_count = int((span_end - last_start) // dt_s) + 1
        marked = []  # (t, reasons-key or None, margins)
        feasible_runs = []
        run_start = None
        prev_t = None
        first_feasible = None

        if span_end < last_start:
            # No candidate start time can fit the exposure in the horizon.
            marked = [
                (
                    last_start,
                    frozenset({REASON_ACCESS}),
                    {},
                )
            ]

        for i in range(max(0, sample_count)):
            t = last_start + i * dt_s
            reasons: set[str] = set()
            worst: dict[str, float] = {}

            if not _access_contains(target.access_windows, t, t + exposure_s):
                reasons.add(REASON_ACCESS)
            else:
                verdict = evaluate_transition(
                    prev_boresight,
                    release_t,
                    target,
                    t,
                    position_at,
                    sun_at,
                    limits,
                )
                reasons.update(verdict.reasons)
                for k, v in verdict.worst_margin_rad.items():
                    worst[k] = min(worst.get(k, v), v)
                exp_reasons, exp_worst = _exposure_clear(
                    target, t, exposure_s, sun_at, limits, position_at
                )
                reasons.update(exp_reasons)
                for k, v in exp_worst.items():
                    worst[k] = min(worst.get(k, v), v)

            if reasons:
                if run_start is not None:
                    feasible_runs.append(
                        TimeWindow(run_start, min(prev_t + dt_s, horizon.end_tai_s))
                    )
                    run_start = None
                marked.append((t, frozenset(reasons), dict(worst)))
            else:
                if run_start is None:
                    run_start = t
                marked.append((t, None, None))
                if first_feasible is None:
                    first_feasible = t
            prev_t = t

        if run_start is not None:
            feasible_runs.append(
                TimeWindow(run_start, min(prev_t + dt_s, horizon.end_tai_s))
            )

        blocked = tuple(_blocked_runs(marked, dt_s, horizon.end_tai_s))

        if first_feasible is None:
            results.append(TargetWindowResult(target.name, tuple(feasible_runs), None, blocked))
            chain_ok = False
            continue

        start = first_feasible
        end_dir = target.direction_at(start).unit()
        angle = prev_boresight.angle(end_dir)
        slew_t = bang_bang_slew_time(
            angle, limits.max_rate_rad_s, limits.max_accel_rad_s2
        )
        settle_t = 0.0 if angle < 1e-12 else limits.settle_s
        slew_start = start - settle_t - slew_t
        opportunity = ImagingOpportunity(
            target.name,
            TimeWindow(start, start + exposure_s),
            release_t,
            slew_start,
            angle,
            slew_t,
            settle_t,
        )
        results.append(
            TargetWindowResult(target.name, tuple(feasible_runs), opportunity, blocked)
        )
        # The next slew departs from the tracked boresight at exposure end.
        prev_boresight = target.direction_at(start + exposure_s).unit()
        release_t = start + exposure_s
        last_start = start

    return ChainResult(tuple(results), chain_ok)
