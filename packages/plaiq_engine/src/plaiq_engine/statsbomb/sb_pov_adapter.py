"""
statsbomb/sb_pov_adapter.py — Thin adapter bridging StatsBomb freeze-frame
players to the defender_data format expected by the analytics engines.

``player_pov()`` and ``targeted_pcz()`` in analytics/pass_option_value.py
accept a generic ``defender_data`` list:

    [{"pos": (2,), "vel": (2,), "v_max": float, "a_max": float,
      "reaction_time": float}, ...]

The tracking-data version of this uses _build_defender_data() to read
velocities from tracking row columns.  For StatsBomb we have no velocities, so:

  - vel = [0, 0] for all players (static snapshot assumption).
  - If the stitch graph has produced a displacement/dt velocity estimate
    (passed as "vel" inside the FF player dict), that is used instead.

Neither ``player_pov`` nor ``targeted_pcz`` are modified — this adapter is
the only glue between the SB pipeline and the analytics engines.
"""
from __future__ import annotations

import numpy as np

from plaiq_engine.analytics.pitch_control import DEFAULT_PROFILE

DEFAULT_ADAPTER_PROFILE = DEFAULT_PROFILE.copy()


def sb_defender_data(
    ff_players: list[dict],
    side: str,
    profile: dict | None = None,
) -> list[dict]:
    """
    Convert identified FF players into the defender_data format expected by
    ``player_pov()`` / ``targeted_pcz()``.

    Parameters
    ----------
    ff_players : list of FF player dicts (may be raw or identity-enriched)
                 Each: {"teammate": bool, "actor": bool, "keeper": bool,
                        "location": [x, y],
                        optional "vel": [vx, vy]}
    side       : "teammate" — return team-mates of the actor (same side)
                 "opponent" — return opponents
    profile    : optional physical profile override
                 {"v_max": float, "a_max": float, "reaction_time": float}

    Returns
    -------
    list of defender_data dicts: [{pos, vel, v_max, a_max, reaction_time}]
    """
    prof = profile if profile is not None else DEFAULT_ADAPTER_PROFILE

    if side == "teammate":
        subset = [p for p in ff_players if p.get("teammate") and not p.get("actor")]
    elif side == "opponent":
        subset = [p for p in ff_players if not p.get("teammate") and not p.get("actor")]
    else:
        raise ValueError(f"side must be 'teammate' or 'opponent', got {side!r}")

    result = []
    for p in subset:
        loc = p.get("location", [0.0, 0.0])
        vel = p.get("vel", [0.0, 0.0])
        result.append({
            "pos":           np.array([float(loc[0]), float(loc[1])]),
            "vel":           np.array([float(vel[0]), float(vel[1])]),
            "v_max":         float(prof.get("v_max",         DEFAULT_PROFILE["v_max"])),
            "a_max":         float(prof.get("a_max",         DEFAULT_PROFILE["a_max"])),
            "reaction_time": float(prof.get("reaction_time", DEFAULT_PROFILE["reaction_time"])),
        })
    return result


def sb_all_teammates(
    ff_players: list[dict],
    ball_xy: list[float] | np.ndarray,
    xT_surfaces: tuple,
    hex_centers: np.ndarray,
    lineup: dict | None = None,
) -> list[dict]:
    """
    Build a list of {name, xy, pov} for every visible teammate, ready for
    finding max_POV at decision time.

    Imports ``player_pov`` locally to avoid a circular-import at module load.

    Parameters
    ----------
    ff_players    : list of FF player dicts (identity-enriched preferred)
    ball_xy       : [x, y] in PLAIQ metres
    xT_surfaces   : (xT_home, xT_away) — from hex_value.xT_surface()
    hex_centers   : (N, 2) array of hex centre coords — from pitch_control
    lineup        : optional {"home": [...], "away": [...]} for fallback names

    Returns
    -------
    list of {"name": str, "xy": ndarray, "pov": float}
    """
    from plaiq_engine.analytics.pass_option_value import player_pov  # local import

    xT_home, _xT_away = xT_surfaces
    ball_np = np.array([float(ball_xy[0]), float(ball_xy[1])])

    # xT at the ball position
    hex_idx = _nearest_hex(ball_np, hex_centers)
    xT_ball = float(xT_home[hex_idx]) if hex_idx is not None else 0.0

    defenders_opp = sb_defender_data(ff_players, "opponent")

    teammates = [p for p in ff_players if p.get("teammate") and not p.get("actor")]

    result = []
    for tm in teammates:
        loc = tm.get("location", [0.0, 0.0])
        recv_np = np.array([float(loc[0]), float(loc[1])])

        # xT at receiver position
        ridx = _nearest_hex(recv_np, hex_centers)
        xT_recv = float(xT_home[ridx]) if ridx is not None else 0.0

        try:
            pov_val = player_pov(
                ball_np, recv_np,
                xT_ball, xT_recv,
                defenders_opp,
            )
        except Exception:
            pov_val = 0.0

        name = tm.get("player_name") or f"Tm#{tm.get('jersey', '?')}"
        result.append({
            "name": name,
            "xy":   recv_np,
            "pov":  float(pov_val),
        })

    return result


def sb_actor_info(
    ff_players: list[dict],
    event: dict,
) -> dict:
    """
    Extract actor position and name from the FF or event dict.

    Returns
    -------
    {"name": str | None, "xy": ndarray | None}
    """
    actor_dot = next(
        (p for p in ff_players if p.get("actor")),
        None,
    )
    name = None
    xy   = None

    if actor_dot is not None:
        loc = actor_dot.get("location")
        if loc:
            xy = np.array([float(loc[0]), float(loc[1])])
        name = actor_dot.get("player_name") or event.get("From")
    else:
        start_x = event.get("Start X")
        start_y = event.get("Start Y")
        if start_x is not None and start_y is not None:
            xy = np.array([float(start_x), float(start_y)])
        name = event.get("From")

    return {"name": name, "xy": xy}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nearest_hex(xy: np.ndarray, hex_centers: np.ndarray) -> int | None:
    if hex_centers is None or len(hex_centers) == 0:
        return None
    dists = np.hypot(hex_centers[:, 0] - xy[0], hex_centers[:, 1] - xy[1])
    return int(np.argmin(dists))
