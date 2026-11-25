"""
registry.py — the context package contract in code (P0).

A **context package** turns a Matrix-normalised match into named context fields on the
context grid (``grid.py``). It declares its name, version, dependencies and fields (each
with a unit, an observed or estimated tag and a definition), computes one column per field,
and contributes its own Data check entries. The registry resolves run order from
dependencies, runs the packages on one match, stamps every row with each package's version,
and attaches the Data check entries to the match's report.

See ``ponta-blueprint/context-packages.md`` for the contract this implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from matrix_engine.match import NormalisedMatch
from matrix_engine.parity import (
    MAX_ESTIMATED_SHARE,
    NEAR_BALL_M,
    AnalyticsParityReport,
    PackageEntry,
    with_package_entries,
)

from plaiq_engine.context.grid import GRID_COLUMNS, build_grid
from plaiq_engine.context.thresholds import THRESHOLDS_VERSION, values

Kind = Literal["observed", "estimated", "mixed"]


@dataclass(frozen=True)
class ContextField:
    """One output column. ``mixed`` means the row's source column says which rows are observed."""

    name: str
    unit: str
    kind: Kind
    definition: str


@dataclass
class ContextInput:
    """What a package receives: the match, the grid, threshold values, its dependencies' outputs."""

    match: NormalisedMatch
    grid: pd.DataFrame
    config: dict[str, float]
    upstream: dict[str, PackageOutput]
    extras: dict = field(default_factory=dict)


@dataclass
class PackageOutput:
    """
    A package's results. ``frames`` has one row per grid point (same order) and one column
    per declared field. ``segments`` holds segment-level tables (for example possession
    chains). ``measurements`` are figures for reports, never shown to the model.
    """

    frames: pd.DataFrame
    entries: list[PackageEntry] = field(default_factory=list)
    segments: dict[str, pd.DataFrame] = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)


#: Data check 1.1 in the context packages (review GC-01, 25 November 2026): where the provider flags estimated
#: positions (the schema's ``observed`` column), a Full entry built from player positions is Degraded when more than
#: MAX_ESTIMATED_SHARE of the positions it reads were estimated, as in the Matrix Data check. Entries about play near
#: the ball read the share within NEAR_BALL_M of it; the rest the whole pitch; entries not built from positions are kept.
NEAR_BALL_ENTRIES = frozenset({("p1", "possession"), ("p1", "possession_state"), ("p5", "relationships"), ("p4v", "engagements")})
NOT_POSITIONAL = frozenset({("p1", "game_state"), ("probe", "ball_detected")})


class _EstimatedShares:
    """The match's estimated-position shares (whole pitch, near the ball), computed once and only when asked."""

    def __init__(self, match: NormalisedMatch):
        self._m, self._wide, self._near, self._done = match, None, None, set()

    def estimated(self, near: bool) -> float | None:
        key = "near" if near else "wide"
        if key not in self._done:
            obs = self._m.observed_near_ball_share(NEAR_BALL_M) if near else self._m.observed_share()
            setattr(self, f"_{key}", None if obs is None else 1.0 - obs)
            self._done.add(key)
        return self._near if near else self._wide


def cap_for_estimated(entries: list[PackageEntry], shares: _EstimatedShares) -> list[PackageEntry]:
    """Data check 1.1 for context entries: a Full positional entry is Degraded above MAX_ESTIMATED_SHARE estimated."""
    import dataclasses

    out = []
    for e in entries:
        key = (e.package, e.metric)
        if e.state == "full" and key not in NOT_POSITIONAL:
            near = key in NEAR_BALL_ENTRIES
            est = shares.estimated(near)
            if est is not None and est > MAX_ESTIMATED_SHARE:
                where = f" within {NEAR_BALL_M:.0f} m of the ball" if near else ""
                e = dataclasses.replace(e, state="degraded", reason=f"{e.reason} {est:.0%} of player positions{where} were estimated by "
                                        f"the provider (off camera), above the Data check's {MAX_ESTIMATED_SHARE:.0%}.")
        out.append(e)
    return out


class ContextPackage(ABC):
    name: str
    version: str
    depends_on: tuple[str, ...] = ()
    fields: tuple[ContextField, ...] = ()

    @abstractmethod
    def compute(self, ctx: ContextInput) -> PackageOutput: ...

    def entry(self, metric: str, label: str, state: str, reason: str, **kw) -> PackageEntry:
        return PackageEntry(package=self.name, package_version=self.version, metric=metric, label=label,
                            state=state, reason=reason, **kw)


@dataclass
class ContextResult:
    """All packages' outputs for one match."""

    match_id: str
    frames: pd.DataFrame
    outputs: dict[str, PackageOutput]
    versions: dict[str, str]
    thresholds_version: str

    @property
    def entries(self) -> list[PackageEntry]:
        return [e for o in self.outputs.values() for e in o.entries]

    def attach_to(self, report: AnalyticsParityReport) -> AnalyticsParityReport:
        """The match's Data check with every package's entries added."""
        return with_package_entries(report, self.entries)

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Write ``context_frames.parquet`` and one Parquet file per segment table. Deterministic."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {"context_frames": out / "context_frames.parquet"}
        self.frames.to_parquet(paths["context_frames"], index=False)
        for pkg, o in sorted(self.outputs.items()):
            for name, seg in sorted(o.segments.items()):
                p = out / f"{pkg}_{name}.parquet"
                seg.to_parquet(p, index=False)
                paths[f"{pkg}_{name}"] = p
        return paths


class Registry:
    def __init__(self, packages: list[ContextPackage] | None = None) -> None:
        self._pkgs: dict[str, ContextPackage] = {}
        for p in packages or []:
            self.register(p)

    def register(self, pkg: ContextPackage) -> None:
        if pkg.name in self._pkgs:
            raise ValueError(f"package {pkg.name!r} registered twice")
        for f in pkg.fields:
            if not f.name.startswith(f"{pkg.name}_"):
                raise ValueError(f"field {f.name!r} must be prefixed with {pkg.name!r}_")
        self._pkgs[pkg.name] = pkg

    def order(self) -> list[str]:
        """Dependency order; ties broken by name, so the order is deterministic."""
        missing = {d for p in self._pkgs.values() for d in p.depends_on if d not in self._pkgs}
        if missing:
            raise ValueError(f"missing dependencies: {sorted(missing)}")
        indeg = {n: len(p.depends_on) for n, p in self._pkgs.items()}
        dependents: dict[str, list[str]] = {n: [] for n in self._pkgs}
        for n, p in self._pkgs.items():
            for d in p.depends_on:
                dependents[d].append(n)
        ready = sorted(n for n, k in indeg.items() if k == 0)
        out: list[str] = []
        while ready:
            n = ready.pop(0)
            out.append(n)
            for m in sorted(dependents[n]):
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
                    ready.sort()
        if len(out) != len(self._pkgs):
            raise ValueError(f"dependency cycle among: {sorted(set(self._pkgs) - set(out))}")
        return out

    def run(self, match: NormalisedMatch, *, config: dict[str, float] | None = None, extras: dict | None = None) -> ContextResult:
        cfg = config if config is not None else values()
        grid = build_grid(match, cfg["context_grid_rate_hz"])
        outputs: dict[str, PackageOutput] = {}
        frames = grid.copy()
        shares = _EstimatedShares(match)
        for name in self.order():
            pkg = self._pkgs[name]
            ctx = ContextInput(match=match, grid=grid, config=cfg,
                               upstream={d: outputs[d] for d in pkg.depends_on}, extras=dict(extras or {}))
            out = pkg.compute(ctx)
            declared = [f.name for f in pkg.fields]
            if list(out.frames.columns) != declared:
                raise ValueError(f"{name}: columns {list(out.frames.columns)} do not match declared fields {declared}")
            if len(out.frames) != len(grid):
                raise ValueError(f"{name}: {len(out.frames)} rows for a grid of {len(grid)}")
            out.entries = cap_for_estimated(out.entries, shares)  # before any package downstream reads them
            outputs[name] = out
            block = out.frames.reset_index(drop=True)
            block[f"{name}_version"] = pkg.version
            frames = pd.concat([frames, block], axis=1)
        assert list(frames.columns[: len(GRID_COLUMNS)]) == list(GRID_COLUMNS)
        return ContextResult(match_id=str(grid["match_id"].iloc[0]) if len(grid) else "", frames=frames, outputs=outputs,
                             versions={n: self._pkgs[n].version for n in self.order()}, thresholds_version=THRESHOLDS_VERSION)

    def fields(self) -> list[ContextField]:
        return [f for n in self.order() for f in self._pkgs[n].fields]
