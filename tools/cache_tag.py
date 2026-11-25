"""
cache_tag.py — the version tag in every measurement cache key (review DE-02, SE-07, 25 November 2026).

A cached match or table is reused only when the engines (and, for a reference reader, the reader) are the versions
that wrote it: the tag is part of the file name, so a version bump makes every older cache a miss and the tool
rebuilds it rather than silently reading output from older code.
"""

from __future__ import annotations

from importlib import metadata


def _v(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def tag(reader: str = "", reader_version: str = "") -> str:
    """``m<matrix version>-p<plaiq version>[-<reader><version>]``, for cache file names."""
    parts = [f"m{_v('ponta-matrix-engine')}", f"p{_v('ponta-plaiq-engine')}"]
    if reader:
        parts.append(f"{reader}{reader_version}")
    return "-".join(parts)
