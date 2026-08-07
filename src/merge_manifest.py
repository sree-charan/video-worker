#!/usr/bin/env python3
"""Merge two manifest versions instead of picking one.

Runs are per-unit concurrent, so five of them commit state/manifest.json at once.
The commit step used to resolve conflicts with `git checkout --theirs`, which
keeps one run's file and silently throws away every record the other runs wrote -
that is how a fired unit vanished from the manifest while its video generated
happily upstream.

Manifests are a dict keyed by "subject/unit", so they merge cleanly: take the
union of keys, and for a key present in both, keep the record with the newer
`updated_at`. Field-level merge is deliberately avoided; a whole record from one
run is always self-consistent, whereas a mixture of two might not be.

Usage:  merge_manifest.py OURS THEIRS OUT
"""

from __future__ import annotations

import json
import sys


def load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "units": {}}
    data.setdefault("version", 1)
    data.setdefault("units", {})
    return data


def newer(a: dict, b: dict) -> dict:
    """Whichever record was written last; ties keep the more complete one."""
    ta, tb = a.get("updated_at", ""), b.get("updated_at", "")
    if ta != tb:
        return a if ta > tb else b
    return a if len(a) >= len(b) else b


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    ours, theirs, out = sys.argv[1:4]
    a, b = load(ours), load(theirs)

    units = dict(a["units"])
    for key, rec in b["units"].items():
        units[key] = newer(units[key], rec) if key in units else rec

    merged = {"version": max(a["version"], b["version"]), "units": units}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True)
        fh.write("\n")

    only_a = set(a["units"]) - set(b["units"])
    only_b = set(b["units"]) - set(a["units"])
    print(f"merged {len(units)} unit(s): "
          f"{len(only_a)} only in ours, {len(only_b)} only in theirs, "
          f"{len(set(a['units']) & set(b['units']))} in both")


if __name__ == "__main__":
    main()
