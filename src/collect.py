"""Stage: collect.

Polls every unit the manifest says is generating and downloads the MP4 for any
that finished. Cheap and idempotent, so it is safe to run on a short cron: a
poll costs seconds, while the generation it is waiting on costs 20-30 minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BUILD, CliError, load_syllabus, log, nlm, read_manifest, record, unit_by_id,
)

TERMINAL_OK = {"completed", "complete", "succeeded", "success", "ready", "done"}
TERMINAL_BAD = {"failed", "error", "cancelled", "canceled"}


def status_of(payload: dict) -> str:
    for key in ("status", "state", "generation_status"):
        v = payload.get(key)
        if isinstance(v, str) and v:
            return v.lower()
    if payload.get("done") is True:
        return "completed"
    return "unknown"


def collect_unit(syl: dict, unit: dict, rec: dict, retry_failed: bool) -> str:
    subject_id = syl["subject"]["id"]
    nb, art, profile = rec.get("notebook_id"), rec.get("artifact_id"), rec.get("profile")
    if not (nb and art):
        return "skipped (not fired)"

    try:
        payload = nlm("artifact", "poll", art, "-n", nb, "--json", profile=profile)
    except CliError as e:
        log(f"poll failed for {unit['id']}: {e}")
        return "poll-error"

    st = status_of(payload)

    if st in TERMINAL_BAD:
        log(f"unit {unit['id']} generation failed upstream")
        if retry_failed:
            try:
                nlm("artifact", "retry", art, "-n", nb, "--json", profile=profile)
                record(subject_id, unit["id"], state="generating", error=None)
                return "retried"
            except CliError as e:
                record(subject_id, unit["id"], state="failed", error=str(e)[:500])
                return f"retry-refused ({'quota' if e.rate_limited else 'error'})"
        record(subject_id, unit["id"], state="failed")
        return "failed"

    if st not in TERMINAL_OK:
        return f"still {st}"

    outdir = BUILD / f"{subject_id}-{unit['id']}"
    outdir.mkdir(parents=True, exist_ok=True)
    mp4 = outdir / "raw.mp4"

    if mp4.exists() and mp4.stat().st_size > 0:
        record(subject_id, unit["id"], state="downloaded", raw_mp4=str(mp4))
        return "already downloaded"

    log(f"downloading unit {unit['id']} -> {mp4}")
    nlm("download", "video", art, "-n", nb, "-o", str(mp4),
        profile=profile, parse_json=False, timeout=1800)

    if not mp4.exists() or mp4.stat().st_size == 0:
        record(subject_id, unit["id"], state="failed", error="download produced no file")
        return "download-empty"

    size_mb = mp4.stat().st_size / 1e6
    record(subject_id, unit["id"], state="downloaded", raw_mp4=str(mp4),
           raw_size_mb=round(size_mb, 2))
    return f"downloaded ({size_mb:.1f} MB)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Poll and download finished video overviews.")
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--unit", action="append", default=[])
    ap.add_argument("--retry-failed", action="store_true",
                    help="issue an in-place retry for upstream failures")
    a = ap.parse_args()

    syl = load_syllabus(a.syllabus)
    subject_id = syl["subject"]["id"]
    manifest = read_manifest()

    units = [unit_by_id(syl, u) for u in a.unit] if a.unit else syl["units"]
    pending = 0
    for u in units:
        rec = manifest["units"].get(f"{subject_id}/{u['id']}", {})
        if rec.get("state") in ("postprocessed", "published"):
            continue
        result = collect_unit(syl, u, rec, a.retry_failed)
        log(f"unit {u['n']} ({u['id']}): {result}")
        if result.startswith("still") or result == "retried":
            pending += 1

    log(f"{pending} unit(s) still generating")


if __name__ == "__main__":
    main()
