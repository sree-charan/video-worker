"""Stage: fire.

Per unit: create a notebook, attach the course file by reference, spend one chat
to extract the beat sheet, then start video generation and exit immediately.

Deliberately does NOT wait for the video. Generation takes 20-30+ minutes, and
blocking a runner for that would cost ~1800 Actions-minutes/day at full rate.
`collect.py` picks the artifact up later.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan as P  # noqa: E402
from common import (  # noqa: E402
    BUILD, CliError, get_record, load_syllabus, log, nlm, nlm_retry, record,
    unit_by_id, unit_key,
)


def ensure_notebook(syl: dict, unit: dict, profile: str | None) -> str:
    """Reuse the notebook from a previous run if the manifest has one."""
    subject_id = syl["subject"]["id"]
    rec = get_record(subject_id, unit["id"])
    if rec.get("notebook_id"):
        log(f"reusing notebook {rec['notebook_id']}")
        return rec["notebook_id"]

    title = f"{syl['subject']['title']} - Unit {unit['n']}: {unit['title']}"
    out = nlm_retry("create", title, "--json", profile=profile)
    nb = out.get("id") or out.get("notebook_id")
    if not nb:
        raise SystemExit(f"create returned no id: {out}")
    log(f"created notebook {nb}")
    record(subject_id, unit["id"], notebook_id=nb, notebook_title=title,
           profile=profile, state="notebook")
    return nb


def ensure_source(syl: dict, unit: dict, nb: str, profile: str | None) -> None:
    subject_id = syl["subject"]["id"]
    if get_record(subject_id, unit["id"]).get("source_added"):
        return
    subj = syl["subject"]
    # A Drive PDF goes in by reference - the file never leaves Google.
    nlm_retry("source", "add-drive", subj["source_drive_id"], subj["source_title"],
              "--mime-type", "pdf", "-n", nb, "--json", profile=profile)
    log("attached course file by Drive reference")
    record(subject_id, unit["id"], source_added=True, state="sourced")


def build_beats(syl: dict, unit: dict, nb: str, profile: str | None) -> tuple[str, list[str]]:
    """Spend one cheap chat to ground the expensive video prompt."""
    subject_id = syl["subject"]["id"]
    rec = get_record(subject_id, unit["id"])
    outline = rec.get("outline")

    if not outline:
        prompt = P.outline_prompt(syl, unit)
        log("asking notebook for the unit beat sheet")
        out = nlm_retry("ask", prompt, "-n", nb, "--json", profile=profile, timeout=600)
        outline = out.get("answer") or out.get("response") or out.get("text") or ""
        if not outline.strip():
            log("WARNING outline came back empty; falling back to raw syllabus scope")
        record(subject_id, unit["id"], outline=outline, state="outlined")

    beats = P.beats_from_outline(outline, unit["topics"])
    return beats, P.chapter_labels(outline, unit)


def fire(syl: dict, unit: dict, profile: str | None, minutes: int,
         style: str, dry_run: bool) -> None:
    subject_id = syl["subject"]["id"]
    rec = get_record(subject_id, unit["id"])
    if rec.get("artifact_id") and rec.get("state") not in ("failed",):
        log(f"unit {unit['id']} already fired (artifact {rec['artifact_id']}); skipping")
        return

    nb = None if dry_run else ensure_notebook(syl, unit, profile)
    if not dry_run:
        ensure_source(syl, unit, nb, profile)
        beats, labels = build_beats(syl, unit, nb, profile)
    else:
        beats, labels = P._bullets(unit["topics"]), P.chapter_labels("", unit)

    prompt = P.video_prompt(syl, unit, beats, minutes=minutes)

    outdir = BUILD / unit_key(syl, unit)
    outdir.mkdir(parents=True, exist_ok=True)
    pf = outdir / "video-prompt.txt"
    pf.write_text(prompt, encoding="utf-8")
    log(f"steering prompt written to {pf} ({len(prompt)} chars)")

    if dry_run:
        print(prompt)
        return

    # explainer = 16:9 horizontal and the only format with a workable daily cap
    # (cinematic is 2/day on Pro, and 'short' is vertical).
    out = nlm_retry(
        "generate", "video",
        "--prompt-file", str(pf),
        "--format", "explainer",
        "--style", style,
        "--language", "en",
        "-n", nb,
        "--no-wait",
        "--json",
        profile=profile,
    )
    task = out.get("task_id") or out.get("id") or out.get("artifact_id")
    if not task:
        raise SystemExit(f"generate video returned no task id: {out}")

    log(f"generation started: task {task}")
    record(subject_id, unit["id"], artifact_id=task, state="generating",
           chapter_labels=labels, prompt_chars=len(prompt), style=style)


def main() -> None:
    ap = argparse.ArgumentParser(description="Start video generation for one or more units.")
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--unit", action="append", default=[],
                    help="unit id or number; repeat for several. Default: all.")
    ap.add_argument("--profile", default=None, help="notebooklm profile (= which Google account)")
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--style", default="classic")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and print the prompt without spending quota")
    a = ap.parse_args()

    syl = load_syllabus(a.syllabus)
    units = [unit_by_id(syl, u) for u in a.unit] if a.unit else syl["units"]

    failures = 0
    for u in units:
        log(f"=== unit {u['n']} ({u['id']}): {u['title']}")
        try:
            fire(syl, u, a.profile, a.minutes, a.style, a.dry_run)
        except CliError as e:
            failures += 1
            log(f"FAILED unit {u['id']}: {e}")
            record(syl["subject"]["id"], u["id"], state="failed", error=str(e)[:500])
    if failures:
        raise SystemExit(f"{failures} unit(s) failed to start")


if __name__ == "__main__":
    main()
