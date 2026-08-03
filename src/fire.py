"""Stage: fire.

Per unit: notebook, source, three scrutinised chat rounds, then start the video
and exit. Never blocks on generation - see README for why.

Round results are cached in the manifest, so a re-run after a failure resumes at
the round that failed instead of re-spending chat quota.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plan as P  # noqa: E402
from common import (  # noqa: E402
    BUILD, CliError, dig, get_record, load_syllabus, log, nlm_retry, record,
    unit_by_id, unit_key,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "its", "it",
    "definition", "introduction", "overview", "basic", "basics", "using", "use",
    "uses", "concepts", "concept", "types", "type", "understanding", "creating",
    "accessing", "defining", "implementing", "preventing", "between", "vs",
}


def find_notebook_by_title(title: str, profile: str | None) -> str | None:
    """Look for an existing notebook with this exact title.

    Guards against duplicate notebooks when a run failed after `create` but
    before the manifest was written - which is precisely what happened on the
    first real run.
    """
    try:
        payload = nlm_retry("list", "--json", profile=profile)
    except CliError:
        return None
    items = payload if isinstance(payload, list) else (
        payload.get("notebooks") or payload.get("items") or [])
    for nb in items:
        if isinstance(nb, dict) and (nb.get("title") or "").strip() == title.strip():
            return nb.get("id") or nb.get("notebook_id")
    return None


def ensure_notebook(syl: dict, unit: dict, profile: str | None) -> str:
    sid = syl["subject"]["id"]
    rec = get_record(sid, unit["id"])
    if rec.get("notebook_id"):
        log(f"reusing notebook {rec['notebook_id']}")
        return rec["notebook_id"]

    title = f"{syl['subject']['title']} - Unit {unit['n']}: {unit['title']}"

    existing = find_notebook_by_title(title, profile)
    if existing:
        log(f"adopting existing notebook {existing} (same title)")
        record(sid, unit["id"], notebook_id=existing, notebook_title=title,
               profile=profile, state="notebook")
        return existing

    out = nlm_retry("create", title, "--json", profile=profile)
    nb = dig(out, "id", "notebook_id")
    # Record before validating: if the id shape is ever unexpected again, the
    # notebook still exists upstream and must not be orphaned.
    if nb:
        record(sid, unit["id"], notebook_id=nb, notebook_title=title,
               profile=profile, state="notebook")
        log(f"created notebook {nb}")
        return nb
    raise SystemExit(f"create returned no id: {out}")


def ensure_source(syl: dict, unit: dict, nb: str, profile: str | None) -> None:
    sid = syl["subject"]["id"]
    if get_record(sid, unit["id"]).get("source_added"):
        return
    subj = syl["subject"]
    nlm_retry("source", "add-drive", subj["source_drive_id"], subj["source_title"],
              "--mime-type", "pdf", "-n", nb, "--json", profile=profile)
    log("attached course file by Drive reference")
    record(sid, unit["id"], source_added=True, state="sourced")


def ask(prompt: str, nb: str, profile: str | None, label: str,
        outdir: Path, slot: str) -> str:
    """One chat round, with the prompt passed as a file.

    Round 3 embeds rounds 1 and 2, so the prompt runs to several thousand
    characters. Passing that as an argv element is fragile and makes any failure
    unreadable in CI logs; --prompt-file avoids both. The prompt is also kept on
    disk so a failed round can be inspected in the workflow artifact.
    """
    pf = outdir / f"prompt-{slot}.txt"
    pf.write_text(prompt, encoding="utf-8")
    log(f"chat round: {label} ({len(prompt)} chars -> {pf.name})")
    if len(prompt) > P.MAX_PROMPT_CHARS:
        log(f"  WARNING prompt exceeds {P.MAX_PROMPT_CHARS} chars; the chat "
            "endpoint rejects over-long questions")
    out = nlm_retry("ask", "--prompt-file", str(pf), "-n", nb, "--json",
                    profile=profile, timeout=900)
    text = dig(out, "answer", "response", "text", "content") or ""
    if not text.strip():
        raise SystemExit(f"round '{label}' returned an empty answer")
    (outdir / f"answer-{slot}.txt").write_text(text, encoding="utf-8")
    return text


def fill_gaps(syl: dict, unit: dict, nb: str, profile: str | None,
              gaps: list[str], deep: bool) -> None:
    """The book does not cover something the syllabus demands.

    Rather than let the video invent it or skip it, pull real sources in with
    Deep Research and let the later rounds ground on those instead.
    """
    q = P.research_prompt(syl, unit, gaps)
    log(f"{len(gaps)} gap(s) in the course file; running "
        f"{'deep' if deep else 'fast'} research to fill them")
    nlm_retry("source", "add-research", q,
              "--from", "web", "--mode", "deep" if deep else "fast",
              "--import-all", "--cited-only", "-n", nb,
              "--timeout", "1800", "--json", profile=profile, timeout=3900)
    record(syl["subject"]["id"], unit["id"], gaps_filled=gaps)


def build_spec(syl: dict, unit: dict, nb: str, profile: str | None,
               minutes: int, deep_research: bool, outdir: Path) -> dict:
    """Rounds 1-3. Each round's raw text is cached in the manifest."""
    sid = syl["subject"]["id"]
    rec = get_record(sid, unit["id"])

    # ---- round 1: verbatim structure -------------------------------------
    r1_raw = rec.get("round1")
    if not r1_raw:
        r1_raw = ask(P.round1_prompt(syl, unit), nb, profile, "1/3 structure",
                     outdir, "1-structure")
        record(sid, unit["id"], round1=r1_raw)
    r1 = P.parse_round1(r1_raw)

    # One corrective re-ask if the model echoed syllabus lines back as headings
    # or returned too few sections. Costs one chat out of 500/day and prevents a
    # 12 minute video being spread across three headings.
    problem = P.audit_round1(r1)
    if problem and not rec.get("round1_retried"):
        log(f"  round 1 rejected: {problem.splitlines()[0]}")
        retry_raw = ask(P.round1_retry_prompt(unit, problem), nb, profile,
                        "1/3 structure (corrective re-ask)", outdir, "1-retry")
        retry = P.parse_round1(retry_raw)
        # Only accept the retry if it is actually better.
        if len(retry["sections"]) > len(r1["sections"]) or not P.audit_round1(retry):
            r1_raw, r1 = retry_raw, retry
            record(sid, unit["id"], round1=r1_raw)
        else:
            log("  corrective re-ask did not improve; keeping the first answer")
        record(sid, unit["id"], round1_retried=True)

    log(f"  {len(r1['sections'])} sections, {len(r1['terms'])} locked terms, "
        f"notes_present={r1['notes_present']}, {len(r1['missing'])} gap(s)")
    for h in r1["sections"]:
        log(f"    - {h}")
    if not r1["sections"]:
        raise SystemExit("round 1 produced no sections; inspect manifest 'round1'")
    residual = P.audit_round1(r1)
    if residual:
        log("  WARNING section granularity still weak after re-ask; "
            "the video may spend too long on single headings")
    record(sid, unit["id"], section_warning=bool(residual))

    # ---- optional: repair a thin course file ------------------------------
    needs_research = (not r1["notes_present"]) or len(r1["missing"]) >= 2
    if needs_research and not rec.get("gaps_filled"):
        targets = r1["missing"] or [unit["title"]]
        try:
            fill_gaps(syl, unit, nb, profile, targets, deep_research)
            # Structure may improve once real sources exist, so re-ask round 1.
            r1_raw = ask(P.round1_prompt(syl, unit), nb, profile,
                         "1/3 structure (post-research)", outdir, "1-postresearch")
            record(sid, unit["id"], round1=r1_raw)
            r1 = P.parse_round1(r1_raw)
            log(f"  after research: {len(r1['sections'])} sections, "
                f"{len(r1['missing'])} gap(s) remain")
        except CliError as e:
            log(f"  research unavailable ({'quota' if e.rate_limited else 'error'}); "
                "continuing with the book as-is")

    # ---- round 2: substance ----------------------------------------------
    r2_raw = rec.get("round2")
    if not r2_raw:
        r2_raw = ask(P.round2_prompt(unit, r1["sections"], unit["example"]),
                     nb, profile, "2/3 substance", outdir, "2-substance")
        record(sid, unit["id"], round2=r2_raw)

    # ---- round 3: audit + time budget ------------------------------------
    r3_raw = rec.get("round3")
    if not r3_raw:
        # Round 3 relies on this conversation already containing rounds 1-2:
        # re-sending them pushed the question past the server's size limit.
        r3_raw = ask(P.round3_prompt(unit, r1["sections"], minutes),
                     nb, profile, "3/3 scrutiny and time budget", outdir, "3-scrutiny")
        record(sid, unit["id"], round3=r3_raw)

    spec = P.parse_round3(r3_raw, r1["sections"], minutes * 60)
    spec["terms"] = r1["terms"][:24]

    log(f"  final spec: {len(spec['sections'])} sections, "
        f"{spec['budgeted']}s budgeted of {minutes * 60}s target")
    if spec["gaps"]:
        log(f"  WARNING still uncovered after audit: {'; '.join(spec['gaps'])}")

    # Verify the lock actually held: every final heading must have come from
    # round 1 verbatim. A reworded heading is the exact failure we are guarding
    # against, so surface it loudly rather than shipping a renamed chapter.
    known = {h.lower() for h in r1["sections"]}
    drifted = [s["heading"] for s in spec["sections"] if s["heading"].lower() not in known]
    if drifted:
        log(f"  WARNING round 3 reworded {len(drifted)} heading(s): {drifted}")
    record(sid, unit["id"], heading_drift=drifted or None)

    return spec


def anchor_for(heading: str) -> str:
    """Transcript search phrase for a heading.

    Prefers a technical token (ALLCAPS or CamelCase, e.g. CLASSPATH, ArrayList,
    JDBC) because those are spoken distinctively and rarely appear by accident.
    Otherwise falls back to the longest content word. Plain longest-word alone
    is wrong: "understanding CLASSPATH" would anchor on "understanding".
    """
    words = re.findall(r"[A-Za-z][\w'-]*", heading)
    content = [w for w in words if w.lower() not in STOPWORDS]
    if not content:
        content = words or [heading]
    technical = [w for w in content
                 if w.isupper() and len(w) > 2 or re.match(r"^[A-Z][a-z]+[A-Z]", w)]
    pick = max(technical, key=len) if technical else max(content, key=len)
    return pick.lower()


def fire(syl: dict, unit: dict, profile: str | None, minutes: int, style: str,
         deep_research: bool, dry_run: bool) -> None:
    sid = syl["subject"]["id"]
    rec = get_record(sid, unit["id"])
    if rec.get("artifact_id") and rec.get("state") != "failed":
        log(f"unit {unit['id']} already fired (artifact {rec['artifact_id']}); skipping")
        return

    outdir = BUILD / unit_key(syl, unit)
    outdir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        # No quota spent. Round 1 has not run, so headings here are approximated
        # from the syllabus line at roughly the granularity round 1 is asked for.
        # Real headings come from the book and will be shorter and more specific.
        raw = P.clean(unit["topics"]).replace(";", ".")
        headings: list[str] = []
        for seg in raw.split("."):
            seg = seg.strip()
            if not seg:
                continue
            headings += ([q.strip() for q in seg.split(",") if q.strip()]
                         if len(seg.split()) > 12 else [seg])
        headings = [h for h in headings if h][:P.MAX_SECTIONS]
        per = max(minutes * 60 // max(len(headings), 1), 25)
        spec = {"sections": [{"k": str(i + 1), "heading": h, "points": [],
                              "spec": "", "step": "", "seconds": per}
                             for i, h in enumerate(headings)],
                "terms": [], "gaps": [], "budgeted": minutes * 60}
        print(P.video_prompt(syl, unit, spec, minutes))
        return

    nb = ensure_notebook(syl, unit, profile)
    ensure_source(syl, unit, nb, profile)
    spec = build_spec(syl, unit, nb, profile, minutes, deep_research, outdir)

    prompt = P.video_prompt(syl, unit, spec, minutes)
    pf = outdir / "video-prompt.txt"
    pf.write_text(prompt, encoding="utf-8")
    (outdir / "spec.json").write_text(
        __import__("json").dumps(spec, indent=2), encoding="utf-8")
    log(f"steering prompt written to {pf} ({len(prompt)} chars)")

    labels = P.chapter_labels(spec)
    out = nlm_retry(
        "generate", "video",
        "--prompt-file", str(pf),
        "--format", "explainer",      # 16:9; 'short' is vertical, cinematic is 2/day
        "--style", style,
        "--language", "en",
        "-n", nb, "--no-wait", "--json",
        profile=profile,
    )
    task = dig(out, "task_id", "artifact_id", "id")
    if not task:
        raise SystemExit(f"generate video returned no task id: {out}")

    log(f"generation started: task {task}")
    record(sid, unit["id"], artifact_id=task, state="generating", style=style,
           chapter_labels=labels,
           chapter_anchors=[anchor_for(h) for h in labels],
           section_seconds=[s["seconds"] for s in spec["sections"]],
           prompt_chars=len(prompt), target_minutes=minutes)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan and start video generation.")
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--unit", action="append", default=[])
    ap.add_argument("--profile", default=None)
    ap.add_argument("--minutes", type=int, default=12)
    ap.add_argument("--style", default="classic")
    ap.add_argument("--deep-research", action="store_true",
                    help="use deep (not fast) research when filling course-file gaps")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    syl = load_syllabus(a.syllabus)
    units = [unit_by_id(syl, u) for u in a.unit] if a.unit else syl["units"]

    failures = 0
    for u in units:
        log(f"=== unit {u['n']} ({u['id']}): {u['title']}")
        try:
            fire(syl, u, a.profile, a.minutes, a.style, a.deep_research, a.dry_run)
        except (CliError, SystemExit) as e:
            failures += 1
            log(f"FAILED unit {u['id']}: {e}")
            record(syl["subject"]["id"], u["id"], state="failed", error=str(e)[:500])
    if failures:
        raise SystemExit(f"{failures} unit(s) failed")


if __name__ == "__main__":
    main()
