"""Stage: postprocess.

One ffmpeg pass to swap the NotebookLM watermark for the GCTC logo in place,
then transcription for captions and real chapter timestamps, then a thumbnail
and the catalog JSON entry the Blogger page expects.

Chapters are derived from the transcript, never invented. NotebookLM returns no
timing information, so the only honest source of a chapter time is the moment
the concept is actually spoken. We locate each beat's anchor phrase in the
transcript and use its first occurrence.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import watermark  # noqa: E402
from common import BUILD, ROOT, get_record, load_syllabus, log, record, unit_by_id  # noqa: E402

LOGO_CFG = ROOT / "config" / "logo.json"

# Words too common in any lecture to localise a chapter.
STOPWORDS_NORM = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "its", "it",
    "is", "are", "this", "that", "on", "as", "by", "be", "we", "you",
}


# ------------------------------------------------------------------ ffmpeg

def run(cmd: list[str], timeout: int = 3600) -> str:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed:\n{p.stderr[-2000:]}")
    return p.stdout


def probe(mp4: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(mp4)])
    data = json.loads(out)
    v = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    return {
        "duration_sec": int(float(data["format"]["duration"])),
        "width": v.get("width"),
        "height": v.get("height"),
    }


def resolve_box(cfg: dict, width: int, height: int) -> dict:
    """Normalised config -> integer pixel box for this frame size.

    The box is stored as fractions because explainer output is not one fixed
    resolution: the sample frames supplied were 1600x900 while the real render
    came back 1280x720. Hardcoded pixels would have missed the mark entirely.
    """
    x = int(round(cfg["x"] * width))
    y = int(round(cfg["y"] * height))
    w = int(round(cfg["w"] * width))
    h = int(round(cfg["h"] * height))
    w = max(2, min(w - w % 2, width - x))
    h = max(2, min(h - h % 2, height - y))
    return {"x": x, "y": y, "w": w, "h": h}


def _cover_chain(tag_in: str, tag_out: str, logo_tag: str, box: dict,
                 segs: list[dict], align: str, window=None) -> list[str]:
    """Per-segment plate matching the background, then the logo inside the box.

    A saturated segment gets the matching plate plus a small inset white card,
    so the seam still vanishes while the logo stays legible.
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    inset = max(int(h * 0.12), 2)
    steps: list[str] = []
    cur = tag_in
    for i, s in enumerate(segs):
        gate = f"enable='between(t,{s['start']:.2f},{s['end']:.2f})'"
        nxt = f"{tag_out}b{i}"
        steps.append(f"[{cur}]drawbox=x={x}:y={y}:w={w}:h={h}:"
                     f"color={s['plate']}@1:t=fill:{gate}[{nxt}]")
        cur = nxt
        if s.get("card"):
            nxt = f"{tag_out}c{i}"
            steps.append(f"[{cur}]drawbox=x={x + inset}:y={y + inset}:"
                         f"w={w - 2 * inset}:h={h - 2 * inset}:"
                         f"color=white@1:t=fill:{gate}[{nxt}]")
            cur = nxt
    ox = f"{x}+{w}-overlay_w-{inset}" if align == "right" else f"{x}+({w}-overlay_w)/2"
    oy = f"{y}+({h}-overlay_h)/2"
    enable = (f":enable='between(t,{window[0]:.2f},{window[1]:.2f})'" if window else "")
    steps.append(f"[{cur}][{logo_tag}]overlay={ox}:{oy}:format=auto{enable}[{tag_out}]")
    return steps


def swap_logo(src: Path, dst: Path, logo: Path, cfg: dict, meta: dict,
              ff: str = "ffmpeg") -> dict:
    """Replace both NotebookLM marks, matching the background behind each.

    `delogo` is deliberately not used: it blurs a smeared patch rather than
    replacing the mark, which looks cheap. Frame geometry is never changed.
    """
    W, H = meta["width"], meta["height"]
    report: dict = {}

    br = resolve_box(cfg["bottom_right"], W, H)
    br_segs = watermark.segments(watermark.region_series(ff, str(src), br, W, H))
    report["bottom_right_segments"] = [
        {**s, "start": round(s["start"], 1), "end": round(s["end"], 1)} for s in br_segs]

    # Logo inset slightly so it sits inside the card when one is drawn.
    lw = max(br["w"] - 2 * max(int(br["h"] * 0.12), 2), 8)
    lh = max(br["h"] - 2 * max(int(br["h"] * 0.12), 2), 6)
    filters = [f"[1:v]scale={lw}:{lh}:force_original_aspect_ratio=decrease[lgbr]"]
    filters += _cover_chain("0:v", "v1", "lgbr", br, br_segs,
                            cfg["bottom_right"].get("align", "right"))
    last = "v1"

    # Title-card mark: present only for the opening seconds, and how many is not
    # fixed, so the window is measured rather than assumed.
    tc_cfg = cfg.get("centre_top")
    if tc_cfg:
        tc = resolve_box(tc_cfg, W, H)
        search = float(tc_cfg.get("search_seconds", 25))
        tc_bg = watermark.region_series(ff, str(src), tc, W, H, limit=search)
        window = watermark.detect_presence(
            watermark.darkness_series(ff, str(src), tc, limit=search),
            tc_bg, limit=search)
        report["centre_top_window"] = window
        if window:
            tc_segs = watermark.segments(tc_bg, window=window)
            report["centre_top_segments"] = [
                {**s, "start": round(s["start"], 1), "end": round(s["end"], 1)}
                for s in tc_segs]
            filters.append(
                f"[1:v]scale={tc['w']}:{tc['h']}:force_original_aspect_ratio=decrease[lgtc]")
            filters += _cover_chain(last, "v2", "lgtc", tc, tc_segs,
                                    tc_cfg.get("align", "centre"), window=window)
            last = "v2"

    run([ff, "-y", "-i", str(src), "-i", str(logo),
         "-filter_complex", ";".join(filters), "-map", f"[{last}]", "-map", "0:a?",
         "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "copy", str(dst)])
    return report


def thumbnail(mp4: Path, dst: Path, at: float) -> None:
    run(["ffmpeg", "-y", "-ss", f"{at:.2f}", "-i", str(mp4),
         "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3", str(dst)])


def sample_frames(mp4: Path, outdir: Path, count: int = 4) -> list[Path]:
    """Dump frames so the watermark box can be measured by eye, once."""
    outdir.mkdir(parents=True, exist_ok=True)
    dur = probe(mp4)["duration_sec"]
    shots = []
    for i in range(count):
        t = dur * (i + 1) / (count + 1)
        p = outdir / f"frame-{i + 1}.png"
        run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1", str(p)])
        shots.append(p)
    return shots


# ------------------------------------------------------------ transcription

def transcribe(mp4: Path, outdir: Path, model: str = "small") -> list[dict]:
    """faster-whisper -> word-ish segments. CPU int8 is fine for 10 minutes."""
    from faster_whisper import WhisperModel

    log(f"transcribing with faster-whisper ({model}, cpu/int8)")
    wm = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _info = wm.transcribe(str(mp4), language="en", vad_filter=True)

    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    (outdir / "transcript.json").write_text(json.dumps(segs, indent=2), encoding="utf-8")
    return segs


def write_vtt(segs: list[dict], dst: Path) -> None:
    def ts(t: float) -> str:
        h, rem = divmod(max(t, 0.0), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    lines = ["WEBVTT", ""]
    for i, s in enumerate(segs, 1):
        lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", s["text"], ""]
    dst.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------- chapters

def _norm(text: str) -> str:
    """Lowercase, punctuation to spaces, whitespace collapsed.

    Needed on both sides of the match: a heading reads "Object-Oriented
    Programming :" while the narration says "object oriented programming", so a
    literal search finds nothing and the chapter silently falls back to an even
    split.
    """
    return " ".join(re.sub(r"[^0-9a-z]+", " ", text.lower()).split())


def build_chapters(segs: list[dict], labels: list[str], anchors: list[str],
                   duration: int) -> list[dict]:
    """Align each chapter to the first moment its heading is actually spoken.

    Strategy per heading, most specific first:
      1. the whole normalised heading
      2. progressively shorter leading phrases from it
      3. its rarest content word

    Rarity beats "longest word": for "String handling" the longest word is
    "handling", which recurs throughout a Java lecture, while "string" localises
    it. Only if nothing matches does a chapter fall back to an even split.
    """
    timeline = [(s["start"], _norm(s["text"])) for s in segs]
    corpus = " ".join(t for _, t in timeline)

    def first_hit(needle: str, after: float) -> float | None:
        if not needle:
            return None
        for t, txt in timeline:
            if t >= after and needle in txt:
                return t
        for t, txt in timeline:          # allow going back if nothing ahead
            if needle in txt:
                return t
        return None

    def candidates(heading: str, anchor: str) -> list[str]:
        toks = [w for w in _norm(heading).split() if w not in STOPWORDS_NORM]
        out: list[str] = []
        if len(toks) > 1:
            out.append(" ".join(toks))                    # full phrase
            for cut in range(len(toks) - 1, 1, -1):       # shorter phrases
                out.append(" ".join(toks[:cut]))
        if toks:
            # rarest token in the transcript, ignoring absent ones
            counted = [(corpus.count(t), t) for t in toks]
            present = [(c, t) for c, t in counted if c > 0]
            if present:
                out.append(min(present)[1])
            out.append(max(toks, key=len))
        if anchor:
            out.append(_norm(anchor))
        seen, uniq = set(), []
        for c in out:
            if c and c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    pairs = list(zip(labels, anchors)) if len(labels) == len(anchors) else [
        (l, "") for l in labels]

    found: list[dict] = []
    cursor = 0.0
    for label, anchor in pairs:
        hit = None
        for cand in candidates(label, anchor):
            hit = first_hit(cand, cursor)
            if hit is not None:
                break
        found.append({"t": None if hit is None else int(hit), "label": label})
        if hit is not None:
            cursor = hit + 1

    matched = sum(1 for c in found if c["t"] is not None)
    log(f"chapter alignment: {matched}/{len(found)} located in the transcript")

    # Fill gaps and enforce monotonicity.
    out: list[dict] = []
    last = -1
    for i, ch in enumerate(found):
        t = ch["t"]
        if t is None:
            t = int(duration * i / max(len(found), 1))
        if t <= last:
            t = last + 1
        t = min(t, max(duration - 1, 0))
        out.append({"t": t, "label": ch["label"]})
        last = t

    if out and out[0]["t"] > 5:
        out.insert(0, {"t": 0, "label": "Overview"})
    return out


# ----------------------------------------------------------------- catalog

def catalog_entry(syl: dict, unit: dict, meta: dict, chapters: list[dict],
                  video_url: str, thumb_url: str, captions_url: str) -> dict:
    """Exactly the shape VideoCatalogRepository.parseVideos expects.

    An explicit `id` is set on purpose: the fallback id is slug(title)+hash(url),
    which collides across the catalog whenever titles and URLs repeat.
    """
    return {
        "id": f"{syl['subject']['id']}-{unit['id']}",
        "title": f"Unit {unit['n']}: {unit['title']}",
        "url": video_url,
        "subtitle": f"{meta['duration_sec'] // 60} min",
        "dur": meta["duration_sec"],
        "thumb": thumb_url,
        "vtt": captions_url,
        "chapters": chapters,
    }


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Postprocess a downloaded video overview.")
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--logo", default=str(ROOT / "assets" / "logo.png"))
    ap.add_argument("--sample-frames", action="store_true",
                    help="dump frames to measure the watermark box, then stop")
    ap.add_argument("--skip-logo", action="store_true",
                    help="run everything except the watermark swap (box not measured yet)")
    a = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} not found on PATH")

    syl = load_syllabus(a.syllabus)
    unit = unit_by_id(syl, a.unit)
    subject_id = syl["subject"]["id"]
    rec = get_record(subject_id, unit["id"])

    raw = Path(rec.get("raw_mp4") or "")
    if not raw.exists():
        raise SystemExit(f"no downloaded video for unit {unit['id']}; run collect.py first")

    outdir = BUILD / f"{subject_id}-{unit['id']}"
    outdir.mkdir(parents=True, exist_ok=True)

    if a.sample_frames:
        shots = sample_frames(raw, outdir / "frames")
        log("measure the NotebookLM mark in these, then fill config/logo.json:")
        for s in shots:
            log(f"  {s}")
        return

    meta = probe(raw)
    log(f"source: {meta['width']}x{meta['height']}, {meta['duration_sec']}s")

    final = outdir / "final.mp4"
    cfg = json.loads(LOGO_CFG.read_text(encoding="utf-8")) if LOGO_CFG.exists() else {}
    if a.skip_logo or not cfg.get("bottom_right"):
        log("watermark boxes not configured; copying source through unchanged")
        shutil.copy2(raw, final)
        wm_report = {}
    else:
        wm_report = swap_logo(raw, final, Path(a.logo), cfg, meta)
        segs = wm_report.get("bottom_right_segments", [])
        cards = sum(1 for s in segs if s.get("card"))
        log(f"watermark: bottom-right covered in {len(segs)} background segment(s)"
            f"{f', {cards} needing a white card (saturated background)' if cards else ''}")
        for s in segs:
            log(f"    {s['start']:>6.1f}-{s['end']:<6.1f}s plate={s['plate']}"
                f"{' (card)' if s.get('card') else ''}")
        win = wm_report.get("centre_top_window")
        log(f"    centre-top mark: {'covered %.1f-%.1fs' % win if win else 'not detected'}")

    segs = transcribe(final, outdir, a.whisper_model)
    write_vtt(segs, outdir / "captions.vtt")

    # Chapter labels are the course file's verbatim headings, captured in fire.py.
    # Anchors are the distinctive word from each heading, so alignment searches
    # for the same wording the student sees in their own notes.
    labels = rec.get("chapter_labels") or [t.title() for t in unit.get("anchors", [])]
    anchors = rec.get("chapter_anchors") or unit.get("anchors", [])
    chapters = build_chapters(segs, labels, anchors, meta["duration_sec"])
    log(f"{len(chapters)} chapters: " + ", ".join(f"{c['t']}s {c['label']}" for c in chapters))

    thumbnail(final, outdir / "thumb.jpg", at=min(meta["duration_sec"] * 0.1, 30))

    entry = catalog_entry(
        syl, unit, meta, chapters,
        video_url="TBD_AFTER_UPLOAD",
        thumb_url="TBD_AFTER_UPLOAD",
        captions_url="TBD_AFTER_UPLOAD",
    )
    (outdir / "catalog-entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

    record(subject_id, unit["id"], watermark=wm_report or None,
           state="postprocessed",
           final_mp4=str(final), duration_sec=meta["duration_sec"],
           chapters=chapters, thumb=str(outdir / "thumb.jpg"),
           captions=str(outdir / "captions.vtt"))
    log(f"done: {final}")


if __name__ == "__main__":
    main()
