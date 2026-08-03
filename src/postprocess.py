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

from common import BUILD, ROOT, get_record, load_syllabus, log, record, unit_by_id  # noqa: E402

LOGO_CFG = ROOT / "config" / "logo.json"


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


def swap_logo(src: Path, dst: Path, logo: Path, box: dict) -> None:
    """Cover the source watermark with our own, same box, no geometry change.

    `delogo` is deliberately not used: it blurs a smeared patch rather than
    replacing the mark. An opaque overlay scaled to the measured box keeps the
    frame size and aspect identical, which is what the app's 16:9 thumbnails
    and the player both assume.
    """
    w, h, x, y = box["w"], box["h"], box["x"], box["y"]
    filt = (
        # Optional flat plate first, for when the watermark sits on moving
        # background and our logo has transparency.
        (f"[0:v]drawbox=x={x}:y={y}:w={w}:h={h}:color={box.get('plate', 'black')}@1:t=fill[bg];"
         if box.get("plate") else "[0:v]null[bg];")
        + f"[1:v]scale={w}:{h}[lg];[bg][lg]overlay={x}:{y}:format=auto[v]"
    )
    run(["ffmpeg", "-y", "-i", str(src), "-i", str(logo),
         "-filter_complex", filt, "-map", "[v]", "-map", "0:a?",
         "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-c:a", "copy", str(dst)])


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

def build_chapters(segs: list[dict], labels: list[str], anchors: list[str],
                   duration: int) -> list[dict]:
    """Align each intended chapter to the first time its anchor is spoken.

    Falls back to an even split only for chapters whose anchor never appears,
    and always keeps timestamps strictly increasing so the app's marker strip
    renders sanely.
    """
    joined = [(s["start"], s["text"].lower()) for s in segs]
    pairs = list(zip(labels, anchors)) if len(labels) == len(anchors) else [
        (l, l) for l in labels
    ]

    found: list[dict] = []
    cursor = 0.0
    for label, anchor in pairs:
        needle = anchor.lower().strip()
        hit = next((t for t, txt in joined if t >= cursor and needle in txt), None)
        if hit is None:
            hit = next((t for t, txt in joined if needle in txt), None)
        if hit is not None:
            found.append({"t": int(hit), "label": label})
            cursor = hit + 1
        else:
            found.append({"t": None, "label": label})

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
    box = json.loads(LOGO_CFG.read_text(encoding="utf-8")) if LOGO_CFG.exists() else {}
    if a.skip_logo or not box.get("w"):
        log("watermark box not configured; copying source through unchanged")
        shutil.copy2(raw, final)
    else:
        swap_logo(raw, final, Path(a.logo), box)
        log("watermark replaced in place")

    segs = transcribe(final, outdir, a.whisper_model)
    write_vtt(segs, outdir / "captions.vtt")

    labels = rec.get("chapter_labels") or [t.title() for t in unit.get("anchors", [])]
    chapters = build_chapters(segs, labels, unit.get("anchors", []), meta["duration_sec"])
    log(f"{len(chapters)} chapters: " + ", ".join(f"{c['t']}s {c['label']}" for c in chapters))

    thumbnail(final, outdir / "thumb.jpg", at=min(meta["duration_sec"] * 0.1, 30))

    entry = catalog_entry(
        syl, unit, meta, chapters,
        video_url="TBD_AFTER_UPLOAD",
        thumb_url="TBD_AFTER_UPLOAD",
        captions_url="TBD_AFTER_UPLOAD",
    )
    (outdir / "catalog-entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

    record(subject_id, unit["id"], state="postprocessed",
           final_mp4=str(final), duration_sec=meta["duration_sec"],
           chapters=chapters, thumb=str(outdir / "thumb.jpg"),
           captions=str(outdir / "captions.vtt"))
    log(f"done: {final}")


if __name__ == "__main__":
    main()
