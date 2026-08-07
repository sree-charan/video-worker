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
MARK_TEMPLATE = ROOT / "assets" / "notebooklm-mark.png"

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
    fps = 24.0
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "24/1"
    try:
        num, den = (float(x) for x in rate.split("/"))
        if den:
            fps = num / den
    except ValueError:
        pass
    return {
        "duration_sec": int(float(data["format"]["duration"])),
        "width": v.get("width"),
        "height": v.get("height"),
        "fps": round(fps, 3),
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


def light_variant(logo: Path, dst: Path) -> Path:
    """A white version of the logo, alpha preserved.

    Needed for saturated slides: the brand cyan-and-green is unreadable on
    orange, and a white plate behind it looks like a sticker. Recolouring the
    glyphs instead keeps the plate matching the background exactly.
    """
    from PIL import Image

    im = Image.open(logo).convert("RGBA")
    r, g, b, a = im.split()
    white = Image.new("L", im.size, 255)
    Image.merge("RGBA", (white, white, white, a)).save(dst)
    return dst


def _enable(segs: list[dict], key: str, want: bool) -> str | None:
    """ffmpeg enable expression covering every segment matching `key == want`.

    between() returns 1 or 0, so summing them ORs the ranges together.
    """
    picked = [s for s in segs if bool(s.get(key)) is want]
    if not picked:
        return None
    return "+".join(f"between(t,{s['start']:.2f},{s['end']:.2f})" for s in picked)


def feathered_plate(colour: str, w: int, h: int, path: Path,
                    feather_px: int = 4) -> Path:
    """An RGBA plate of one colour whose edges fade out.

    A hard-edged drawbox is unforgiving: if the sampled background colour is even
    slightly off, the rectangle's border shows as a visible seam. Ramping the
    alpha over a few pixels makes a small colour mismatch invisible instead.
    """
    from PIL import Image

    rgb = colour.lower().replace("0x", "").replace("#", "")
    if len(rgb) != 6:
        rgb = "fcfcfc"
    r, g, b = (int(rgb[i:i + 2], 16) for i in (0, 2, 4))

    pad = max(int(feather_px), 1)
    img = Image.new("RGBA", (w, h), (r, g, b, 0))
    px = img.load()
    for y in range(h):
        for x in range(w):
            # Distance to the nearest edge, clamped to the feather width.
            d = min(x, y, w - 1 - x, h - 1 - y)
            a = 255 if d >= pad else int(255 * (d + 1) / (pad + 1))
            px[x, y] = (r, g, b, a)
    img.save(path)
    return path


def append_outro(ff: str, main: Path, outro: Path, dst: Path, meta: dict,
                 enc: dict, bars: dict | None = None) -> None:
    """Concatenate the branded outro onto the finished lecture.

    Done last, after chapters and captions are built, so those describe the
    lecture rather than the branding: every chapter time stays valid because the
    outro only ever lands after all of them.

    Audio is resampled and the outro padded to the lecture's geometry, because
    concat demands matching streams and the two sources differ (the lecture is
    mono 44.1kHz from NotebookLM, the outro stereo 48kHz).
    """
    W, H = meta["width"], meta["height"]
    fps = meta.get("fps") or 24
    b = bars or {}
    l, r = int(b.get("left", 0)), int(b.get("right", 0))
    t, bo = int(b.get("top", 0)), int(b.get("bottom", 0))

    # Match the lecture's picture area by CROPPING the outro, not scaling it.
    # scale=1261:720:force_original_aspect_ratio=decrease rounds to even and
    # returned 1262, which then could not be padded down to 1261 - "Padded
    # dimensions cannot be smaller than input dimensions", and the run failed
    # every hour. Cropping has no rounding hazard, costs 19 of 1280 columns, and
    # keeps the outro unscaled and sharp.
    iw = max((W - l - r) // 2 * 2, 2)
    ih = max((H - t - bo) // 2 * 2, 2)
    ox, oy = (W - iw) // 2, (H - ih) // 2
    filt = (
        f"[0:v]scale={W}:{H},setsar=1,fps={fps}[v0];"
        f"[1:v]scale={W}:{H},crop={iw}:{ih}:{ox}:{oy},"
        f"pad={W}:{H}:{l}:{t}:color=black,setsar=1,fps={fps}[v1];"
        "[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a0];"
        "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a1];"
        "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
    )
    cmd = [ff, "-y", "-i", str(main), "-i", str(outro),
           "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-crf", str(enc.get("crf", 16)),
           "-preset", str(enc.get("preset", "slow"))]
    if enc.get("tune"):
        cmd += ["-tune", str(enc["tune"])]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k", str(dst)]
    run(cmd)


def _variant_scales(segs: list[dict], box: dict, prefix: str,
                    ratio: float) -> tuple[list[str], dict]:
    """Scale filters for only the logo variants this box actually needs.

    Declaring both and using one leaves an unconnected filter output, which
    ffmpeg rejects outright ("Filter scale:default has an unconnected output").

    Height is a multiple of the plate height so the logo can overhang onto
    untouched background: the plate stays at the original mark's footprint,
    which is what keeps it from cutting through artwork behind it.
    """
    lh = max(int(box["h"] * ratio) // 2 * 2, 8)
    steps, tags = [], {}
    for want, src_idx, suffix in ((False, 1, "d"), (True, 2, "w")):
        if _enable(segs, "light_logo", want) is None:
            continue
        tag = f"{prefix}{suffix}"
        steps.append(f"[{src_idx}:v]scale=-2:{lh}[{tag}]")
        tags[want] = tag
    return steps, tags


def _plate_overlays(tag_in: str, tag_out: str, box: dict, segs: list[dict],
                    workdir: Path, extra_inputs: list[Path],
                    base_index: int) -> tuple[list[str], str]:
    """One feathered plate per distinct colour, time-gated to its segments.

    Grouped by colour rather than per segment: colours repeat heavily (a light
    grey and an orange account for nearly every segment), so a handful of inputs
    covers hundreds of time ranges. A plate per segment would mean hundreds of
    ffmpeg inputs.
    """
    steps: list[str] = []
    cur = tag_in
    by_colour: dict[str, list[dict]] = {}
    for sg in segs:
        by_colour.setdefault(sg["plate"], []).append(sg)

    # The ramp must finish OUTSIDE the original box, or the mark's own edge sits
    # under a partly transparent plate. Growing by 0.18x the box size left the
    # box edge at alpha 204, and the NotebookLM icon showed through at 20%.
    # Grow strictly further than the feather width instead.
    feather_px = 4
    grow = feather_px + 2
    pw, ph = box["w"] + 2 * grow, box["h"] + 2 * grow
    px0, py0 = max(box["x"] - grow, 0), max(box["y"] - grow, 0)

    for i, (colour, group) in enumerate(by_colour.items()):
        png = feathered_plate(colour, pw, ph,
                              workdir / f"plate-{tag_out}-{i}.png",
                              feather_px=feather_px)
        extra_inputs.append(png)
        idx = base_index + len(extra_inputs) - 1
        expr = "+".join(f"between(t,{g['start']:.2f},{g['end']:.2f})" for g in group)
        nxt = f"{tag_out}p{i}"
        steps.append(f"[{cur}][{idx}:v]overlay={px0}:{py0}:"
                     f"format=auto:enable='{expr}'[{nxt}]")
        cur = nxt
    return steps, cur


def _cover_chain(tag_in: str, tag_out: str, tags: dict,
                 box: dict, segs: list[dict], align: str,
                 window=None, workdir: Path | None = None,
                 extra_inputs: list[Path] | None = None,
                 base_index: int = 3) -> list[str]:
    """Per-segment plate matching the background, then the right logo variant.

    Two overlays with complementary time gates: the brand-coloured logo over
    light segments, the white one over saturated segments. No card, no inset -
    the plate already matches, so only the glyphs need to change.
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    steps, cur = _plate_overlays(tag_in, tag_out, box, segs,
                                 workdir or Path("."), extra_inputs
                                 if extra_inputs is not None else [], base_index)

    # Centred on the plate. Right-aligned boxes keep the right edge flush with
    # the plate, matching where the original mark sat.
    ox = f"{x}+{w}-overlay_w" if align == "right" else f"{x}+({w}-overlay_w)/2"
    oy = f"{y}+({h}-overlay_h)/2"

    for idx, want in enumerate((False, True)):
        tag = tags.get(want)
        expr = _enable(segs, "light_logo", want)
        if not tag or not expr:
            continue
        if window:
            expr = f"({expr})*between(t,{window[0]:.2f},{window[1]:.2f})"
        nxt = f"{tag_out}o{idx}"
        steps.append(f"[{cur}][{tag}]overlay={ox}:{oy}:format=auto:"
                     f"enable='{expr}'[{nxt}]")
        cur = nxt
    steps.append(f"[{cur}]null[{tag_out}]")
    return steps


def swap_logo(src: Path, dst: Path, logo: Path, cfg: dict, meta: dict,
              ff: str = "ffmpeg", trim_outro: bool = True,
              workdir: Path | None = None) -> dict:
    """Replace both NotebookLM marks and cut the trailing end card.

    `delogo` is deliberately not used: it blurs a smeared patch rather than
    replacing the mark, which looks cheap. Frame geometry is never changed.

    The closing "notebooklm.google.com" card is full-frame branding, so unlike
    the two wordmarks it cannot be covered - it is trimmed, in this same encode
    so no extra pass is needed.
    """
    W, H = meta["width"], meta["height"]
    report: dict = {}

    cut = (watermark.detect_outro(ff, str(src), float(meta["duration_sec"]))
           if trim_outro else None)
    report["outro_cut_at"] = round(cut, 2) if cut else None

    light = light_variant(logo, (workdir or dst.parent) / "logo-light.png")

    # Locate rather than trust config. Config supplies only the search region.
    br_cfg = cfg["bottom_right"]
    found_br = watermark.locate_mark_any(
        ff, str(src), W, H, tuple(br_cfg["search_region"]), MARK_TEMPLATE,
        limit=float(br_cfg.get("search_seconds", 60)))
    if found_br:
        log(f"  bottom-right mark located by {found_br.get('method')} at "
            f"y={found_br['y']:.4f} x={found_br['x']:.4f}")
        br = resolve_box(found_br, W, H)
    else:
        log("  bottom-right mark not located; using the configured fallback box")
        br = resolve_box(br_cfg["fallback"], W, H)
    report["bottom_right_box"] = br
    report["bottom_right_box_norm"] = found_br or br_cfg["fallback"]

    br_present = watermark.smooth_presence(
        watermark.presence_any(ff, str(src), br, W, H, MARK_TEMPLATE, fps=1.0),
        max_gap=2.0, extend=1.0)
    br_segs = watermark.segments(watermark.region_series(ff, str(src), br, W, H),
                                 presence=br_present)
    if not br_segs:
        log("  bottom-right mark never matched; nothing will be drawn there")
    covered = sum(1 for _t, p in br_present if p)
    log(f"  bottom-right mark present in {covered}/{len(br_present)} sampled frames")
    report["bottom_right_segments"] = [
        {**s, "start": round(s["start"], 1), "end": round(s["end"], 1)} for s in br_segs]

    extra_inputs: list[Path] = []
    wd = workdir or dst.parent
    filters, br_tags = _variant_scales(
        br_segs, br, "lgbr", float(cfg["bottom_right"].get("logo_height_ratio", 1.5)))
    filters += _cover_chain("0:v", "v1", br_tags, br, br_segs,
                            cfg["bottom_right"].get("align", "right"),
                            workdir=wd, extra_inputs=extra_inputs, base_index=3)
    last = "v1"

    # Title-card mark: present only for the opening seconds, and how many is not
    # fixed, so the window is measured rather than assumed. Sampled at 8fps
    # because a coarse window overshoots onto the next slide.
    tc_cfg = cfg.get("centre_top")
    if tc_cfg:
        search = float(tc_cfg.get("search_seconds", 25))
        found_tc = watermark.locate_mark_any(
            ff, str(src), W, H, tuple(tc_cfg["search_region"]), MARK_TEMPLATE,
            limit=search)
        report["centre_top_box"] = found_tc
        if not found_tc:
            log("  centre-top mark not present in this video")
            tc_cfg = None
    if tc_cfg:
        log(f"  centre-top mark located by {found_tc.get('method')} at "
            f"y={found_tc['y']:.4f} x={found_tc['x']:.4f}")
        tc = resolve_box(found_tc, W, H)
        fps = watermark.SAMPLE_FPS
        tc_bg = watermark.region_series(ff, str(src), tc, W, H, limit=search, fps=fps)
        window = watermark.detect_presence(
            watermark.darkness_series(ff, str(src), tc, limit=search, fps=fps),
            tc_bg, limit=search, step=1.0 / fps)
        if window:
            # Clamp to the slide cut, but never earlier than the cut itself: the
            # mark stays on screen until the card leaves, so ending before the cut
            # exposes it for the last few frames.
            cuts = [c for c in watermark.slide_changes(ff, str(src), search)
                    if c > window[0] + 0.5]
            if cuts:
                window = (window[0], max(min(window[1], cuts[0]), cuts[0] - 0.05))
                report["clamped_to_slide_cut"] = round(cuts[0], 3)
        report["centre_top_window"] = window
        if window:
            # 4fps and generous smoothing: the title card is short, and its last
            # few frames are exactly where a missed OCR sample shows the mark.
            tc_present = watermark.smooth_presence(
                watermark.presence_any(ff, str(src), tc, W, H, MARK_TEMPLATE,
                                       fps=4.0, limit=search),
                max_gap=1.5, extend=0.75)
            tc_segs = watermark.segments(tc_bg, window=window, presence=tc_present)
            report["centre_top_segments"] = [
                {**s, "start": round(s["start"], 2), "end": round(s["end"], 2)}
                for s in tc_segs]
            tc_steps, tc_tags = _variant_scales(
                tc_segs, tc, "lgtc", float(tc_cfg.get("logo_height_ratio", 1.4)))
            filters += tc_steps
            filters += _cover_chain(last, "v2", tc_tags, tc, tc_segs,
                                    tc_cfg.get("align", "centre"), window=window,
                                    workdir=wd, extra_inputs=extra_inputs,
                                    base_index=3)
            last = "v2"

    enc = cfg.get("encode") or {}
    crf = str(enc.get("crf", 16))
    preset = str(enc.get("preset", "slow"))
    tune = enc.get("tune") or None
    report["encode"] = {"crf": crf, "preset": preset, "tune": tune}

    # Whole-frame sweep for placements the configured regions do not cover.
    if cfg.get("discover", True):
        known = [b for b in (report.get("bottom_right_box_norm"),
                             report.get("centre_top_box")) if b]
        extra = watermark.discover_marks(ff, str(src), W, H,
                                         fps=float(cfg.get("discover_fps", 0.2)),
                                         known=known)
        report["discovered"] = extra
        for b in extra:
            log(f"  WARNING unhandled NotebookLM mark found at "
                f"y={b['y']:.4f} x={b['x']:.4f} w={b['w']:.4f} "
                f"({b['frames_seen']} frames) - covering it")
        for i, b in enumerate(extra):
            bx = resolve_box(b, W, H)
            present = watermark.smooth_presence(
                watermark.presence_any(ff, str(src), bx, W, H, MARK_TEMPLATE, fps=1.0),
                max_gap=2.0, extend=1.0)
            segs = watermark.segments(
                watermark.region_series(ff, str(src), bx, W, H), presence=present)
            if not segs:
                continue
            steps, tags = _variant_scales(segs, bx, f"lgx{i}", 1.4)
            filters += steps
            filters += _cover_chain(last, f"vx{i}", tags, bx, segs, "centre",
                                    workdir=wd, extra_inputs=extra_inputs,
                                    base_index=3)
            last = f"vx{i}"

    cmd = [ff, "-y"]
    if cut:
        cmd += ["-t", f"{cut:.2f}"]        # applies to both video and audio
    cmd += ["-i", str(src), "-i", str(logo), "-i", str(light)]
    for extra in extra_inputs:
        cmd += ["-i", str(extra)]
    cmd += ["-filter_complex", ";".join(filters), "-map", f"[{last}]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", crf, "-preset", preset]
    if tune:
        cmd += ["-tune", tune]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "copy", str(dst)]
    log(f"  encoding at crf={crf} preset={preset}"
        + (f" tune={tune}" if tune else ""))
    run(cmd)
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

def load_whisper(model: str, attempts: int = 5):
    """Load the model, retrying the download.

    Unauthenticated Hugging Face downloads are rate limited, and a 429 here took
    down a whole batch after all five videos had already been generated and
    downloaded. The workflow caches the model between runs so this normally does
    not touch the network at all; the retry covers a cold cache.
    """
    import time as _time

    from faster_whisper import WhisperModel

    for i in range(1, attempts + 1):
        try:
            return WhisperModel(model, device="cpu", compute_type="int8")
        except Exception as exc:                       # noqa: BLE001 - hub errors vary
            if i == attempts:
                raise
            delay = 15 * (2 ** (i - 1))
            log(f"  whisper model load failed ({type(exc).__name__}); "
                f"retrying in {delay}s ({i}/{attempts - 1})")
            _time.sleep(delay)


def transcribe(mp4: Path, outdir: Path, model: str = "small") -> list[dict]:
    """faster-whisper -> word-ish segments. CPU int8 is fine for 10 minutes."""
    log(f"transcribing with faster-whisper ({model}, cpu/int8)")
    wm = load_whisper(model)
    segments, _info = wm.transcribe(str(mp4), language="en", vad_filter=True)

    segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    (outdir / "transcript.json").write_text(json.dumps(segs, indent=2), encoding="utf-8")
    return segs


def readability(segs: list[dict]) -> dict:
    """Flesch reading ease and grade level of the narration.

    Measured because "too complex" is otherwise a matter of opinion. The first
    pilot scored -1.7 ease / grade 18 - postgraduate prose - which is why the
    prompt now carries explicit language rules. Target is roughly 60+ ease and
    grade 8-10 for a second-year undergraduate audience.
    """
    text = " ".join(s["text"] for s in segs)
    words = re.findall(r"[A-Za-z']+", text)
    sentences = [x for x in re.split(r"[.!?]+", text) if x.strip()]
    if not words or not sentences:
        return {}

    def syllables(w: str) -> int:
        w = w.lower()
        n, prev = 0, False
        for ch in w:
            vowel = ch in "aeiouy"
            if vowel and not prev:
                n += 1
            prev = vowel
        if w.endswith("e") and n > 1:
            n -= 1
        return max(n, 1)

    W, S = len(words), len(sentences)
    SY = sum(syllables(w) for w in words)
    return {
        "words": W,
        "sentences": S,
        "words_per_sentence": round(W / S, 1),
        "flesch_reading_ease": round(206.835 - 1.015 * (W / S) - 84.6 * (SY / W), 1),
        "grade_level": round(0.39 * (W / S) + 11.8 * (SY / W) - 15.59, 1),
        "long_word_pct": round(
            100 * sum(1 for w in words if syllables(w) >= 4) / W, 1),
    }


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

    ff_bin = "ffmpeg"
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
        wm_report = swap_logo(raw, final, Path(a.logo), cfg, meta,
                              workdir=outdir)
        segs = wm_report.get("bottom_right_segments", [])
        log(f"  {len(segs)} background segment(s) tracked")
        lights = sum(1 for s in segs if s.get("light_logo"))
        log(f"watermark: bottom-right covered in {len(segs)} background segment(s)"
            f"{f', {lights} using the white logo (saturated background)' if lights else ''}")
        for s in segs:
            log(f"    {s['start']:>6.1f}-{s['end']:<6.1f}s plate={s['plate']}"
                f"{' white-logo' if s.get('light_logo') else ''}")
        win = wm_report.get("centre_top_window")
        log(f"    centre-top mark: {'covered %.1f-%.1fs' % win if win else 'not detected'}")
        cut = wm_report.get("outro_cut_at")
        log(f"    end card: {'trimmed at %.1fs (cut %.1fs)' % (cut, meta['duration_sec'] - cut)
                             if cut else 'not detected, nothing trimmed'}")

    # Re-probe after the encode: trimming changed the duration, and every piece
    # of metadata downstream (chapters, durationSec, subtitle) must describe the
    # file that actually ships.
    meta = probe(final)
    raw_mb = raw.stat().st_size / 1e6
    out_mb = final.stat().st_size / 1e6
    log(f"final: {meta['width']}x{meta['height']}, {meta['duration_sec']}s, "
        f"{out_mb:.1f} MB from {raw_mb:.1f} MB source "
        f"({out_mb / raw_mb * 100:.0f}% of source size, "
        f"{out_mb * 8000 / max(meta['duration_sec'], 1):.0f} kbps)")

    segs = transcribe(final, outdir, a.whisper_model)
    write_vtt(segs, outdir / "captions.vtt")

    read = readability(segs)
    if read:
        verdict = ("plain" if read["flesch_reading_ease"] >= 55
                   else "hard" if read["flesch_reading_ease"] >= 30 else "TOO COMPLEX")
        log(f"readability: ease={read['flesch_reading_ease']} "
            f"grade={read['grade_level']} "
            f"{read['words_per_sentence']} words/sentence "
            f"{read['long_word_pct']}% long words -> {verdict}")

    # Chapter labels are the course file's verbatim headings, captured in fire.py.
    # Anchors are the distinctive word from each heading, so alignment searches
    # for the same wording the student sees in their own notes.
    labels = rec.get("chapter_labels") or [t.title() for t in unit.get("anchors", [])]
    anchors = rec.get("chapter_anchors") or unit.get("anchors", [])
    chapters = build_chapters(segs, labels, anchors, meta["duration_sec"])
    log(f"{len(chapters)} chapters: " + ", ".join(f"{c['t']}s {c['label']}" for c in chapters))

    thumbnail(final, outdir / "thumb.jpg", at=min(meta["duration_sec"] * 0.1, 30))

    # Outro last: chapters and captions already describe the lecture, and the
    # outro only lands after every chapter time, so none of them shift.
    outro_cfg = cfg.get("outro") or {}
    outro_path = ROOT / str(outro_cfg.get("file", "assets/outro-light.mp4"))
    if outro_cfg.get("enabled", True) and outro_path.exists():
        with_outro = outdir / "final-with-outro.mp4"
        log(f"appending outro {outro_path.name} "
            f"({probe(outro_path)['duration_sec']}s)")
        bars = watermark.detect_pillarbox(ff_bin, str(final), meta["width"],
                                          meta["height"])
        if any(bars.values()):
            log(f"  matching pillarbox: left={bars['left']} right={bars['right']} "
                f"top={bars['top']} bottom={bars['bottom']}")
        try:
            append_outro(ff_bin, final, outro_path, with_outro, meta,
                         cfg.get("encode") or {}, bars=bars)
            with_outro.replace(final)
            meta = probe(final)
            log(f"final with outro: {meta['duration_sec']}s, "
                f"{final.stat().st_size / 1e6:.1f} MB")
            report["outro_appended"] = True
        except SystemExit as exc:
            # Branding is not worth losing a lecture over, and a hard failure here
            # made the hourly cron fail forever on one unit.
            log(f"  WARNING outro failed, shipping without it: {str(exc)[:200]}")
            with_outro.unlink(missing_ok=True)
            report["outro_appended"] = False
    elif outro_cfg.get("enabled", True):
        log(f"outro not found at {outro_path}; skipping")

    entry = catalog_entry(
        syl, unit, meta, chapters,
        video_url="TBD_AFTER_UPLOAD",
        thumb_url="TBD_AFTER_UPLOAD",
        captions_url="TBD_AFTER_UPLOAD",
    )
    (outdir / "catalog-entry.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

    record(subject_id, unit["id"], watermark=wm_report or None,
           readability=read or None,
           state="postprocessed",
           final_mp4=str(final), duration_sec=meta["duration_sec"],
           chapters=chapters, thumb=str(outdir / "thumb.jpg"),
           captions=str(outdir / "captions.vtt"))
    log(f"done: {final}")


if __name__ == "__main__":
    main()
