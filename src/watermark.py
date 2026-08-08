"""Watermark removal that adapts to the frame behind it.

NotebookLM stamps its mark in two places:
  - a persistent wordmark, bottom right, for the whole video
  - a second wordmark centred near the top of the opening title card only

A fixed white plate is wrong for both. Measuring a real 15 minute explainer
showed the background under the bottom-right mark is (250,250,250) grey for most
of the video, pure white on the title and closing cards, and full-bleed ORANGE
(254,140,3) on a number of slides. A white rectangle there is glaringly visible.

So the background is sampled over time and the cover is drawn per segment.

The statistic matters: an average is useless here. A region straddling a white
card and an orange bar averages to beige (243,227,206), a colour present nowhere
in the frame. So a small pixel grid is sampled instead and the DOMINANT colour
taken, after discarding the darkest quarter (which is the wordmark itself).

  light, unsaturated background -> plate of exactly that colour, then our logo.
                                   The patch is imperceptible.
  saturated background          -> plate of that colour so the seam still
                                   disappears, plus a small inset white card to
                                   carry the logo, because cyan-and-green on
                                   orange is illegible. The card reads as
                                   deliberate branding.

Sampling is cheap: ffmpeg crops the region, scales it to a 16x8 grid and emits
rawvideo, so each sampled frame costs 384 bytes.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

# Frame-accurate. At 2fps the plate colour changed up to half a second before
# the background did, so an orange plate showed on a white slide - clearly
# visible. Sampling a tiny crop is cheap enough to do at video frame rate.
SAMPLE_FPS = 24.0
GRID_W, GRID_H = 16, 8    # pixels kept per sampled frame
QUANTISE = 6              # colour bucket size; below this a change is invisible
# Two frames at 24fps. At 0.2s a colour run shorter than that was absorbed into
# its neighbour, so when the next slide's yellow began animating in 0.05s before
# the cut, the plate stayed white and showed as a white box on yellow.
MIN_SEGMENT = 0.08
# Each segment becomes a drawbox filter, so a pathological video could build an
# unusable command line. Past this, the minimum is relaxed until it fits.
MAX_SEGMENTS = 240
DROP_DARKEST = 0.25       # fraction discarded before taking the mode


# ---------------------------------------------------------------- sampling

def _sample_grid(ff: str, mp4: str, crop: str, fps: float = SAMPLE_FPS,
                 limit: float | None = None
                 ) -> list[tuple[float, list[tuple[int, int, int]]]]:
    """A small pixel grid of a cropped region, once per sampled frame.

    `limit` caps how much of the input is decoded. The title-card mark only
    needs the opening seconds, and decoding a 15 minute video three times over
    to find it is wasted runner time.
    """
    cmd = [ff, "-v", "error"]
    if limit:
        cmd += ["-t", f"{limit:.2f}"]
    cmd += ["-i", mp4,
            "-vf", f"fps={fps},{crop},scale={GRID_W}:{GRID_H}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    per = GRID_W * GRID_H * 3
    out = []
    for i in range(len(raw) // per):
        chunk = raw[i * per:(i + 1) * per]
        px = [(chunk[j], chunk[j + 1], chunk[j + 2]) for j in range(0, per, 3)]
        out.append((i / fps, px))
    return out


def _crop(w: int, h: int, x: int, y: int) -> str:
    return f"crop={max(w,2)}:{max(h,2)}:{max(x,0)}:{max(y,0)}"


def dominant(pixels: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Most common colour, ignoring the darkest pixels.

    The darkest quarter is the wordmark's own glyphs; keeping them would drag a
    mode toward grey on a light background.
    """
    if not pixels:
        return (255, 255, 255)
    ordered = sorted(pixels, key=_lum)
    keep = ordered[int(len(ordered) * DROP_DARKEST):] or ordered
    buckets = Counter(_q(p) for p in keep)
    return buckets.most_common(1)[0][0]


def region_series(ff: str, mp4: str, box: dict, W: int, H: int,
                  pad_ratio: float = 0.6, limit: float | None = None,
                  fps: float = SAMPLE_FPS
                  ) -> list[tuple[float, tuple[int, int, int]]]:
    """Dominant background colour around a mark, over time.

    The box is expanded rather than sampled beside: a strip beside the mark can
    land on neighbouring content (in one sample frame the word "goals" sat
    immediately to its left), whereas expanding keeps the mark's own
    surroundings in view and the mode ignores the glyphs.
    """
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    px, py = int(w * 0.12), max(int(h * pad_ratio), 4)
    cx, cy = max(x - px, 0), max(y - py, 0)
    cw, ch = min(w + 2 * px, W - cx), min(h + 2 * py, H - cy)
    return [(t, dominant(px_list))
            for t, px_list in _sample_grid(ff, mp4, _crop(cw, ch, cx, cy),
                                           fps=fps, limit=limit)]


def darkness_series(ff: str, mp4: str, box: dict, limit: float | None = None,
                    fps: float = SAMPLE_FPS) -> list[tuple[float, float]]:
    """Mean luminance inside a box, for detecting whether the mark is present."""
    grid = _sample_grid(ff, mp4, _crop(box["w"], box["h"], box["x"], box["y"]),
                        fps=fps, limit=limit)
    return [(t, sum(_lum(p) for p in px) / len(px)) for t, px in grid]


# ------------------------------------------------------------- presence

def _lum(c: Iterable[int]) -> float:
    r, g, b = c
    return 0.299 * r + 0.587 * g + 0.114 * b


def detect_presence(box_series: list[tuple[float, float]],
                    bg_series: list[tuple[float, tuple[int, int, int]]],
                    limit: float, margin: float = 6.0,
                    step: float = 0.125) -> tuple[float, float] | None:
    """Time window in which the box is measurably darker than its background.

    Used for the title-card mark, whose duration is not fixed: measuring it per
    video is more robust than hardcoding "the first 8 seconds".

    The end is padded by only one sample step. Padding by a full second left the
    replacement logo lingering about 0.7s onto the following slide, which is
    visible. Better to leave a fraction of a frame of the original mark than to
    stamp our logo over unrelated content.
    """
    hits = [t for (t, boxlum), (_, bgc) in zip(box_series, bg_series)
            if t <= limit and _lum(bgc) - boxlum > margin]
    if not hits:
        return None
    end = hits[0]
    for t in hits[1:]:
        if t - end > 1.5:      # a lone later hit is unrelated content
            break
        end = t
    return (max(hits[0] - step, 0.0), end + step)


def slide_changes(ff: str, mp4: str, limit: float, threshold: float = 6.0,
                  fps: float = 24.0) -> list[float]:
    """Times at which the whole frame changes, i.e. slide cuts.

    Used to clamp the title-card window: presence detection alone left the
    replacement logo on screen for one frame of the following slide.
    """
    cmd = [ff, "-v", "error", "-t", f"{limit:.2f}", "-i", mp4,
           "-vf", f"fps={fps},scale=64:36", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    per = 64 * 36
    n = len(raw) // per
    out, prev = [], None
    for i in range(n):
        cur = raw[i * per:(i + 1) * per]
        if prev is not None:
            d = sum(abs(a - b) for a, b in zip(cur, prev)) / per
            if d > threshold:
                out.append(i / fps)
        prev = cur
    return out


def detect_pillarbox(ff: str, mp4: str, W: int, H: int, fps: float = 0.5,
                     limit: float = 60.0, start: float = 30.0,
                     black: int = 32) -> dict:
    """Width of the black bars framing the picture, if any.

    NotebookLM renders 1280x720 with roughly 9 and 10 pixel black bars at left
    and right, so the actual picture is about 1261 wide. An outro that fills the
    frame edge to edge therefore appears to widen at the cut. Measuring the bars
    lets the outro be padded to the same picture area, which removes the jump.
    """
    import numpy as np
    from collections import Counter

    cmd = [ff, "-v", "error", "-ss", f"{start:.2f}", "-t", f"{limit:.2f}",
           "-i", mp4, "-vf", f"fps={fps}", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    per = W * H
    if len(raw) < per:
        return {"left": 0, "right": 0, "top": 0, "bottom": 0}

    counts = {k: Counter() for k in ("left", "right", "top", "bottom")}
    for i in range(len(raw) // per):
        fr = np.frombuffer(raw[i * per:(i + 1) * per], dtype=np.uint8).reshape(H, W)
        cols, rows = fr.max(axis=0), fr.max(axis=1)
        counts["left"][int(np.argmax(cols > black))] += 1
        counts["right"][int(np.argmax(cols[::-1] > black))] += 1
        counts["top"][int(np.argmax(rows > black))] += 1
        counts["bottom"][int(np.argmax(rows[::-1] > black))] += 1
    # Modal value: a single busy frame must not move the answer.
    return {k: (c.most_common(1)[0][0] if c else 0) for k, c in counts.items()}


# ------------------------------------------------------------- outro card

def detect_outro(ff: str, mp4: str, duration: float, tail: float = 20.0,
                 fps: float = 2.0, max_outro: float = 8.0) -> float | None:
    """Start time of the trailing NotebookLM end card, or None.

    The video closes on a full-frame "notebooklm.google.com" card that cannot be
    patched over - it has to be cut. Its length is not fixed, so it is found by
    measuring how much each frame in the tail differs from the very last frame:
    content frames differ a lot and by a constant amount, the animating card
    differs less, and the settled card differs by nothing. The boundary is the
    last frame that still looks like content.

    Returns None rather than guessing if the result is implausible, so a video
    without an end card is never truncated.
    """
    start = max(duration - tail, 0.0)
    cmd = [ff, "-v", "error", "-ss", f"{start:.2f}", "-i", mp4,
           "-vf", f"fps={fps},scale=160:90", "-pix_fmt", "gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    per = 160 * 90
    n = len(raw) // per
    if n < 4:
        return None

    frames = [raw[i * per:(i + 1) * per] for i in range(n)]
    last = frames[-1]
    diffs = [sum(abs(a - b) for a, b in zip(f, last)) / per for f in frames]

    peak = max(diffs)
    if peak < 4.0:                      # the whole tail is one static frame
        return None
    threshold = max(3.0, peak * 0.25)

    boundary = None
    for i in range(n - 1, -1, -1):
        if diffs[i] > threshold:
            boundary = i
            break
    if boundary is None or boundary == n - 1:
        return None

    cut = start + (boundary + 0.5) / fps
    outro_len = duration - cut
    if not (0.3 <= outro_len <= max_outro):
        return None
    return cut


def _frames_gray(ff: str, mp4: str, W: int, H: int, limit: float | None,
                 fps: float) -> list[tuple[float, Any]]:
    """Full-resolution greyscale frames, for measuring where a mark actually is."""
    import numpy as np

    cmd = [ff, "-v", "error"]
    if limit:
        cmd += ["-t", f"{limit:.2f}"]
    cmd += ["-i", mp4, "-vf", f"fps={fps}", "-pix_fmt", "gray",
            "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    per = W * H
    return [(i / fps,
             np.frombuffer(raw[i * per:(i + 1) * per], dtype=np.uint8).reshape(H, W))
            for i in range(len(raw) // per)]


def _candidates(frame, region: tuple[float, float, float, float],
                W: int, H: int, thr: int = 120) -> list[dict]:
    """Dark row-bands inside a region, with their bounding boxes."""
    import numpy as np

    y0f, y1f, x0f, x1f = region
    y0, y1 = int(H * y0f), int(H * y1f)
    x0, x1 = int(W * x0f), int(W * x1f)
    sub = frame[y0:y1, x0:x1] < thr
    rows = sub.any(axis=1)

    out, start = [], None
    for i, v in enumerate(list(rows) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            seg = sub[start:i]
            xs = np.nonzero(seg.any(axis=0))[0]
            if len(xs):
                out.append({"x": x0 + int(xs[0]), "y": y0 + start,
                            "w": int(xs[-1] - xs[0]) + 1, "h": i - start})
            start = None
    return out


# The wordmark's shape, as fractions of the frame. Measured across five renders:
# 179x19 to 182x25 at 1280x720, so aspect 7-10 and about 14% of frame width.
# The title text fails these tests by being far taller and wider, and corner
# decorations by being nearly square.
MARK_MIN_H, MARK_MAX_H = 0.018, 0.055
MARK_MIN_W, MARK_MAX_W = 0.070, 0.230
MARK_MIN_AR, MARK_MAX_AR = 4.5, 14.0


def _mark_like(b: dict, W: int, H: int) -> bool:
    nh, nw = b["h"] / H, b["w"] / W
    ar = b["w"] / max(b["h"], 1)
    return (MARK_MIN_H <= nh <= MARK_MAX_H and MARK_MIN_W <= nw <= MARK_MAX_W
            and MARK_MIN_AR <= ar <= MARK_MAX_AR)


def locate_mark(ff: str, mp4: str, W: int, H: int,
                region: tuple[float, float, float, float],
                limit: float = 20.0, fps: float = 2.0,
                pad_x: float = 0.006, pad_y: float = 0.010) -> dict | None:
    """Find the wordmark by shape, instead of trusting fixed coordinates.

    The title-card mark is positioned relative to the title block, so its height
    on screen depends on how many lines the title takes. Across five units it sat
    at y=100 for a one-line title and y=50 for a two-line one. A hardcoded box
    hit unit 1, missed unit 4 completely, and on unit 5 stamped our logo over the
    title text while the original mark stayed visible 30px above.

    So the mark is measured per video: dark row-bands in the search region are
    filtered by the wordmark's shape, and the box agreed on by the most frames
    wins. Returns normalised coordinates, padded, or None if nothing matches.
    """
    from collections import Counter

    frames = _frames_gray(ff, mp4, W, H, limit, fps)
    if not frames:
        return None

    votes: Counter = Counter()
    for _t, fr in frames:
        for b in _candidates(fr, region, W, H):
            if _mark_like(b, W, H):
                # Quantise so tiny per-frame jitter still agrees on one box.
                votes[(b["x"] // 4, b["y"] // 4, b["w"] // 4, b["h"] // 4)] += 1
    if not votes:
        return None

    (qx, qy, qw, qh), seen = votes.most_common(1)[0]
    x, y, w, h = qx * 4, qy * 4, qw * 4, qh * 4
    return {
        "x": max(x / W - pad_x, 0.0),
        "y": max(y / H - pad_y, 0.0),
        "w": min(w / W + 2 * pad_x, 1.0),
        "h": min(h / H + 2 * pad_y, 1.0),
        "frames_agreeing": seen,
        "frames_sampled": len(frames),
    }


def presence_series(ff: str, mp4: str, box: dict, W: int, H: int,
                    limit: float | None = None, fps: float = SAMPLE_FPS,
                    margin: float = 6.0) -> list[tuple[float, bool]]:
    """Whether the mark is actually on screen, sampled over time.

    A mark must only be covered where it exists. NotebookLM omits the
    bottom-right wordmark on the title card when it shows the centred one, so
    plating that corner there destroys background and adds a logo the original
    never had.
    """
    box_lum = darkness_series(ff, mp4, box, limit=limit, fps=fps)
    bg = region_series(ff, mp4, box, W, H, limit=limit, fps=fps)
    return [(t, (_lum(bgc) - bl) > margin)
            for (t, bl), (_, bgc) in zip(box_lum, bg)]


def _box_masks(ff: str, mp4: str, box: dict, fps: float,
               limit: float | None, drop: int = 45):
    """Binary dark-masks of a box over time, each relative to its own background.

    Masking against the frame's own local background makes the mark's footprint
    identical whether it sits on white, grey or orange, which is what allows one
    template to match everywhere.
    """
    import numpy as np

    w, h = box["w"], box["h"]
    cmd = [ff, "-v", "error"]
    if limit:
        cmd += ["-t", f"{limit:.2f}"]
    cmd += ["-i", mp4,
            "-vf", f"fps={fps},{_crop(w, h, box['x'], box['y'])}",
            "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    per = w * h
    out = []
    for i in range(len(raw) // per):
        crop = np.frombuffer(raw[i * per:(i + 1) * per], dtype=np.uint8).reshape(h, w)
        bg = np.percentile(crop, 85)
        out.append((i / fps, crop < (bg - drop)))
    return out


def mark_presence(ff: str, mp4: str, box: dict, fps: float = 4.0,
                  limit: float | None = None,
                  iou_threshold: float = 0.45) -> list[tuple[float, bool]]:
    """Whether the wordmark itself is in the box, by matching its shape.

    An earlier version asked only "is this box darker than its surroundings",
    which any dark artwork satisfies. On the title card a decoration bar 111px
    tall runs through the bottom-right box, so the mark was reported present and
    our logo was stamped into a corner NotebookLM leaves empty.

    The wordmark renders identically every time it appears, so a template of its
    dark footprint is built from the frames that look like it, and each frame is
    then scored by intersection-over-union against that template. A decoration
    bar overlaps poorly and is rejected.
    """
    import numpy as np

    series = _box_masks(ff, mp4, box, fps, limit)
    if not series:
        return []

    h, w = series[0][1].shape
    # A frame looks like the wordmark if its dark pixels span most of the box
    # width and a plausible share of its area.
    candidates = []
    for _t, m in series:
        frac = m.mean()
        if not (0.04 <= frac <= 0.55):
            continue
        cols = np.nonzero(m.any(axis=0))[0]
        if len(cols) and (cols[-1] - cols[0] + 1) >= 0.55 * w:
            candidates.append(m)

    if not candidates:
        return [(t, False) for t, _ in series]

    template = (np.mean(np.stack(candidates), axis=0) >= 0.5)
    if not template.any():
        return [(t, False) for t, _ in series]

    out = []
    for t, m in series:
        inter = np.logical_and(m, template).sum()
        union = np.logical_or(m, template).sum()
        out.append((t, bool(union and (inter / union) >= iou_threshold)))
    return out


def _ocr_boxes(img, min_conf: int = 40) -> list[dict]:
    """OCR a PIL image and return boxes for the generator's wordmark.

    Tesseract splits and mangles the wordmark in several ways depending on the
    frame - "NotebookLM", "Notebook LM", "NotebooklM" - so matching is done on a
    letters-only lowercase form and accepts any word containing "notebook", plus
    a bare "lm" immediately to its right.

    The product was renamed from NotebookLM to Google Notebook, and the mark
    changed with it: the wordmark is now two words, so "Google" sits to the LEFT
    of the word we key on and has to be absorbed as well, or half the mark stays
    uncovered. "gemini" is accepted for the same reason - the branding moved once
    and can move again, and a missed mark ships in the video.
    """
    import re

    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(img, output_type=Output.DICT,
                                     config="--psm 11")
    words = []
    for i, raw in enumerate(data["text"]):
        txt = re.sub(r"[^a-z]", "", (raw or "").lower())
        if not txt:
            continue
        try:
            conf = int(float(data["conf"][i]))
        except (TypeError, ValueError):
            conf = -1
        if conf < min_conf:
            continue
        words.append({"text": txt, "conf": conf,
                      "x": data["left"][i], "y": data["top"][i],
                      "w": data["width"][i], "h": data["height"][i]})

    def same_line(a: dict, b: dict) -> bool:
        return abs(a["y"] - b["y"]) < max(a["h"], b["h"])

    hits = []
    for i, wd in enumerate(words):
        if "notebook" not in wd["text"] and "gemini" not in wd["text"]:
            continue
        box = dict(wd)
        # Absorb a trailing "lm" that tesseract split off.
        for other in words[i + 1:i + 3]:
            if other["text"] in ("lm", "im", "l", "m") and \
               same_line(other, wd) and \
               0 <= other["x"] - (wd["x"] + wd["w"]) < wd["h"] * 2:
                right = max(box["x"] + box["w"], other["x"] + other["w"])
                box["w"] = right - box["x"]
        # Absorb a leading "Google" from the current two-word wordmark.
        for other in words[max(i - 3, 0):i]:
            if other["text"] in ("google", "googie", "gemini") and \
               same_line(other, wd) and \
               0 <= wd["x"] - (other["x"] + other["w"]) < wd["h"] * 2:
                right = box["x"] + box["w"]
                box["x"] = min(box["x"], other["x"])
                box["y"] = min(box["y"], other["y"])
                box["w"] = right - box["x"]
                box["h"] = max(box["h"], other["h"])
        hits.append(box)
    return hits


def refine_extent(frame, box: dict, W: int, H: int, drop: int = 40,
                  gap_ratio: float = 0.7) -> dict:
    """Grow an OCR text box to the mark's true painted extent.

    OCR returns the text only, so a fixed "icon is 1.5x the text height" guess
    left the icon and the final M poking out either side of the replacement logo.
    Measuring instead: column runs that differ from the local background around
    the OCR box are merged across gaps up to 0.7x the mark height - the icon sits
    11px from the text on a 20px mark, while inter-letter gaps are 1-2px - and the
    run overlapping the OCR box wins.

    The comparison is on absolute difference from the background, in either
    direction. It used to look only for pixels DARKER than the 85th percentile,
    which cannot see a white mark on a saturated background: on those segments the
    measurement silently returned the text-only box, that box won the vote, and
    the icon was left uncovered 14px to the left of the plate.
    """
    import numpy as np

    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    ny0, ny1 = max(y - int(h * 1.2), 0), min(y + h + int(h * 1.2), H)
    nx0, nx1 = max(x - int(h * 4), 0), min(x + w + int(h * 4), W)
    sub = frame[ny0:ny1, nx0:nx1].astype(float)
    if sub.size == 0:
        return box

    # Median, not a high percentile: the window is mostly background whichever
    # side of it the mark sits on in brightness.
    bg = np.median(sub)
    mask = np.abs(sub - bg) > drop
    cols = mask.any(axis=0)

    runs, start = [], None
    for i, v in enumerate(list(cols) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i - 1])
            start = None
    if not runs:
        return box

    merged = [runs[0]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] - 1 <= max(int(h * gap_ratio), 2):
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    lo, hi = x - nx0, x + w - nx0
    chosen = next((r for r in merged if r[0] <= hi and r[1] >= lo), None)
    if chosen is None:
        return box

    band = mask[:, chosen[0]:chosen[1] + 1]
    rws = np.nonzero(band.any(axis=1))[0]
    if not len(rws):
        return box
    return {"x": nx0 + chosen[0], "y": ny0 + int(rws[0]),
            "w": chosen[1] - chosen[0] + 1, "h": int(rws[-1] - rws[0]) + 1}


def locate_mark_ocr(ff: str, mp4: str, W: int, H: int,
                    region: tuple[float, float, float, float],
                    limit: float = 20.0, fps: float = 2.0, upscale: int = 3,
                    icon_ratio: float = 1.5, pad: float = 0.30) -> dict | None:
    """Find the wordmark by reading it, rather than guessing from its shape.

    Shape filtering was not safe: on a two-line title card the top edge of the
    second line measured 243x32 with aspect 7.6, which fits the wordmark's shape
    signature, won the vote, and put our logo across the word "Unit 4" while the
    real mark stayed visible above.

    Text is unambiguous - the title never reads "NotebookLM". The region is
    upscaled before OCR because the mark is only ~19px tall at 720p, and the box
    is widened to the left to take in the icon that precedes the text.
    """
    from collections import Counter

    import numpy as np
    from PIL import Image

    y0f, y1f, x0f, x1f = region
    ry0, rx0 = int(H * y0f), int(W * x0f)
    rh, rw = int(H * y1f) - ry0, int(W * x1f) - rx0

    frames = _frames_gray(ff, mp4, W, H, limit, fps)
    votes: Counter = Counter()
    for _t, fr in frames:
        crop = fr[ry0:ry0 + rh, rx0:rx0 + rw]
        img = Image.fromarray(crop.astype(np.uint8)).resize(
            (rw * upscale, rh * upscale), Image.LANCZOS)
        for hit in _ocr_boxes(img):
            raw_box = {"x": rx0 + hit["x"] / upscale, "y": ry0 + hit["y"] / upscale,
                       "w": hit["w"] / upscale, "h": hit["h"] / upscale}
            ext = refine_extent(fr, raw_box, W, H)
            votes[(ext["x"] // 2, ext["y"] // 2, ext["w"] // 2, ext["h"] // 2)] += 1

    if not votes:
        return None
    (qx, qy, qw, qh), seen = votes.most_common(1)[0]
    x, y, w, h = qx * 2, qy * 2, qw * 2, qh * 2
    # Small uniform pad: the extent is measured, so this only covers antialiasing.
    px, py = max(w * 0.02, 2.0), max(h * 0.25, 3.0)
    return {
        "x": max((x - px) / W, 0.0),
        "y": max((y - py) / H, 0.0),
        "w": min((w + 2 * px) / W, 1.0),
        "h": min((h + 2 * py) / H, 1.0),
        "frames_agreeing": seen,
        "frames_sampled": len(frames),
        "method": "ocr",
    }


# Plausibility bounds for a discovered wordmark, as fractions of the frame.
# The real mark measures w=0.073 h=0.022 (aspect 3.3) bottom-right, and roughly
# w=0.15 h=0.03 on the title card. The false positive that triggered these was
# w=0.39 h=0.067 - a line of slide text that got plated over.
MAX_MARK_W = 0.22
MAX_MARK_H = 0.05
MIN_MARK_ASPECT = 1.8
MAX_MARK_ASPECT = 14.0
# A mark seen in a single sampled frame is an OCR misread, not branding.
MIN_DISCOVERY_FRAMES = 2


def discover_marks(ff: str, mp4: str, W: int, H: int, fps: float = 0.2,
                   limit: float | None = None, upscale: int = 2,
                   known: list[dict] | None = None,
                   overlap_tol: float = 0.04) -> list[dict]:
    """Sweep the whole frame for the wordmark, anywhere it might appear.

    The two configured regions cover every placement seen so far - bottom right
    for the whole video, centred above the title on the opening card - but a
    layout we have not seen would ship NotebookLM branding silently. This sweeps
    the entire frame at a low rate and reports anything found, so an unknown
    placement becomes a logged warning and a covered box instead of a surprise in
    a published lecture.

    Deliberately coarse: a full-frame OCR pass is expensive, and a mark that
    persists for even a few seconds will be caught at this rate.
    """
    from collections import Counter

    import numpy as np
    from PIL import Image

    # Discovery is OCR-only: there is no template sweep for an unknown placement.
    # Without this guard a missing tesseract raised TesseractNotFoundError and took
    # the whole run down, instead of simply skipping the sweep.
    if not ocr_available():
        return []

    frames = _frames_gray(ff, mp4, W, H, limit, fps)
    votes: Counter = Counter()
    for _t, fr in frames:
        img = Image.fromarray(fr.astype(np.uint8)).resize(
            (W * upscale, H * upscale), Image.LANCZOS)
        for hit in _ocr_boxes(img, min_conf=45):
            raw = {"x": hit["x"] / upscale, "y": hit["y"] / upscale,
                   "w": hit["w"] / upscale, "h": hit["h"] / upscale}
            ext = refine_extent(fr, raw, W, H)
            votes[(ext["x"] // 8, ext["y"] // 8, ext["w"] // 8, ext["h"] // 8)] += 1

    out: list[dict] = []
    for (qx, qy, qw, qh), seen in votes.most_common():
        x, y, w, h = qx * 8, qy * 8, qw * 8, qh * 8
        box = {"x": x / W, "y": y / H, "w": w / W, "h": h / H,
               "frames_seen": seen, "method": "discovery"}
        # Skip anything already handled by a configured region, and near-duplicates.
        if any(abs(box["x"] - k["x"]) < overlap_tol and abs(box["y"] - k["y"]) < overlap_tol
               for k in (known or []) + out):
            continue
        # A wordmark is small and wide. Slide text is not, and covering it
        # destroys teaching content: a box 39% of the frame wide over the words
        # "Deactivate / Short voltage sources" was plated as if it were branding.
        if box["w"] > MAX_MARK_W or box["h"] > MAX_MARK_H:
            log(f"  discovery ignored an implausible box at y={box['y']:.4f} "
                f"x={box['x']:.4f} w={box['w']:.4f} h={box['h']:.4f}: too large "
                f"for a wordmark, almost certainly slide text")
            continue
        aspect = box["w"] / box["h"] if box["h"] else 0
        if not MIN_MARK_ASPECT <= aspect <= MAX_MARK_ASPECT:
            log(f"  discovery ignored a box at y={box['y']:.4f} x={box['x']:.4f}: "
                f"aspect {aspect:.1f} is not wordmark-shaped")
            continue
        # One frame is not evidence. At 0.2fps a real mark persists across
        # samples; a single OCR misread does not.
        if seen < MIN_DISCOVERY_FRAMES:
            log(f"  discovery saw a box at y={box['y']:.4f} x={box['x']:.4f} in "
                f"only {seen} frame(s); ignoring as a misread")
            continue
        out.append(box)
    return out


def presence_ocr(ff: str, mp4: str, box: dict, W: int, H: int,
                 fps: float = 1.0, limit: float | None = None,
                 upscale: int = 3, slack: float = 0.6) -> list[tuple[float, bool]]:
    """Read the box over time and report when it actually says 'NotebookLM'.

    Decides where to draw. A brightness test put a logo into the title card's
    empty bottom-right corner because a decoration bar 111px tall passes through
    that box; reading the text cannot make that mistake.
    """
    import numpy as np
    from PIL import Image

    x, y = box["x"], box["y"]
    w, h = box["w"], box["h"]
    pad = int(h * slack)
    cx, cy = max(x - pad, 0), max(y - pad, 0)
    cw, ch = min(w + 2 * pad, W - cx), min(h + 2 * pad, H - cy)

    frames = _frames_gray(ff, mp4, W, H, limit, fps)
    out = []
    for t, fr in frames:
        crop = fr[cy:cy + ch, cx:cx + cw]
        img = Image.fromarray(crop.astype(np.uint8)).resize(
            (cw * upscale, ch * upscale), Image.LANCZOS)
        out.append((t, bool(_ocr_boxes(img))))
    return out


def ocr_available() -> bool:
    import shutil
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return shutil.which("tesseract") is not None


def locate_mark_any(ff: str, mp4: str, W: int, H: int, region, template_path,
                    limit: float = 20.0) -> dict | None:
    """Locate the mark, preferring OCR and falling back to template matching.

    OCR is preferred because it is the only unambiguous signal. Greyscale
    template correlation cannot tell NotebookLM's mark from our own replacement
    logo - measured 0.64 for the real mark against 0.65 for ours, since both are
    an icon beside a wide wordmark. It is kept as a fallback for when tesseract
    is unavailable.
    """
    if ocr_available():
        found = locate_mark_ocr(ff, mp4, W, H, region, limit=limit)
        if found:
            return found
    return locate_mark_template(ff, mp4, W, H, region, template_path, limit=limit)


def presence_any(ff: str, mp4: str, box: dict, W: int, H: int, template_path,
                 fps: float = 1.0, limit: float | None = None
                 ) -> list[tuple[float, bool]]:
    if ocr_available():
        return presence_ocr(ff, mp4, box, W, H, fps=fps, limit=limit)
    return presence_template(ff, mp4, box, W, H, template_path,
                             fps=max(fps, 2.0), limit=limit)


def _ncc(window, template, t_mean: float, t_std: float) -> float:
    """Normalised cross-correlation of one window against the template."""
    import numpy as np

    w_mean = window.mean()
    w_std = window.std()
    if w_std < 1e-6 or t_std < 1e-6:
        return 0.0
    return float(np.mean((window - w_mean) * (template - t_mean)) / (w_std * t_std))


def _load_template(path: Path | str):
    import numpy as np
    from PIL import Image

    img = Image.open(str(path)).convert("L")
    return np.asarray(img).astype(float)


def locate_mark_template(ff: str, mp4: str, W: int, H: int,
                         region: tuple[float, float, float, float],
                         template_path, limit: float = 20.0, fps: float = 1.0,
                         scales=(0.75, 0.875, 1.0, 1.125, 1.25, 1.5),
                         step: int = 2, min_score: float = 0.55,
                         pad: float = 0.25) -> dict | None:
    """Find the wordmark by matching its actual pixels.

    Shape filtering was not safe: on a two-line title card the top edge of the
    second line measured 243x32 with aspect 7.6, fitting the wordmark's shape
    signature well enough to win the vote. Our logo landed across the word
    "Unit 4" while the real mark stayed visible above it.

    The mark is a fixed raster, so correlating against a reference crop of it is
    exact and cannot confuse itself with text. Several scales are tried because
    render resolution varies, and correlation is normalised so it holds on white,
    grey or orange backgrounds.
    """
    import numpy as np
    from PIL import Image

    base = _load_template(template_path)
    y0f, y1f, x0f, x1f = region
    ry0, rx0 = int(H * y0f), int(W * x0f)
    ry1, rx1 = int(H * y1f), int(W * x1f)

    frames = _frames_gray(ff, mp4, W, H, limit, fps)
    best = None
    for _t, fr in frames:
        crop = fr[ry0:ry1, rx0:rx1].astype(float)
        ch, cw = crop.shape
        for sc in scales:
            th, tw = max(int(base.shape[0] * sc), 6), max(int(base.shape[1] * sc), 20)
            if th >= ch or tw >= cw:
                continue
            tpl = np.asarray(Image.fromarray(base.astype(np.uint8))
                             .resize((tw, th), Image.LANCZOS)).astype(float)
            t_mean, t_std = tpl.mean(), tpl.std()
            for yy in range(0, ch - th + 1, step):
                for xx in range(0, cw - tw + 1, step):
                    s = _ncc(crop[yy:yy + th, xx:xx + tw], tpl, t_mean, t_std)
                    if best is None or s > best["score"]:
                        best = {"score": s, "x": rx0 + xx, "y": ry0 + yy,
                                "w": tw, "h": th, "scale": sc}
    if not best or best["score"] < min_score:
        return None

    px, py = best["w"] * 0.03, best["h"] * pad
    return {
        "x": max((best["x"] - px) / W, 0.0),
        "y": max((best["y"] - py) / H, 0.0),
        "w": min((best["w"] + 2 * px) / W, 1.0),
        "h": min((best["h"] + 2 * py) / H, 1.0),
        "score": round(best["score"], 3),
        "scale": best["scale"],
        "method": "template",
    }


def presence_template(ff: str, mp4: str, box: dict, W: int, H: int,
                      template_path, fps: float = 2.0,
                      limit: float | None = None, jitter: int = 3,
                      min_score: float = 0.45) -> list[tuple[float, bool]]:
    """Whether the mark is in the box at each sampled time, by correlation.

    Only a small neighbourhood of the located box is searched, so this is cheap
    enough to run across a whole lecture. A brightness test previously reported
    the mark present wherever any dark artwork crossed the box, which is how a
    logo ended up in the title card's empty corner.
    """
    import numpy as np
    from PIL import Image

    base = _load_template(template_path)
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    tpl = np.asarray(Image.fromarray(base.astype(np.uint8))
                     .resize((max(w, 20), max(h, 6)), Image.LANCZOS)).astype(float)
    t_mean, t_std = tpl.mean(), tpl.std()

    frames = _frames_gray(ff, mp4, W, H, limit, fps)
    out = []
    for t, fr in frames:
        best = 0.0
        for dy in range(-jitter, jitter + 1, jitter or 1):
            for dx in range(-jitter, jitter + 1, jitter or 1):
                yy, xx = y + dy, x + dx
                if yy < 0 or xx < 0 or yy + tpl.shape[0] > H or xx + tpl.shape[1] > W:
                    continue
                s = _ncc(fr[yy:yy + tpl.shape[0], xx:xx + tpl.shape[1]].astype(float),
                         tpl, t_mean, t_std)
                best = max(best, s)
        out.append((t, best >= min_score))
    return out


def smooth_presence(series: list[tuple[float, bool]], max_gap: float = 1.5,
                    extend: float = 0.5) -> list[tuple[float, bool]]:
    """Close short holes in a presence series and extend its trailing edge.

    OCR does not read the mark in every single frame - motion blur during a slide
    transition is enough to miss one - and a single missed sample used to end
    coverage, which is how NotebookLM became visible for the last few frames of a
    title card while the log cheerfully reported it covered.

    Holes shorter than `max_gap` between two positives are filled, and each run
    of positives is extended `extend` seconds past its last positive sample, so
    coverage outlasts the mark rather than stopping just short of it.
    """
    if not series:
        return series
    times = [t for t, _ in series]
    vals = [v for _, v in series]

    # Fill short holes.
    i = 0
    while i < len(vals):
        if vals[i]:
            i += 1
            continue
        j = i
        while j < len(vals) and not vals[j]:
            j += 1
        before = i > 0 and vals[i - 1]
        after = j < len(vals) and vals[j]
        if before and after and (times[min(j, len(times) - 1)] - times[i]) <= max_gap:
            for k in range(i, j):
                vals[k] = True
        i = max(j, i + 1)

    # Extend each trailing edge.
    out = list(vals)
    for i, v in enumerate(vals):
        if not v or (i + 1 < len(vals) and vals[i + 1]):
            continue
        for k in range(i + 1, len(vals)):
            if times[k] - times[i] > extend:
                break
            out[k] = True
    return list(zip(times, out))


# ------------------------------------------------------------- segmenting

def _q(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, round(v / QUANTISE) * QUANTISE) for v in c)  # type: ignore


def _saturation(c: tuple[int, int, int]) -> int:
    return max(c) - min(c)


def plate_for(colour: tuple[int, int, int]) -> tuple[str, bool]:
    """Plate colour for a background, and which logo variant to sit on it.

    The plate is ALWAYS the dominant colour of that part of the frame, so the
    seam disappears on every kind of slide - light grey, white or full-bleed
    orange alike. There is no white card: pasting one onto an orange slide reads
    as a sticker.

    What changes instead is the logo. On a light background the brand colours
    are legible as-is; on a dark or saturated one they are not, so a white
    version of the same logo is used. That is how a broadcast bug behaves.
    """
    hexc = "0x%02X%02X%02X" % colour
    light_logo = _lum(colour) < 200 or _saturation(colour) > 24
    return hexc, light_logo


def segments(series: list[tuple[float, tuple[int, int, int]]],
             window: tuple[float, float] | None = None,
             presence: list[tuple[float, bool]] | None = None) -> list[dict]:
    """Merge the sampled series into runs sharing one plate colour.

    `presence` restricts the runs to times the mark is actually on screen, so no
    plate is ever drawn over a frame that never had a watermark.
    """
    # Presence is sampled coarser than colour (shape matching is heavier), so
    # each colour sample takes the nearest presence sample rather than requiring
    # an exact timestamp match.
    ptimes = [t for t, _ in (presence or [])]
    pvals = [v for _, v in (presence or [])]

    def _present_at(t: float) -> bool:
        if not ptimes:
            return True
        import bisect
        i = bisect.bisect_left(ptimes, t)
        best = min((abs(ptimes[j] - t), j) for j in (i - 1, i, i + 1)
                   if 0 <= j < len(ptimes))[1]
        return pvals[best]
    runs: list[dict] = []
    for t, colour in series:
        if window and not (window[0] <= t <= window[1]):
            continue
        if presence is not None and not _present_at(t):
            continue
        plate, light = plate_for(colour)
        contiguous = runs and (t - runs[-1]["end"]) <= (1.5 / SAMPLE_FPS)
        if runs and runs[-1]["plate"] == plate and contiguous:
            runs[-1]["end"] = t
        else:
            runs.append({"start": t, "end": t, "plate": plate, "light_logo": light})

    def _merge(min_len: float) -> list[dict]:
        out: list[dict] = []
        for r in runs:
            if out and (r["end"] - r["start"]) < min_len:
                out[-1]["end"] = r["end"]
            else:
                out.append(dict(r))
        return out

    min_len = MIN_SEGMENT
    merged = _merge(min_len)
    while len(merged) > MAX_SEGMENTS and min_len < 2.0:
        min_len *= 1.6
        merged = _merge(min_len)
    # No padding. Padding bled each segment into its neighbour, which is exactly
    # how an orange plate ended up on a white slide. Boundaries butt against
    # each other so every frame is covered exactly once.
    # Butt boundaries together only where the runs are actually adjacent; a real
    # gap (the mark absent) must stay a gap.
    for i, r in enumerate(merged):
        if i and (r["start"] - merged[i - 1]["end"]) <= (1.5 / SAMPLE_FPS):
            r["start"] = merged[i - 1]["end"]
        r["end"] = r["end"] + 1.0 / SAMPLE_FPS
    return merged
