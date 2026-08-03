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
from typing import Iterable

# Frame-accurate. At 2fps the plate colour changed up to half a second before
# the background did, so an orange plate showed on a white slide - clearly
# visible. Sampling a tiny crop is cheap enough to do at video frame rate.
SAMPLE_FPS = 24.0
GRID_W, GRID_H = 16, 8    # pixels kept per sampled frame
QUANTISE = 6              # colour bucket size; below this a change is invisible
MIN_SEGMENT = 0.2         # seconds; shorter runs are absorbed by their neighbour
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
                 fps: float) -> list[tuple[float, "np.ndarray"]]:
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
    present = dict(presence or [])
    runs: list[dict] = []
    for t, colour in series:
        if window and not (window[0] <= t <= window[1]):
            continue
        if presence is not None and not present.get(t, False):
            continue
        plate, light = plate_for(colour)
        contiguous = runs and (t - runs[-1]["end"]) <= (1.5 / SAMPLE_FPS)
        if runs and runs[-1]["plate"] == plate and contiguous:
            runs[-1]["end"] = t
        else:
            runs.append({"start": t, "end": t, "plate": plate, "light_logo": light})

    merged: list[dict] = []
    for r in runs:
        if merged and (r["end"] - r["start"]) < MIN_SEGMENT:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(r)
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
