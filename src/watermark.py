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

SAMPLE_FPS = 2.0          # 0.5s resolution
GRID_W, GRID_H = 16, 8    # pixels kept per sampled frame
QUANTISE = 6              # colour bucket size; below this a change is invisible
MIN_SEGMENT = 0.5         # seconds; shorter runs are absorbed by their neighbour
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
                  pad_ratio: float = 0.6, limit: float | None = None
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
            for t, px_list in _sample_grid(ff, mp4, _crop(cw, ch, cx, cy), limit=limit)]


def darkness_series(ff: str, mp4: str, box: dict,
                    limit: float | None = None) -> list[tuple[float, float]]:
    """Mean luminance inside a box, for detecting whether the mark is present."""
    grid = _sample_grid(ff, mp4, _crop(box["w"], box["h"], box["x"], box["y"]),
                        limit=limit)
    return [(t, sum(_lum(p) for p in px) / len(px)) for t, px in grid]


# ------------------------------------------------------------- presence

def _lum(c: Iterable[int]) -> float:
    r, g, b = c
    return 0.299 * r + 0.587 * g + 0.114 * b


def detect_presence(box_series: list[tuple[float, float]],
                    bg_series: list[tuple[float, tuple[int, int, int]]],
                    limit: float, margin: float = 6.0) -> tuple[float, float] | None:
    """Time window in which the box is measurably darker than its background.

    Used for the title-card mark, whose duration is not fixed: measuring it per
    video is more robust than hardcoding "the first 8 seconds".
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
    return (max(hits[0] - 0.5, 0.0), end + 1.0)


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


# ------------------------------------------------------------- segmenting

def _q(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(min(255, round(v / QUANTISE) * QUANTISE) for v in c)  # type: ignore


def _saturation(c: tuple[int, int, int]) -> int:
    return max(c) - min(c)


def plate_for(colour: tuple[int, int, int]) -> tuple[str, bool]:
    """Plate colour for a background, and whether the logo needs a card.

    The plate always matches the background so the seam disappears. A saturated
    background additionally needs a card, because the cyan-and-green logo is
    unreadable directly on it.
    """
    hexc = "0x%02X%02X%02X" % colour
    needs_card = _lum(colour) < 200 or _saturation(colour) > 24
    return hexc, needs_card


def segments(series: list[tuple[float, tuple[int, int, int]]],
             window: tuple[float, float] | None = None) -> list[dict]:
    """Merge the sampled series into runs sharing one plate colour."""
    runs: list[dict] = []
    for t, colour in series:
        if window and not (window[0] <= t <= window[1]):
            continue
        plate, card = plate_for(colour)
        if runs and runs[-1]["plate"] == plate:
            runs[-1]["end"] = t
        else:
            runs.append({"start": t, "end": t, "plate": plate, "card": card})

    merged: list[dict] = []
    for r in runs:
        if merged and (r["end"] - r["start"]) < MIN_SEGMENT:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(r)
    for i, r in enumerate(merged):
        r["start"] = (window[0] if window else 0.0) if i == 0 else max(
            r["start"] - 0.25, merged[i - 1]["end"])
        r["end"] = r["end"] + 0.25
    return merged
