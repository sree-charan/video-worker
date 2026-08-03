"""Prompt construction. This is where video quality is actually decided.

Two prompts are built per unit:

  1. OUTLINE prompt  -> spent on `notebooklm ask` (cheap: 500 chats/day on Pro)
     Pulls the unit's real substance out of the course file's Detailed Notes and
     returns a compact beat sheet. This exists because a video steered only by
     syllabus keywords is generic: the model has to guess what depth to go to.
     Feeding it the book's own explanation of those keywords is what makes the
     output specific to this course rather than to Java in general.

  2. VIDEO prompt    -> spent on `generate video` (expensive: 20/day on Pro)
     Wraps that beat sheet in hard structural and anti-filler constraints.

The asymmetry is deliberate: burn abundant chat quota to protect scarce video
quota, so a video is generated once and doesn't need regenerating.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Stage 1: extract the unit's substance from the book.
# --------------------------------------------------------------------------

OUTLINE_PROMPT = """\
You are preparing a lecture script outline from this course file.

Restrict yourself to UNIT-{n} only: "{title}".
The syllabus scope for this unit is exactly:
{topics}

Use ONLY the course file's "Detailed notes" and "Additional topics" sections as
evidence. Ignore entirely: {excludes}.

Produce a beat sheet with 6 to 8 beats. For each beat give:
  BEAT <k>: <the concept, named in the book's own terminology>
  DEFINE: <one sentence, the precise definition the book uses>
  MECHANISM: <one or two sentences on how it actually works, not why it matters>
  BOOK_DETAIL: <one concrete specific from the course file - a rule, a syntax
    form, a constraint, a named method, a numbered classification. If the book
    gives no specific for this beat, write NONE.>

Then:
  EXAM_CRITICAL: <the three distinctions from this unit that the question bank
    and previous university papers in this course file actually test>

Rules: no preamble, no restatement of this instruction, no motivational text.
Do not mention that you are an AI or that you were given a course file.
Order the beats in teaching order, simplest dependency first.
"""


def outline_prompt(syl: dict[str, Any], unit: dict[str, Any]) -> str:
    excludes = "; ".join(syl["subject"].get("exclude_sections", [])) or "none"
    return OUTLINE_PROMPT.format(
        n=unit["n"],
        title=unit["title"],
        topics=_bullets(unit["topics"]),
        excludes=excludes,
    )


# --------------------------------------------------------------------------
# Stage 2: the video steering prompt.
# --------------------------------------------------------------------------

VIDEO_PROMPT = """\
Lecture {n} of {total} for "{subject}" ({branch}, year {year}, semester {sem}).
Target length: about {minutes} minutes. Audience: engineering undergraduates who
will be examined on this unit.

COVER EXACTLY THIS, IN THIS ORDER:
{beats}

{continuity}

ONE WORKED EXAMPLE, CARRIED THROUGHOUT:
{example}
Show real Java code on screen for it. Extend the same example as new concepts
arrive - never abandon it and start a different example.

CLOSE WITH these three distinctions, stated as contrasts:
{exam_focus}

STRUCTURE:
- Open with a single sentence naming what the student will be able to do after
  this unit. No greeting, no channel intro, no "in this video we will".
- Then the beats above, in order. Definition first, mechanism second, then the
  example advanced by one step.
- No section may restate a definition already given earlier in this video.

DO NOT INCLUDE: welcome messages, agenda slides that preview what is coming,
recaps of what was just said, "as we saw earlier", "let us now move on to",
study advice, encouragement, filler transitions, or any discussion of the
college, department, syllabus document, question papers or timetables.

TONE: an expert lecturer with limited time who respects the student's
intelligence. Declarative sentences. Define each technical term precisely on
first use, then use it without re-explaining. Prefer a concrete mechanism over
an analogy. Silence is better than filler.
"""

FIRST_UNIT_CONTINUITY = """\
CONTINUITY: This is the first unit of the course. Assume no prior Java, but do
assume competence in basic programming. Do not summarise the course as a whole.
"""

LATER_UNIT_CONTINUITY = """\
CONTINUITY: {delivered} of this course {has} already been delivered and covered:
{covered}.
The student has seen all of that. Do not re-teach or re-define any of it. Open
by connecting from unit {prev} in one sentence, then proceed immediately to new
material. Where this unit builds on an earlier concept, reference it by name in
passing rather than explaining it again.
"""


def video_prompt(syl: dict[str, Any], unit: dict[str, Any], beats: str,
                 minutes: int = 10) -> str:
    subj = syl["subject"]
    units = syl["units"]
    prev_units = [u for u in units if u["n"] < unit["n"]]

    if not prev_units:
        continuity = FIRST_UNIT_CONTINUITY
    else:
        prev = unit["n"] - 1
        delivered = "Unit 1" if prev == 1 else f"Units 1 to {prev}"
        covered = "; ".join(f"unit {u['n']} - {u['title']}" for u in prev_units)
        continuity = LATER_UNIT_CONTINUITY.format(
            delivered=delivered,
            has="has" if prev == 1 else "have",
            prev=prev,
            covered=covered,
        )

    return VIDEO_PROMPT.format(
        n=unit["n"],
        total=len(units),
        subject=subj["title"],
        branch=subj["branch"],
        year=subj["year"],
        sem=subj["semester"],
        minutes=minutes,
        beats=beats.strip(),
        continuity=continuity.strip(),
        example=_clean(unit["example"]),
        exam_focus=_bullets_list(unit.get("exam_focus", [])),
    )


# --------------------------------------------------------------------------
# Beat sheet -> prompt-ready text, and chapter labels.
# --------------------------------------------------------------------------

def beats_from_outline(outline: str, fallback_topics: str) -> str:
    """Reduce the chat outline to the lines the video prompt needs.

    We keep BEAT/DEFINE/MECHANISM/BOOK_DETAIL and drop everything else, so a
    chatty model cannot smuggle filler into the video prompt.
    """
    keep = ("BEAT", "DEFINE", "MECHANISM", "BOOK_DETAIL")
    lines = [ln.strip() for ln in outline.splitlines() if ln.strip()]
    picked = [ln for ln in lines if ln.upper().startswith(keep)]
    picked = [ln for ln in picked if not ln.upper().endswith("NONE")]
    if len(picked) < 4:
        # Outline stage failed or returned prose; fall back to raw syllabus so
        # the pipeline still produces something rather than nothing.
        return _bullets(fallback_topics)
    return "\n".join(picked)


def chapter_labels(outline: str, unit: dict[str, Any]) -> list[str]:
    """Human-facing chapter titles, taken from the beat sheet when available."""
    labels: list[str] = []
    for ln in outline.splitlines():
        s = ln.strip()
        if s.upper().startswith("BEAT"):
            _, _, rest = s.partition(":")
            rest = rest.strip(" -\u2013")
            if rest:
                labels.append(rest[:60])
    return labels or [a.title() for a in unit.get("anchors", [])]


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def _bullets(topics: str) -> str:
    parts = [p.strip() for p in _clean(topics).replace(";", ".").split(".") if p.strip()]
    return "\n".join(f"- {p}" for p in parts)


def _bullets_list(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) or "- (none specified)"
