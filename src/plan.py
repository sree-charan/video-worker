"""Prompt engine. Three chat rounds, then one video generation.

Chat is 500/day on Pro; video is 20/day. So we spend three cheap rounds building
a narration spec that has already been audited, and only then spend the video.

  Round 1  STRUCTURE  - the course file's own headings and terms, VERBATIM.
                        Also reports which syllabus topics the book does not
                        actually cover, and whether detailed notes exist at all.
  Round 2  SUBSTANCE  - per section: the book's definition, mechanism, one
                        concrete specific, one example step.
  Round 3  SCRUTINY   - audits rounds 1-2 against the syllabus, deletes
                        anything that is filler or repetition, and allocates a
                        per-section second budget weighted by dependency depth.

Round 1 exists mainly to solve a specific observed failure: left alone, the
model renames sections to catchier synonyms ("Access Control Rules" for the
book's "Member access rules"). Chapter labels and on-screen headings must match
what the student sees in their own course file, so headings are extracted
verbatim in round 1 and then locked as a hard constraint in the video prompt.

Every round speaks a line-prefix protocol so the output is parseable and a
chatty model cannot smuggle prose into the next stage.
"""

from __future__ import annotations

import re
from typing import Any

# =========================================================================
# Round 1 - structure, verbatim
# =========================================================================

ROUND1 = """\
You are indexing a college course file. Report only what is physically printed
in it. Do not improve, modernise or rename anything.

Scope: UNIT-{n} only, titled "{title}".
The syllabus scope for this unit is:
{topics}

Ignore these parts of the course file entirely: {excludes}.

Answer using ONLY these line formats, one per line, nothing else:

NOTES_PRESENT | yes or no
  yes only if the "Detailed notes" section contains real explanatory content
  for this unit, not just a topic list.

SECTION | <k> | <heading EXACTLY as printed in the course file>
  One line per teachable section of this unit, in the order the book presents
  them. Copy the heading character for character, including its capitalisation
  and any punctuation. Do not tidy it. Do not translate it. Do not merge two
  headings. If the book writes "Member access rules", write exactly that.
  Aim for 6 to 10 sections.

TERM | <technical term EXACTLY as the book spells it>
  One line per term the book introduces in this unit. Use the book's spelling
  even if it is unusual.

MISSING | <syllabus topic that the detailed notes do NOT actually explain>
  One line per gap. Write "MISSING | NONE" if the book covers everything.

Do not output any other line. No preamble, no summary, no commentary.
"""


def round1_prompt(syl: dict, unit: dict) -> str:
    return ROUND1.format(
        n=unit["n"],
        title=unit["title"],
        topics=bullets(unit["topics"]),
        excludes="; ".join(syl["subject"].get("exclude_sections", [])) or "none",
    )


# =========================================================================
# Round 2 - substance per section
# =========================================================================

ROUND2 = """\
For UNIT-{n} of this course file, expand each section below into teachable
substance. Use the course file's "Detailed notes" as the source of truth. Where
the notes are thin for a section, say so rather than inventing depth.

The sections, with their locked headings:
{sections}

Answer using ONLY these line formats:

DEF | <k> | <the definition the book gives, in one sentence, using the book's own terminology>
MECH | <k> | <how it actually works, one or two sentences. Mechanism, not motivation.>
SPEC | <k> | <one concrete specific printed in the book: a rule, a syntax form, a numbered classification, a named method, a constraint. If the book gives none, write NONE.>
STEP | <k> | <how the running example advances at this section, one sentence>
THIN | <k> | <yes if the book's notes for this section are too thin to fill its share of the video, else no>

Rules: no preamble. Do not restate the headings. Do not write anything that is
not one of the five line types above. Never use a synonym for a term that
appears in the locked headings.
"""


def round2_prompt(unit: dict, sections: list[str], example: str) -> str:
    listing = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(sections))
    return ROUND2.format(n=unit["n"], sections=listing) + (
        f"\nThe running example for this unit, which STEP must advance:\n{clean(example)}\n"
    )


# =========================================================================
# Round 3 - audit and time budget
# =========================================================================

ROUND3 = """\
Audit the lecture plan for UNIT-{n} below before it is turned into a {minutes}
minute video. Be strict. Your job is to catch weakness, not to be agreeable.

{plan}

Syllabus scope this plan must fully cover:
{topics}

Perform four checks, then output the final spec.

CHECK 1 - Coverage. Every syllabus topic must map to at least one section.
CHECK 2 - Redundancy. If two sections explain the same idea, merge them.
CHECK 3 - Substance. Delete any point that is motivational, definitional
          padding, or a restatement of a point already made.
CHECK 4 - Weight. Sections that later sections depend on deserve more time.
          Terminal or purely descriptive sections deserve less. The budget must
          total {seconds} seconds. Give no section less than 25 seconds.

Then answer using ONLY these line formats:

TOTAL | {seconds}
SECTION | <k> | <heading, copied EXACTLY from the plan above, unchanged> | <seconds>
POINT | <k> | <one sentence that must be said in this section>
  Two to four POINT lines per section. Each must carry new information.
SPEC | <k> | <the concrete specific from the book to state on screen, or NONE>
STEP | <k> | <how the running example advances here, or NONE>
GAP | <syllabus topic still not covered after your merges, or NONE>

You may renumber sections after merging, but you may NOT reword any heading.
No preamble, no commentary, no closing summary.
"""


def round3_prompt(unit: dict, plan_text: str, minutes: int) -> str:
    return ROUND3.format(
        n=unit["n"],
        minutes=minutes,
        seconds=minutes * 60,
        plan=plan_text.strip(),
        topics=bullets(unit["topics"]),
    )


# =========================================================================
# Deep Research - only when the book is too thin
# =========================================================================

RESEARCH = """\
Authoritative teaching material for an undergraduate engineering course, on
exactly these topics: {gaps}.
Context: {subject}, unit {n} - {title}. Audience: Indian B.Tech students.
Prefer university lecture notes, standard textbook chapters and official
language or platform documentation. Exclude blog posts, forum answers, exam
dumps and cheat sheets.
"""


def research_prompt(syl: dict, unit: dict, gaps: list[str]) -> str:
    return RESEARCH.format(
        gaps="; ".join(gaps),
        subject=syl["subject"]["title"],
        n=unit["n"],
        title=unit["title"],
    )


# =========================================================================
# The video prompt
# =========================================================================

VIDEO = """\
Lecture {n} of {total} of "{subject}" ({branch}, year {year}, semester {sem}).
Audience: B.Tech students who will be examined on this unit from this exact
course file.

Deliver a complete treatment of the unit in approximately {minutes} minutes.

## TERMINOLOGY LOCK - non-negotiable

Use these headings verbatim, character for character, both as the on-screen
section heading and when naming the topic aloud:

{headings}

Do not paraphrase, shorten, expand, prettify, translate or substitute a synonym
for any heading. Do not invent a more engaging title. If a heading reads
"Member access rules", the on-screen heading must read "Member access rules" -
not "Access Control Rules", not "Accessing Members", not "Member Access".

Use the course file's spelling for every technical term:
{terms}

A student watching this must be able to match each section to their own course
file by name.

## SECTIONS, IN THIS ORDER, WITH TIME BUDGETS

{sections}

Treat each budget as a target. A section budgeted 120 seconds must not be
delivered in 30.

## USE THE FULL DURATION

Total target: {minutes} minutes. Do not finish early. If you find yourself
running short, add depth to the highest-budget sections - a further
consequence, a second constraint, a harder case - never padding, never
repetition, never a recap.

{continuity}

## RUNNING EXAMPLE

{example}
{code_rule}
Advance this same example as each new section arrives. Never abandon it and
start an unrelated example.

## OPENING - about 20 seconds

Open with one precise engineering question this unit answers. Then go straight
into section 1. No greeting, no channel introduction, no statement of what the
video will cover, no mention of the college, department or syllabus document.

## CLOSING - about 40 seconds

Connect every section into one framework, so the student sees how the unit fits
together. State these exam-critical distinctions as direct contrasts:
{exam_focus}
Do not re-explain anything. Connect only.

## INFORMATION DENSITY - highest priority

Every sentence must do one of: introduce a concept, explain a mechanism,
connect two concepts, state why something exists, or give one concrete example.
A sentence that does none of these must be cut.

Never say, in any wording: "let's understand", "now that we know", "before
moving ahead", "as we discussed", "as we saw earlier", "let us now move on to",
"you might be wondering", "imagine this", "think about it", "it is important to
note", "interestingly", "in this video we will", "by the end of this video".

No agenda preview. No recap of what was just said. No motivational or
encouraging language. No study advice. No dramatic pause or rhetorical question
after the opening. No repeating an idea in different words. No re-defining a
term already defined.

{math_rule}

## STYLE

An experienced engineering educator who respects the student's intelligence and
is short on time. Declarative sentences. Define each term precisely on first
use, then use it freely. One high-quality real example per major concept, not
three. Prefer a concrete mechanism over an analogy; if an analogy is used, one
sentence, then drop it. Clarity over entertainment. The result should feel like
a precise technical documentary, not a classroom lecture, podcast or motivational talk.
"""

CODE_RULE = "Show real, correct {lang} code on screen for it.\n"
MATH_NONE = ("## MATHEMATICS\n\nDo not derive formulas and do not solve numerical problems. "
             "State a relationship and explain what it means physically.")
MATH_WORKED = ("## MATHEMATICS\n\nShow the steps that carry meaning. Skip algebraic manipulation "
               "that teaches nothing. At most one worked numeric case per section.")

FIRST_CONTINUITY = """\
## CONTINUITY

This is the first unit. Assume no prior knowledge of the subject, but assume
competence in school mathematics and basic programming. Do not summarise the
course as a whole.
"""

LATER_CONTINUITY = """\
## CONTINUITY

{delivered} of this course {has} already been delivered, covering:
{covered}

The student has seen all of it. Do not re-teach or re-define any of it. Open by
connecting from unit {prev} in a single clause, then proceed to new material.
Where this unit builds on an earlier concept, name it in passing and move on.
"""


def video_prompt(syl: dict, unit: dict, spec: dict, minutes: int) -> str:
    subj = syl["subject"]
    units = syl["units"]
    ped = syl.get("pedagogy", {})
    prev_units = [u for u in units if u["n"] < unit["n"]]

    if prev_units:
        prev = unit["n"] - 1
        continuity = LATER_CONTINUITY.format(
            delivered="Unit 1" if prev == 1 else f"Units 1 to {prev}",
            has="has" if prev == 1 else "have",
            prev=prev,
            covered="\n".join(f"  - unit {u['n']}: {u['title']}" for u in prev_units),
        )
    else:
        continuity = FIRST_CONTINUITY

    lines = []
    for s in spec["sections"]:
        lines.append(f'### {s["k"]}. {s["heading"]}  [{s["seconds"]}s]')
        for p in s["points"]:
            lines.append(f"    - {p}")
        if s.get("spec"):
            lines.append(f"    - state on screen: {s['spec']}")
        if s.get("step"):
            lines.append(f"    - example advances: {s['step']}")
        lines.append("")

    return VIDEO.format(
        n=unit["n"],
        total=len(units),
        subject=subj["title"],
        branch=subj["branch"],
        year=subj["year"],
        sem=subj["semester"],
        minutes=minutes,
        headings="\n".join(f'  {s["k"]}. "{s["heading"]}"' for s in spec["sections"]),
        terms=", ".join(spec.get("terms", [])) or "(as printed in the course file)",
        sections="\n".join(lines).strip(),
        continuity=continuity.strip(),
        example=clean(unit["example"]),
        code_rule=CODE_RULE.format(lang=ped["code_language"]) if ped.get("code_language") else "",
        exam_focus="\n".join(f"  - {i}" for i in unit.get("exam_focus", [])) or "  - (none)",
        math_rule=MATH_WORKED if ped.get("math") == "worked" else MATH_NONE,
    )


# =========================================================================
# Parsers for the line-prefix protocol
# =========================================================================

def _fields(line: str) -> list[str]:
    return [p.strip() for p in line.split("|")]


def parse_round1(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {"notes_present": True, "sections": [], "terms": [], "missing": []}
    for raw in text.splitlines():
        f = _fields(raw.strip())
        if len(f) < 2:
            continue
        tag = f[0].upper()
        if tag == "NOTES_PRESENT":
            out["notes_present"] = f[1].lower().startswith("y")
        elif tag == "SECTION" and len(f) >= 3:
            out["sections"].append(f[2])
        elif tag == "TERM":
            out["terms"].append(f[1])
        elif tag == "MISSING" and f[1].upper() != "NONE":
            out["missing"].append(f[1])
    # A heading repeated verbatim is a model slip, not two sections.
    seen, uniq = set(), []
    for s in out["sections"]:
        if s and s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
    out["sections"] = uniq
    return out


def parse_round3(text: str, fallback_headings: list[str], total_seconds: int) -> dict[str, Any]:
    sections: dict[str, dict] = {}
    order: list[str] = []
    gaps: list[str] = []

    for raw in text.splitlines():
        f = _fields(raw.strip())
        if len(f) < 2:
            continue
        tag = f[0].upper()
        if tag == "SECTION" and len(f) >= 4:
            k = f[1]
            sections.setdefault(k, {"k": k, "heading": f[2], "points": [], "spec": "", "step": ""})
            sections[k]["heading"] = f[2]
            sections[k]["seconds"] = _int(f[3], 60)
            if k not in order:
                order.append(k)
        elif tag == "POINT" and len(f) >= 3:
            sections.setdefault(f[1], {"k": f[1], "heading": "", "points": [], "spec": "", "step": ""})
            sections[f[1]]["points"].append(f[2])
            if f[1] not in order:
                order.append(f[1])
        elif tag in ("SPEC", "STEP") and len(f) >= 3 and f[2].upper() != "NONE":
            sections.setdefault(f[1], {"k": f[1], "heading": "", "points": [], "spec": "", "step": ""})
            sections[f[1]][tag.lower()] = f[2]
        elif tag == "GAP" and f[1].upper() != "NONE":
            gaps.append(f[1])

    ordered = [sections[k] for k in order if sections.get(k, {}).get("heading")]

    if not ordered:
        # Round 3 misbehaved. Fall back to round 1's verbatim headings with an
        # even split, so the terminology lock still holds.
        per = max(total_seconds // max(len(fallback_headings), 1), 25)
        ordered = [{"k": str(i + 1), "heading": h, "points": [], "spec": "", "step": "",
                    "seconds": per} for i, h in enumerate(fallback_headings)]

    for i, s in enumerate(ordered, 1):
        s["k"] = str(i)
        s.setdefault("seconds", 60)

    return {"sections": ordered, "gaps": gaps,
            "budgeted": sum(s["seconds"] for s in ordered)}


def _int(v: str, default: int) -> int:
    m = re.search(r"\d+", v or "")
    return int(m.group(0)) if m else default


def plan_text(sections: list[str], round2: str) -> str:
    """Stitch rounds 1 and 2 into the artefact round 3 audits."""
    listing = "\n".join(f"SECTION | {i + 1} | {s}" for i, s in enumerate(sections))
    return f"{listing}\n\n{round2.strip()}"


def chapter_labels(spec: dict) -> list[str]:
    """Chapter labels ARE the locked headings, so the app matches the course file."""
    return [s["heading"] for s in spec["sections"]]


# =========================================================================

def clean(text: str) -> str:
    return " ".join(str(text).split())


def bullets(topics: str) -> str:
    parts = [p.strip() for p in clean(topics).replace(";", ".").split(".") if p.strip()]
    return "\n".join(f"  - {p}" for p in parts)
