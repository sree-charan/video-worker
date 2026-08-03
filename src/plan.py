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
  One line per teachable section, in the order the book presents them.
  Return between {lo} and {hi} sections.

  A section is ONE concept, not a list of concepts. The syllabus above is
  written as long comma-separated lines; those are NOT headings. If a heading
  you are about to write contains more than two commas, or runs longer than
  about eight words, you are copying a syllabus line instead of finding the
  book's own sub-heading. Go into the Detailed notes and use the actual
  sub-heading printed above that explanation.

  Copy each heading character for character, including capitalisation and
  punctuation. Do not tidy it, translate it, or merge two headings. If the book
  writes "Member access rules", write exactly that.

TERM | <technical term EXACTLY as the book spells it>
  One line per term the book introduces in this unit. Use the book's spelling
  even if it is unusual.

MISSING | <syllabus topic that the detailed notes do NOT actually explain>
  One line per gap. Write "MISSING | NONE" if the book covers everything.

Do not output any other line. No preamble, no summary, no commentary.
"""

ROUND1_RETRY = """\
Your previous answer was rejected.

{problem}

Re-answer using the same line formats. Return between {lo} and {hi} SECTION
lines, each naming a single concept in eight words or fewer, copied verbatim
from a sub-heading inside the course file's Detailed notes for UNIT-{n}.

This is the granularity required - short, one concept each:
{good}

These would be rejected, because each is a syllabus line rather than a heading
printed in the book:
{bad}
"""

# A heading with more commas or words than this is a syllabus line, not a
# heading the book actually prints above an explanation.
MAX_HEADING_COMMAS = 2
MAX_HEADING_WORDS = 10
MIN_SECTIONS = 6
MAX_SECTIONS = 14


def round1_prompt(syl: dict, unit: dict) -> str:
    return ROUND1.format(
        n=unit["n"],
        title=unit["title"],
        topics=bullets(unit["topics"]),
        excludes="; ".join(syl["subject"].get("exclude_sections", [])) or "none",
        lo=MIN_SECTIONS + 2,
        hi=MAX_SECTIONS,
    )


def round1_retry_prompt(unit: dict, problem: str) -> str:
    """Illustrate granularity with this unit's OWN topics.

    Hardcoded examples would have to come from some subject, and examples from
    Java are noise when the subject is Basic Electrical Engineering. Taking the
    short comma-separated fragments as "good" and the full syllabus sentences as
    "bad" is both subject-correct and exactly the distinction being taught: the
    bad examples are the lines the model just echoed back.
    """
    flat = clean(unit["topics"]).replace(";", ".")
    sentences = [x.strip() for x in flat.split(".") if x.strip()]
    fragments = [q.strip() for s in sentences for q in s.split(",")
                 if 1 <= len(q.split()) <= 6]
    good = "\n".join(f"  SECTION | {i + 1} | {t}"
                      for i, t in enumerate(fragments[:3])) or \
           "  SECTION | 1 | <one concept, in six words or fewer>"
    bad = "\n".join(f"  SECTION | {i + 1} | {s[:110]}"
                     for i, s in enumerate(sentences[:2])) or \
          "  SECTION | 1 | <a whole comma-separated syllabus line>"
    return ROUND1_RETRY.format(problem=problem, n=unit["n"],
                               lo=MIN_SECTIONS + 2, hi=MAX_SECTIONS,
                               good=good, bad=bad)


def audit_round1(r1: dict) -> str | None:
    """Return a description of what is wrong with round 1, or None if it passes.

    Guards the specific failure mode where the model echoes the syllabus line
    back as a heading, which produces two or three enormous sections and hands
    each of them four minutes of runtime - the exact place padding reappears.
    """
    problems = []
    broad = [h for h in r1["sections"]
             if h.count(",") > MAX_HEADING_COMMAS or len(h.split()) > MAX_HEADING_WORDS]
    if len(r1["sections"]) < MIN_SECTIONS:
        problems.append(
            f"You returned only {len(r1['sections'])} sections. A 12 minute video "
            f"needs finer granularity than that."
        )
    if broad:
        problems.append(
            "These headings are syllabus lines, not headings printed in the book:\n"
            + "\n".join(f"  - {h}" for h in broad[:4])
        )
    return "\n\n".join(problems) if problems else None



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
Audit the UNIT-{n} lecture plan you produced in this conversation, before it is
turned into a {minutes} minute video. Be strict. Your job is to catch weakness,
not to be agreeable.

The plan's sections, which you must keep referring to by these exact headings:
{headings}

Use the DEF, MECH, SPEC and STEP lines you already gave for each of them.

Syllabus scope this plan must fully cover:
{topics}

Perform four checks, then output the final spec.

CHECK 1 - Coverage. Every syllabus topic must map to at least one section.
CHECK 2 - Redundancy. If two sections explain the same idea, merge them.
CHECK 3 - Substance. Delete any point that is motivational, definitional
          padding, or a restatement of a point already made.
CHECK 4 - Weight. Sections that later sections depend on deserve more time.
          Terminal or purely descriptive sections deserve less. The budget must
          total {seconds} seconds. Give no section less than 25 seconds and no
          section more than 150 seconds - a single heading held for longer than
          that becomes padding.

Then answer using ONLY these line formats:

TOTAL | {seconds}
SECTION | <k> | <heading, copied EXACTLY from the list above, unchanged> | <seconds>
POINT | <k> | <one sentence that must be said in this section>
  Two to four POINT lines per section. Each must carry new information.
SPEC | <k> | <the concrete specific from the book to state on screen, or NONE>
STEP | <k> | <how the running example advances here, or NONE>
GAP | <syllabus topic still not covered after your merges, or NONE>

You may renumber sections after merging, but you may NOT reword any heading.
No preamble, no commentary, no closing summary.
"""

# The chat endpoint rejects over-long questions ("status 3"). Round 1 at ~2.7k
# characters is accepted; round 3 embedding rounds 1-2 verbatim at ~9.9k is not.
# Rounds therefore lean on conversation continuity instead of re-sending the
# plan, and this cap is enforced before spending the call.
MAX_PROMPT_CHARS = 6000


def round3_prompt(unit: dict, headings: list[str], minutes: int) -> str:
    topics = bullets(unit["topics"])
    listing = "\n".join(f"  {i + 1}. {h}" for i, h in enumerate(headings))
    prompt = ROUND3.format(
        n=unit["n"], minutes=minutes, seconds=minutes * 60,
        headings=listing, topics=topics,
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        # Topic list is the only expendable part; the headings and the protocol
        # are both load-bearing.
        keep = MAX_PROMPT_CHARS - (len(prompt) - len(topics))
        prompt = ROUND3.format(
            n=unit["n"], minutes=minutes, seconds=minutes * 60,
            headings=listing, topics=topics[:max(keep, 200)].rstrip() + "\n  - ...",
        )
    return prompt


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
{required}

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

Every sentence must do one of: introduce an idea, explain how something works,
connect two ideas, say why something exists, or give one concrete example.
A sentence that does none of these must be cut. This rule and the LANGUAGE rules
below apply together: short plain sentences that each still teach something.

Never say, in any wording: "let's understand", "now that we know", "before
moving ahead", "as we discussed", "as we saw earlier", "let us now move on to",
"you might be wondering", "imagine this", "think about it", "it is important to
note", "interestingly", "in this video we will", "by the end of this video".

No agenda preview. No recap of what was just said. No motivational or
encouraging language. No study advice. No dramatic pause or rhetorical question
after the opening. No repeating an idea in different words. No re-defining a
term already defined.

{math_rule}

## LANGUAGE - treat this as important as the content

{audience}
Every sentence must be understood the first time it is heard, without pausing or
rewinding.

- Keep sentences short. Aim for about 12 words. Never go past 20.
- One idea per sentence. If a sentence has two ideas, split it.
- Always choose the plainest word that is still correct.{plain_swaps}
- A technical term may only appear after you have said what it means in plain
  words, in the same breath.{term_example} Never introduce two new terms in one
  sentence.
- Never use these words: {banned}.
- Do not stack adjectives in front of a noun.{adjective_example}
- Use the active voice and the present tense: say what a thing does, rather than
  what is done to it.
- Use a real number, a real value or a real worked line instead of an abstract
  description wherever you can.

Simple words are NOT the same as filler. Every sentence must still teach
something. Say the same substance in words a beginner already knows.

Before you say any sentence, ask: would a nineteen-year-old hearing this once
understand it? If not, say it shorter and plainer.

## STYLE

A good teacher explaining something to one student, out loud, at a whiteboard.
Warm but efficient. Clear over clever. One real example per idea, not three. If
you use a comparison to everyday life, keep it to one sentence and then drop it.
The result should feel like the clearest explanation the student has ever heard
of this topic - not like a textbook read aloud.
"""

# Subject-independent. Every one of these appeared in, or is of a kind with, the
# jargon that made the first pilot unreadable (grade level 18).
BANNED_WORDS = [
    "paradigm", "artifact", "mechanism", "entity", "architecture",
    "conceptualise", "systematically", "inherent", "robust", "leverage",
    "facilitate", "cohesive", "singular", "fundamentally", "uniformly",
    "utilise", "comprise", "delineate", "myriad", "plethora", "vis-a-vis",
]


ORDINALS = {"I": "first", "1": "first", "II": "second", "2": "second",
            "III": "third", "3": "third", "IV": "fourth", "4": "fourth"}


def year_word(year: str) -> str:
    """Syllabuses record the year as a Roman numeral; prompts need a word."""
    return ORDINALS.get(str(year).strip().upper(), str(year))


def language_block(syl: dict, unit: dict) -> dict[str, str]:
    """Render the language rules for THIS subject.

    The rules themselves are universal - sentence length, one idea per sentence,
    term-then-meaning, no stacked adjectives. What differs per subject is the
    vocabulary: telling a Basic Electrical Engineering narration to prefer "make"
    over "instantiate" is noise, and telling a Java narration to avoid
    "electromotive force" is too.

    So `pedagogy.language` in the syllabus supplies the subject's own jargon
    swaps, extra banned words and illustrations; everything here has a neutral
    default so a syllabus that omits the block still gets the universal rules.
    """
    subj = syl["subject"]
    ped = syl.get("pedagogy", {})
    lang = ped.get("language", {}) or {}

    code = ped.get("code_language")
    if lang.get("audience"):
        audience = clean(lang["audience"])
    elif code:
        audience = (f"Your listener is a {year_word(subj['year'])}-year engineering "
                    f"student who has never written a line of {code}, and whose first "
                    f"language is usually not English.")
    else:
        audience = (f"Your listener is a {year_word(subj['year'])}-year engineering "
                    f"student meeting {subj['title']} for the first time, and whose "
                    f"first language is usually not English.")

    swaps = lang.get("plain_words") or {}
    swaps_text = ""
    if swaps:
        pairs = ", ".join(f'"{plain}" not "{jargon}"' for jargon, plain in swaps.items())
        swaps_text = f" For this subject: {pairs}."

    term_example = ""
    if lang.get("term_example"):
        term_example = f' Like this: "{clean(lang["term_example"])}"'

    adj = lang.get("adjective_example") or {}
    adj_example = ""
    if adj.get("bad") and adj.get("good"):
        adj_example = (f' "{clean(adj["bad"])}" is wrong. '
                       f'"{clean(adj["good"])}" is right.')

    banned = BANNED_WORDS + list(lang.get("banned_words") or [])
    return {
        "audience": audience,
        "plain_swaps": swaps_text,
        "term_example": term_example,
        "adjective_example": adj_example,
        "banned": ", ".join(banned),
    }

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


REQUIRED_EXTRA = """
## REQUIRED ADDITIONAL COVERAGE

These syllabus topics are examinable but the course file does not explain them,
so no section above owns them:
{gaps}

Cover each one anyway, folded into whichever section above it belongs to. One or
two precise sentences each is enough - a correct definition and its mechanism.
Do not create a separate section for them and do not skip them.
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

    gaps = spec.get("gaps") or []
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
        required=(REQUIRED_EXTRA.format(gaps="\n".join(f"  - {g}" for g in gaps))
                  if gaps else ""),
        continuity=continuity.strip(),
        example=clean(unit["example"]),
        code_rule=CODE_RULE.format(lang=ped["code_language"]) if ped.get("code_language") else "",
        exam_focus="\n".join(f"  - {i}" for i in unit.get("exam_focus", [])) or "  - (none)",
        math_rule=MATH_WORKED if ped.get("math") == "worked" else MATH_NONE,
        **language_block(syl, unit),
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


def chapter_labels(spec: dict) -> list[str]:
    """Chapter labels ARE the locked headings, so the app matches the course file."""
    return [s["heading"] for s in spec["sections"]]


# =========================================================================

def clean(text: str) -> str:
    return " ".join(str(text).split())


def bullets(topics: str) -> str:
    """Topic list for the prompts, split fine enough to be useful.

    Course-file syllabus lines are long comma-separated runs; splitting only on
    sentences yields two or three enormous bullets, which invites the model to
    treat each as one section. Long segments are split again on commas.
    """
    parts = [p.strip() for p in clean(topics).replace(";", ".").split(".") if p.strip()]
    fine: list[str] = []
    for p in parts:
        if len(p.split()) > 12 and "," in p:
            fine += [q.strip() for q in p.split(",") if q.strip()]
        else:
            fine.append(p)
    return "\n".join(f"  - {p}" for p in fine)
