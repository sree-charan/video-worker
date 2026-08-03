# video-worker

Turns a college course file into unit-wise lecture videos, captions, chapters
and catalog JSON for the GCTC Portal app — driving Gemini Notebook (NotebookLM)
through [`notebooklm-py`](https://github.com/teng-lin/notebooklm-py) from GitHub
Actions.

Sibling of `infographic-worker`, same auth model.

## How it runs

```
syllabus/*.yml ──► fire ──► (Gemini Notebook, 20-30 min) ──► collect ──► postprocess ──► publish
                    │                                            │
                    └────────────── state/manifest.json ─────────┘
```

`fire` and `collect` are separate workflows on purpose. An explainer video takes
20–30+ minutes to generate. Blocking a runner on that would cost ~30 minutes of
Actions time per video; at 60 videos/day it would be ~1800 minutes/day. So `fire`
starts generation and exits, and `collect` runs on a 20-minute cron, polling
cheaply and downloading whatever finished.

`state/manifest.json` is the only durable state. Every stage reads it, mutates
one unit, writes it back. That makes the whole pipeline resumable — necessary,
because the CLI drives a consumer product whose session can expire or rate-limit
mid-batch.

## Quota

Per Google's published limits, a **Pro** account gets **20 Video Overviews/day**,
so three accounts give 60/day. Three accounts = three `notebooklm` profiles;
`--profile` decides whose quota a unit spends.

Cinematic is capped at **2/day** on Pro, so it's unusable at scale — everything
here uses `--format explainer`, which is also the 16:9 horizontal format (`short`
is the vertical one).

Chat is 500/day, which is why each unit spends one chat on a beat sheet before
spending a video. Abundant quota protects scarce quota.

## Quality: three scrutinised rounds, then one video

Chat is 500/day on Pro; video is 20/day. So `src/plan.py` spends three cheap
rounds building a spec that has already been audited, and only then spends the
video:

| Round | Purpose |
| --- | --- |
| 1. STRUCTURE | The course file's own section headings and technical terms, **verbatim**. Also reports whether real detailed notes exist, and which syllabus topics the book does not actually explain. |
| 2. SUBSTANCE | Per section: the book's definition, the mechanism, one concrete specific printed in the book, one step of the running example. |
| 3. SCRUTINY | Audits rounds 1–2 against the syllabus. Merges duplicate sections, deletes motivational and definitional padding, and allocates a per-section second budget weighted by how much later material depends on it. |

Every round speaks a strict line-prefix protocol (`SECTION | 2 | Member access
rules`), so output is parseable and a chatty model cannot leak prose into the
next stage. If round 3 misbehaves, the parser falls back to round 1's verbatim
headings with an even split rather than shipping nothing.

### Terminology lock

Round 1 exists mainly to fix an observed failure: left alone, the model renames
sections to catchier synonyms — "Access Control Rules" for the book's "Member
access rules". Headings are extracted verbatim, then locked in the video prompt
as a hard constraint with a worked negative example. `fire.py` also diffs the
final headings against round 1 and logs `heading_drift` if any were reworded, so
a silent rename is visible instead of shipped.

This matters downstream too: chapter labels **are** the locked headings, so a
student can match each chapter in the app to a section in their own course file
by name.

### Filling in a thin course file

If round 1 reports no real detailed notes, or two or more syllabus topics
unexplained, `fire.py` runs `source add-research --mode deep --import-all
--cited-only` to pull real teaching material for exactly those gaps, then re-asks
round 1 against the enlarged source set. So a course file that is only a table
of contents still produces a complete unit, and the gap is recorded in the
manifest rather than being quietly invented or skipped.

### Density and duration

The video prompt carries an explicit banned-phrase list ("let's understand", "now
that we know", "as we discussed", "you might be wondering", "it is important to
note", "in this video we will", …), a rule that every sentence must introduce,
explain, connect, justify or exemplify, and a ban on re-defining any term.

Because generated length is not controllable, the prompt states per-section
second budgets and an explicit instruction: if running short, add depth to the
highest-budget sections — never padding, never repetition, never a recap.

`pedagogy:` in the syllabus adapts this per subject — `code_language: Java` puts
real code on screen, while omitting it suits a conceptual subject; `math: none`
forbids derivations and numericals, `worked` allows one numeric case per section.

The highest-leverage field you can tune by hand is `example:` — one worked
example per unit, carried end to end. For OOP through Java it's a single
`BankAccount` that grows across all five units: fields and constructors in U1,
subclassed and overridden in U2, a custom exception plus a thread race in U3, a
JFrame form in U4, persisted to file/ArrayList/JDBC in U5.

Preview any prompt without spending quota:

```bash
python src/fire.py --syllabus syllabus/oop-java.yml --unit u1 --minutes 12 --dry-run
```

(Dry-run fakes round 1 from the syllabus line, so headings look long; a real run
uses the book's own short headings.)

## Chapters are measured, not invented

Gemini Notebook returns no timing information, so the only honest chapter time is
when the concept is actually spoken. `postprocess.py` transcribes the finished
MP4 with faster-whisper, then locates each locked heading's anchor word and uses
its first occurrence. Anchors prefer a technical token (`CLASSPATH`, `ArrayList`,
`JDBC`) over the longest word, since those are spoken distinctively — plain
longest-word would anchor "understanding CLASSPATH" on "understanding".
Unmatched chapters fall back to an even split, and timestamps are forced strictly
increasing. The same pass emits WebVTT for the app's `captions` field, and
`ffprobe` supplies the true `durationSec` — no guessed metadata anywhere.

## Watermark swap

The NotebookLM mark is a bottom-right wordmark, replaced **in place** with the
GCTC logo: same box, same frame size, no letterboxing. `delogo` is deliberately
not used — it blurs a smeared patch instead of replacing the mark.

The box in `config/logo.json` was measured from real explainer frames: bbox
`x=1384 y=830 w=156 h=16` at 1600×900, aspect 9.75:1, on a near-white background
(level 254). It is stored **normalised** (fractions of frame size) so it holds if
Google changes the render resolution. The GCTC logo is 5:1 against the mark's
9.75:1, so it is fitted inside the box by height with
`force_original_aspect_ratio=decrease` and right-aligned, never stretched.

To re-measure after any change upstream:

```bash
python src/postprocess.py --syllabus syllabus/oop-java.yml --unit u1 --sample-frames
```

## Setup

```bash
pip install -r requirements.txt

# once per Google account
notebooklm profile create acct1
notebooklm -p acct1 login --master-token --account you@example.com
cat ~/.notebooklm/profiles/acct1/master_token.json   # -> repo secret
```

Repo secrets:

| Secret | Purpose |
| --- | --- |
| `NOTEBOOKLM_MASTER_TOKEN` | Durable, self-healing CI auth (preferred) |
| `NOTEBOOKLM_AUTH_JSON` | Fallback cookie snapshot; expires |

## Hosting

MP4s are **not** committed — ~350 lectures would blow past GitHub's repo limits.
They leave as build artifacts and are uploaded by the publish stage.

Pilot target is Drive, because the app already has the code path: with a Drive
API key, `VideoUrl.driveStreamUrl` produces a `googleapis.com/drive/v3` URL that
honors Range requests, so seeking inside a 10-minute lecture works. Without the
key it falls back to `uc?export=download`, which throttles.

Later migration to Archive.org or R2 is a URL swap in the catalog JSON — the app
resolves both as `DIRECT`, no code change.

## Catalog JSON

`catalog-entry.json` matches `VideoCatalogRepository.parseVideos` exactly, and
always sets an explicit `id`. The app's fallback id is
`slug(title) + hash(url)`, which collides whenever titles and URLs repeat — in
the current placeholder catalog, 353 video entries collapse into 20 distinct ids,
which would make watch progress and offline downloads shared across unrelated
subjects.

## Layout

```
config/logo.json      watermark bounding box (measure once)
syllabus/*.yml        per-subject source of truth: units, topics, example, anchors
src/common.py         config, manifest, notebooklm CLI wrapper + rate-limit retry
src/plan.py           the two prompt templates — where quality is decided
src/fire.py           notebook + source + beat sheet + start generation
src/collect.py        poll and download finished artifacts
src/postprocess.py    logo swap, captions, chapters, thumbnail, catalog entry
state/manifest.json   durable pipeline state
```
