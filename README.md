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

## Quality

The whole reason output is specific to *this* course rather than generic Java is
`src/plan.py`. Two prompts per unit:

1. **Outline prompt** → `notebooklm ask`. Pulls the unit's real substance out of
   the course file's *Detailed notes*, as a 6–8 beat sheet with the book's own
   definitions and one concrete detail per beat. Explicitly excludes the
   department vision, PO/PEO tables, timetables and question banks that make up
   half a course file.
2. **Video prompt** → `generate video`. Wraps that beat sheet in hard structural
   constraints and an explicit banned-phrases list ("in this video we will", "as
   we saw earlier", agenda previews, recaps, encouragement).

The highest-leverage field you can tune by hand is `example:` in the syllabus —
one worked example per unit, carried end to end. For OOP through Java it's a
single `BankAccount` that grows across all five units. A named, concrete example
is what separates a lecture from a definition list.

Preview any prompt without spending quota:

```bash
python src/fire.py --syllabus syllabus/oop-java.yml --unit u1 --dry-run
```

## Chapters are measured, not invented

Gemini Notebook returns no timing information, so the only honest chapter time is
when the concept is actually spoken. `postprocess.py` transcribes the finished
MP4 with faster-whisper, then locates each beat's `anchors:` phrase and uses its
first occurrence. Unmatched chapters fall back to an even split, and timestamps
are forced strictly increasing. The same transcription pass emits WebVTT for the
app's `captions` field, and `ffprobe` supplies the true `durationSec` — no
guessed metadata anywhere.

## Watermark swap

The NotebookLM mark is replaced **in place** with the GCTC logo: same bounding
box, same frame size, no letterboxing. `delogo` is deliberately not used — it
blurs a smeared patch instead of replacing the mark.

The box has to be measured once from a real frame:

```bash
python src/postprocess.py --syllabus syllabus/oop-java.yml --unit u1 --sample-frames
# read w/h/x/y off build/<unit>/frames/*.png, then fill config/logo.json
```

Until `config/logo.json` has a non-zero `w`, postprocess copies the video through
unchanged rather than guessing at coordinates.

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
