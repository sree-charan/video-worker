"""Shared plumbing: config loading, the manifest, and the notebooklm CLI wrapper.

The manifest is the only durable state in this pipeline. Every stage reads it,
mutates one unit's record, and writes it back. That is what makes a run
resumable: the CLI drives a consumer product through a session that can expire
or rate-limit mid-batch, so any stage must be safe to re-run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "state" / "manifest.json"
BUILD = ROOT / "build"


# ---------------------------------------------------------------- syllabus

def load_syllabus(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "subject" not in data or "units" not in data:
        raise SystemExit(f"{path}: expected top-level 'subject' and 'units'")
    data["units"].sort(key=lambda u: u["n"])
    return data


def unit_by_id(syl: dict, unit_id: str) -> dict:
    for u in syl["units"]:
        if u["id"] == unit_id or str(u["n"]) == str(unit_id):
            return u
    raise SystemExit(f"unknown unit {unit_id!r}; have {[u['id'] for u in syl['units']]}")


# ---------------------------------------------------------------- manifest

def read_manifest() -> dict[str, Any]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"version": 1, "units": {}}


def write_manifest(m: dict[str, Any]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record(subject_id: str, unit_id: str, **fields: Any) -> dict:
    """Merge fields into one unit's manifest record and persist immediately."""
    m = read_manifest()
    key = f"{subject_id}/{unit_id}"
    rec = m["units"].setdefault(key, {"subject": subject_id, "unit": unit_id, "state": "new"})
    rec.update({k: v for k, v in fields.items() if v is not None})
    rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_manifest(m)
    return rec


def get_record(subject_id: str, unit_id: str) -> dict:
    return read_manifest()["units"].get(f"{subject_id}/{unit_id}", {})


# ---------------------------------------------------------------- CLI wrapper

class CliError(RuntimeError):
    def __init__(self, argv: list[str], code: int, out: str, err: str):
        self.argv, self.code, self.out, self.err = argv, code, out, err
        super().__init__(f"notebooklm {' '.join(argv)} -> exit {code}\n{err.strip()[:800]}")

    @property
    def rate_limited(self) -> bool:
        blob = (self.err + self.out).lower()
        return any(s in blob for s in ("rate limit", "quota", "too many requests", "resource_exhausted"))


def nlm(*argv: str, profile: str | None = None, timeout: int = 900,
        parse_json: bool = True) -> Any:
    """Run the notebooklm CLI. Returns parsed JSON when --json was requested.

    `profile` selects which Google account to spend quota from; three Pro
    accounts are three profiles, so the caller decides the account per unit.
    """
    exe = os.environ.get("NOTEBOOKLM_BIN", "notebooklm")
    cmd = [exe]
    if profile:
        cmd += ["-p", profile]
    cmd += [str(a) for a in argv]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise CliError(cmd[1:], proc.returncode, proc.stdout, proc.stderr)
    if not parse_json:
        return proc.stdout
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        # Some commands print a human line before/after the JSON payload.
        m = re.search(r"[\[{].*[\]}]", proc.stdout, re.S)
        if m:
            return json.loads(m.group(0))
        raise CliError(cmd[1:], 0, proc.stdout, "expected JSON on stdout")


def nlm_retry(*argv: str, profile: str | None = None, attempts: int = 4,
              base_delay: float = 20.0, **kw: Any) -> Any:
    """Retry only on rate limiting; fail fast on everything else."""
    for i in range(1, attempts + 1):
        try:
            return nlm(*argv, profile=profile, **kw)
        except CliError as e:
            if not e.rate_limited or i == attempts:
                raise
            delay = base_delay * (2 ** (i - 1))
            log(f"rate limited, retrying in {delay:.0f}s ({i}/{attempts - 1})")
            time.sleep(delay)


# ---------------------------------------------------------------- misc

def slug(text: str, limit: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit]


def log(msg: str) -> None:
    print(f"[video-worker] {msg}", file=sys.stderr, flush=True)


def unit_key(syl: dict, unit: dict) -> str:
    return f"{syl['subject']['id']}-{unit['id']}"
