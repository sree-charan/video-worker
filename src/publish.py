"""Stage: publish.

Uploads a finished lecture to a Shared Drive, makes it readable by link, and
writes the resulting file ids into the unit's catalog entry.

Why a Shared Drive and a service account: a service account has no storage quota
of its own, so uploading into someone's My Drive fails with
storageQuotaExceeded. In a Shared Drive the storage belongs to the drive, so it
draws on the Workspace pool. Every call therefore passes supportsAllDrives, which
Shared Drives require and which is the usual thing to forget.

The app never sees these ids raw. VideoUrl.driveStreamUrl turns a share URL into
a googleapis.com/drive/v3 streaming URL when a Drive API key is configured, which
is what makes seeking inside a twelve minute lecture work.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BUILD, get_record, load_syllabus, log, record, unit_by_id  # noqa: E402

SCOPES = ["https://www.googleapis.com/auth/drive"]

# What gets uploaded, and the catalog field each one fills.
ARTEFACTS = (
    ("final.mp4", "url"),
    ("captions.vtt", "vtt"),
    ("thumb.jpg", "thumb"),
)


def drive_service():
    """Authenticate as the service account from GDRIVE_SA_KEY."""
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    raw = os.environ.get("GDRIVE_SA_KEY", "").strip()
    if not raw:
        raise SystemExit("GDRIVE_SA_KEY is not set")
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"GDRIVE_SA_KEY is not valid JSON: {exc}") from exc

    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    log(f"authenticated as {info.get('client_email')}")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_or_create_folder(svc, name: str, parent: str) -> str:
    """A folder id for `name` under `parent`, reusing one if it already exists.

    Reuse matters because publishing is re-runnable: a second publish of the same
    unit must not leave a trail of duplicate folders.
    """
    safe = name.replace("'", "\\'")
    q = (f"name='{safe}' and '{parent}' in parents "
         "and mimeType='application/vnd.google-apps.folder' and trashed=false")
    found = svc.files().list(q=q, fields="files(id)", pageSize=1,
                            supportsAllDrives=True,
                            includeItemsFromAllDrives=True).execute()
    if found.get("files"):
        return found["files"][0]["id"]
    created = svc.files().create(
        body={"name": name, "parents": [parent],
              "mimeType": "application/vnd.google-apps.folder"},
        fields="id", supportsAllDrives=True).execute()
    return created["id"]


def upload(svc, path: Path, parent: str) -> str:
    """Upload one file, replacing any previous copy of the same name."""
    from googleapiclient.http import MediaFileUpload

    safe = path.name.replace("'", "\\'")
    existing = svc.files().list(
        q=f"name='{safe}' and '{parent}' in parents and trashed=false",
        fields="files(id)", pageSize=1, supportsAllDrives=True,
        includeItemsFromAllDrives=True).execute().get("files")

    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(path), mimetype=mime, resumable=True,
                            chunksize=8 * 1024 * 1024)

    if existing:
        fid = existing[0]["id"]
        req = svc.files().update(fileId=fid, media_body=media, fields="id",
                                 supportsAllDrives=True)
    else:
        req = svc.files().create(body={"name": path.name, "parents": [parent]},
                                 media_body=media, fields="id",
                                 supportsAllDrives=True)

    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            log(f"    {path.name}: {int(status.progress() * 100)}%")
    return response["id"]


def share_publicly(svc, file_id: str) -> None:
    """Grant anyone-with-the-link read access.

    Without this the app cannot fetch the file at all. If the Shared Drive or the
    organisation forbids link sharing this raises, and that is worth failing on
    loudly rather than publishing a catalog of unreachable URLs.
    """
    from googleapiclient.errors import HttpError

    try:
        svc.permissions().create(
            fileId=file_id, body={"role": "reader", "type": "anyone"},
            supportsAllDrives=True).execute()
    except HttpError as exc:
        if "already" in str(exc).lower() or exc.resp.status == 409:
            return
        raise SystemExit(
            f"could not make {file_id} link-readable: {exc}\n"
            "Check that the Shared Drive and the organisation permit sharing "
            "outside the domain."
        ) from exc


def publish_unit(svc, syl: dict, unit: dict, root: str) -> dict:
    sid = syl["subject"]["id"]
    rec = get_record(sid, unit["id"])
    outdir = BUILD / f"{sid}-{unit['id']}"

    final = Path(rec.get("final_mp4") or (outdir / "final.mp4"))
    if not final.exists():
        raise SystemExit(f"no finished video for {unit['id']}; postprocess it first")

    subject_folder = find_or_create_folder(svc, sid, root)
    unit_folder = find_or_create_folder(svc, unit["id"], subject_folder)
    log(f"  folder {sid}/{unit['id']} -> {unit_folder}")

    ids: dict[str, str] = {}
    for filename, field in ARTEFACTS:
        p = outdir / filename
        if not p.exists():
            log(f"    {filename} missing, skipping")
            continue
        fid = upload(svc, p, unit_folder)
        share_publicly(svc, fid)
        ids[field] = fid

    entry_path = outdir / "catalog-entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8")) if entry_path.exists() else {}
    if "url" in ids:
        entry["url"] = f"https://drive.google.com/file/d/{ids['url']}/view?usp=sharing"
    for field in ("vtt", "thumb"):
        if field in ids:
            entry[field] = ids[field]
    entry_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")

    record(sid, unit["id"], state="published", drive_ids=ids,
           drive_folder=unit_folder, catalog_entry=entry)
    log(f"  published {unit['id']}: " + ", ".join(f"{k}={v[:10]}…" for k, v in ids.items()))
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload finished lectures to Drive.")
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--unit", action="append", default=[])
    ap.add_argument("--folder", default=os.environ.get("GDRIVE_FOLDER_ID", ""))
    a = ap.parse_args()

    if not a.folder:
        raise SystemExit("no destination: pass --folder or set GDRIVE_FOLDER_ID")

    syl = load_syllabus(a.syllabus)
    units = [unit_by_id(syl, u) for u in a.unit] if a.unit else syl["units"]
    svc = drive_service()

    published = 0
    for u in units:
        rec = get_record(syl["subject"]["id"], u["id"])
        if rec.get("state") not in ("postprocessed", "published"):
            log(f"unit {u['id']}: state={rec.get('state')}, not ready; skipping")
            continue
        log(f"=== publishing unit {u['n']} ({u['id']})")
        publish_unit(svc, syl, u, a.folder)
        published += 1

    log(f"{published} unit(s) published")


if __name__ == "__main__":
    main()
