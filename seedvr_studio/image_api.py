"""Loopback-only image workflow routes; files are addressed by registered IDs."""

from __future__ import annotations

import ipaddress
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .backend import MODEL_FILES, TRT_ENCODER_5F, TRT_ENGINE_5F
from .image_batches import ImageBatches
from .paths import MODELS, OUTPUTS

BATCHES = ImageBatches(OUTPUTS / "image-batch-history", OUTPUTS / "images")
_PICKER_LOCK = Lock()


def _local_request(request: Request):
    host = request.url.hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise HTTPException(403, "Local image access requires the Studio localhost address")
    try:
        local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        local = False
    if not local:
        raise HTTPException(403, "Local image access is available only on the Studio computer")
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
        except ValueError as exc:
            raise HTTPException(403, "Invalid request origin") from exc
        if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
            raise HTTPException(403, "Cross-origin local file access is not allowed")
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(403, "Cross-site local file access is not allowed")


router = APIRouter(prefix="/api/images", dependencies=[Depends(_local_request)])


class ReviewRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, max_length=20000)
    folders: list[str] = Field(default_factory=list, max_length=200)
    output_root: str = ""
    settings: dict = Field(default_factory=dict)
    create_output_root: bool = False
    previous_review: str = ""


class StartRequest(BaseModel):
    review_id: str
    preview_item: str = ""


class PickRequest(BaseModel):
    kind: Literal["files", "source_folder", "output_folder"]
    initial_dir: str = ""


def _perform(action, *args, **kwargs):
    try:
        return action(*args, **kwargs)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc).strip("'")) from exc
    except (ValueError, TypeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _urls(record, *, review=False):
    prefix = "reviews" if review else "batches"
    for item in record["items"]:
        base = f"/api/images/{prefix}/{record['id']}/items/{item['id']}"
        item["source_url"] = base + "/source"
        item["output_url"] = base + "/output" if item["status"] == "complete" else None
    return record


@router.get("/config")
def image_config():
    return {"default_output_root": str(BATCHES.default_output_root),
            "models": [name for name, file in MODEL_FILES.items() if (MODELS / file).is_file()] or list(MODEL_FILES),
            "tensorrt_artifacts": TRT_ENGINE_5F.is_file() and TRT_ENCODER_5F.is_file(),
            "native_picker": sys.platform == "win32"}


@router.post("/pick")
def pick_images(values: PickRequest):
    if sys.platform != "win32":
        raise HTTPException(400, "Native selection requires Windows; use an absolute folder path instead")
    if not _PICKER_LOCK.acquire(blocking=False):
        raise HTTPException(409, "A file/folder picker is already open")
    # A separate GUI process keeps Tk on its main thread and avoids tying the
    # browser server's lifetime to a native dialog. User input is never code.
    script = """
import json, pathlib, sys, tkinter as tk
from tkinter import filedialog
kind, initial = sys.argv[1:]
directory = pathlib.Path(initial).expanduser() if initial else pathlib.Path.home() / 'Desktop'
if not directory.is_dir(): directory = pathlib.Path.home()
root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
try:
    if kind == 'files':
        result = list(filedialog.askopenfilenames(parent=root, title='Add images to SeedVR Studio', initialdir=str(directory), filetypes=[('JPEG and PNG images', '*.jpg *.jpeg *.png'), ('All files', '*.*')]))
    else:
        value = filedialog.askdirectory(parent=root, title='Choose output folder' if kind == 'output_folder' else 'Add image folder (no subfolders)', initialdir=str(directory), mustexist=kind != 'output_folder')
        result = [value] if value else []
    print(json.dumps(result))
finally: root.destroy()
"""
    try:
        result = subprocess.run([sys.executable, "-c", script, values.kind, values.initial_dir],
                                capture_output=True, text=True, encoding="utf-8", timeout=300,
                                env={**os.environ, "PYTHONUTF8": "1"},
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise HTTPException(400, "Windows picker could not open. Use the folder path field instead.")
        return {"paths": json.loads(result.stdout)}
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(408, "Folder picker timed out; try again or enter a path") from exc
    finally:
        _PICKER_LOCK.release()


@router.post("/review")
def review_images(values: ReviewRequest):
    return _urls(_perform(BATCHES.review, **values.model_dump()), review=True)


@router.post("/batches")
def start_images(values: StartRequest):
    return _urls(_perform(BATCHES.start, values.review_id, preview_item=values.preview_item))


@router.get("/batches")
def image_history():
    return [{key: value for key, value in record.items() if key not in {"items", "skipped", "settings"}} |
            {"total": len(record["items"]), "completed": sum(item["status"] == "complete" for item in record["items"])}
            for record in BATCHES.history()]


@router.get("/batches/{batch_id}")
def image_status(batch_id: str):
    return _urls(_perform(BATCHES.get, batch_id))


@router.post("/batches/{batch_id}/cancel")
def cancel_images(batch_id: str):
    return _urls(_perform(BATCHES.cancel, batch_id))


@router.post("/batches/{batch_id}/retry")
def retry_images(batch_id: str):
    return _urls(_perform(BATCHES.retry, batch_id))


@router.get("/reviews/{batch_id}/items/{item_id}/source")
def reviewed_source(batch_id: str, item_id: str):
    return Response(_perform(BATCHES.source_preview, batch_id, item_id, review=True), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@router.get("/batches/{batch_id}/items/{item_id}/source")
def batch_source(batch_id: str, item_id: str):
    return Response(_perform(BATCHES.source_preview, batch_id, item_id), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@router.get("/batches/{batch_id}/items/{item_id}/output")
def image_output(batch_id: str, item_id: str):
    path = _perform(BATCHES.file, batch_id, item_id)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/batches/{batch_id}/log")
def image_log(batch_id: str):
    _perform(BATCHES.get, batch_id)
    path = BATCHES.state_root / f"{batch_id}.log"
    if not path.is_file():
        raise HTTPException(404, "No inference log has been saved yet")
    return FileResponse(path, media_type="text/plain", filename=f"image-batch-{batch_id}.log")


@router.post("/batches/{batch_id}/open-folder")
def open_image_folder(batch_id: str):
    record = _perform(BATCHES.get, batch_id)
    path = Path(record["output_dir"])
    if not path.is_dir() or path.resolve() != path:
        raise HTTPException(404, "Output folder is unavailable")
    try:
        subprocess.Popen(["explorer.exe", str(path)], close_fds=True)
    except OSError as exc:
        raise HTTPException(400, "Windows Explorer could not open the output folder") from exc
    return {"ok": True}
