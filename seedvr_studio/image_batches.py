"""Image selection, immutable review, safe publication and durable result history.

The browser deals only in registered batch/item identities. All filesystem
policy, per-image failures, retries and output-root handling live here.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from threading import Event, RLock, Thread
from uuid import uuid4

from PIL import Image

from .gpu_jobs import reserve_gpu
from .image_inference import ImageRenderer, ImageSettings, decode_image, fatal_gpu_error
from .media import RenderCancelledError

EXTENSIONS = {".jpg", ".jpeg", ".png"}
ACTIVE = {"queued", "running", "cancelling"}


def _natural(value: str):
    return tuple((0, int(part)) if part.isdigit() else (1, part.casefold()) for part in re.split(r"(\d+)", value))


def _read_source(path: Path) -> bytes:
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError("Compressed image exceeds the 512 MiB input limit")
    return path.read_bytes()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _output_name(source: str, item_id: str, directory: Path, used: set[str]) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source).strip(" .") or "image"
    stem = name.split(".")[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        name = "_" + name
    # Leave room for Windows' traditional path limit as not every library opts
    # into long paths. Counting UTF-16 units also handles non-BMP filenames.
    budget = min(240, 245 - len(str(directory).encode("utf-16-le")) // 2)
    if budget < 30:
        raise ValueError("Output location is too long; choose a shorter parent folder")
    suffix = "_upscaled.png"
    candidate = name + suffix
    if len(candidate.encode("utf-16-le")) // 2 > budget or candidate.casefold() in used:
        suffix = f"_upscaled-{item_id}.png"
        while len((name + suffix).encode("utf-16-le")) // 2 > budget:
            name = name[:-1]
        candidate = name + suffix
    used.add(candidate.casefold())
    return candidate


class ImageBatches:
    def __init__(self, state_root: Path, default_output_root: Path, *, renderer_factory=ImageRenderer):
        self.state_root = state_root
        self.default_output_root = default_output_root
        self.renderer_factory = renderer_factory
        self.lock = RLock()
        self.reviews: dict[str, dict] = {}
        self.batches: dict[str, dict] = {}
        self.cancellations: dict[str, Event] = {}
        self.threads: dict[str, Thread] = {}
        self._load()

    def _load(self):
        for path in self.state_root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(record, dict) or record.get("version") != 1 or record.get("id") != path.stem:
                    continue
                if (not isinstance(record.get("items"), list)
                        or not all(isinstance(item, dict) and {"id", "source", "name", "status", "output_name"} <= item.keys()
                                   for item in record["items"])
                        or not isinstance(record.get("created_at"), (int, float))
                        or not isinstance(record.get("output_dir"), str)
                        or not isinstance(record.get("settings"), dict)):
                    continue
                if record["status"] in ACTIVE:
                    record.update(status="interrupted", message="Studio stopped before this image batch finished; explicit retry is available")
                    for item in record["items"]:
                        if item["status"] == "running":
                            item["status"] = "pending"
                self.batches[record["id"]] = record
            except (OSError, ValueError, KeyError, TypeError):
                continue  # One damaged history record cannot hide other results.

    def _save(self, batch: dict):
        self.state_root.mkdir(parents=True, exist_ok=True)
        destination = self.state_root / f"{batch['id']}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(batch, ensure_ascii=True, indent=2), encoding="utf-8")
        temporary.replace(destination)

    @staticmethod
    def _public(record: dict) -> dict:
        result = copy.deepcopy(record)
        for item in result["items"]:
            item.pop("fingerprint", None)
            item.pop("output_fingerprint", None)
        return result

    def _output_root(self, value: str, create: bool) -> Path:
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        root = Path(value).expanduser() if value else self.default_output_root
        if not root.is_absolute():
            raise ValueError("Enter an absolute output-folder path")
        root = root.resolve()
        if root.exists() and not root.is_dir():
            raise ValueError("Output root must be a directory")
        if not root.exists() and root != self.default_output_root.resolve() and not create:
            raise ValueError("Output folder does not exist. Create it in Browse, or enable Create output folder.")
        return root

    def review(self, *, paths: list[str], folders: list[str], output_root: str = "", settings: dict | None = None,
               create_output_root: bool = False, previous_review: str = "") -> dict:
        configuration = ImageSettings(**(settings or {}))
        root = self._output_root(output_root, create_output_root)
        batch_id = uuid4().hex[:16]
        folder = root / f"image-batch-{time.strftime('%Y%m%d-%H%M%S')}-{batch_id[:8]}"
        with self.lock:
            previous = self.reviews.get(previous_review, {})
            skipped = copy.deepcopy(previous.get("skipped", []))
        candidates, seen, duplicates = [], set(), 0
        for raw_folder in folders:
            directory = Path(raw_folder).expanduser()
            if not directory.is_absolute() or not directory.is_dir():
                raise ValueError(f"Source folder was not found: {raw_folder}")
            for path in directory.iterdir():
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    candidates.append(str(path))
                else:
                    skipped.append({"path": str(path), "reason": "subfolder" if path.is_dir() else "unsupported file"})
        candidates += paths
        items, used_names = [], set()
        for raw in sorted(candidates, key=lambda value: (_natural(Path(value).name), str(value).casefold())):
            source = Path(raw).expanduser()
            if not source.is_absolute():
                raise ValueError("Source image paths must be absolute")
            source = source.resolve()
            key = os.path.normcase(str(source))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            if source.suffix.lower() not in EXTENSIONS:
                skipped.append({"path": str(source), "reason": "unsupported file"})
                continue
            item_id = uuid4().hex[:12]
            item = {"id": item_id, "source": str(source), "name": source.name, "status": "pending", "error": None}
            try:
                data = _read_source(source)
                image = decode_image(data)
                target = configuration.target_size(*image.size)
                item.update(fingerprint=_digest(data), width=image.width, height=image.height,
                            target_width=target[0], target_height=target[1], alpha=image.mode == "RGBA")
                item["output_name"] = _output_name(source.name, item_id, folder, used_names)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                item.update(status="error", error=str(exc), output_name=None)
            items.append(item)
        if not items:
            raise ValueError("Choose at least one JPEG or PNG image")
        skipped = list({(entry["path"], entry["reason"]): entry for entry in skipped}.values())
        record = {"version": 1, "id": batch_id, "kind": "batch", "status": "review", "created_at": time.time(),
                  "settings": configuration.__dict__, "output_root": str(root), "output_dir": str(folder),
                  "create_output_root": create_output_root or root == self.default_output_root.resolve(),
                  "items": items, "skipped": skipped, "duplicates": duplicates, "progress": 0.0,
                  "message": "Review the selection and destination, then Start"}
        with self.lock:
            self.reviews[batch_id] = record
            while len(self.reviews) > 32:
                self.reviews.pop(next(iter(self.reviews)))
        return self._public(record)

    def get(self, batch_id: str, *, review: bool = False) -> dict:
        with self.lock:
            record = (self.reviews if review else self.batches).get(batch_id)
            if record is None:
                raise KeyError("Image batch was not found")
            return self._public(record)

    def history(self) -> list[dict]:
        with self.lock:
            return [self._public(record) for record in sorted(self.batches.values(), key=lambda item: item["created_at"], reverse=True)]

    def busy(self) -> bool:
        with self.lock:
            return any(record["status"] in ACTIVE for record in self.batches.values())

    def start(self, review_id: str, *, preview_item: str = "") -> dict:
        with self.lock:
            review = self.reviews.get(review_id)
            if review is None:
                raise KeyError("Review expired; review your selection again")
            if not preview_item and review_id in self.batches:
                return self._public(self.batches[review_id])
            batch = copy.deepcopy(review)
            if preview_item:
                batch["items"] = [item for item in batch["items"] if item["id"] == preview_item]
                if not batch["items"]:
                    raise KeyError("Preview image was not found")
                batch["id"], batch["kind"] = uuid4().hex[:16], "preview"
                batch["output_root"] = str(self.default_output_root)
                batch["output_dir"] = str(self.default_output_root / f"image-preview-{batch['id']}")
                batch["create_output_root"] = True
            if not any(item["status"] == "pending" for item in batch["items"]):
                raise ValueError("There are no readable images to process")
            batch["reviewed_at"], batch["created_at"] = batch["created_at"], time.time()
            root = Path(batch["output_root"])
            if batch["create_output_root"]:
                root.mkdir(parents=True, exist_ok=True)
            if root.resolve() != root or not root.is_dir():
                raise ValueError("Output root changed; review the destination again")
            output = Path(batch["output_dir"])
            output.mkdir()  # Intentionally fails if another path already exists.
            # Exercise actual write permission before taking GPU ownership.
            with tempfile.TemporaryFile(dir=output):
                pass
            batch.update(status="queued", message="Waiting for the GPU", started_at=None, elapsed_seconds=0,
                         attempt_started=time.time())
            self.batches[batch["id"]] = batch
            self.cancellations[batch["id"]] = Event()
            self._save(batch)
            self._launch(batch["id"])
            return self._public(batch)

    def _launch(self, batch_id: str):
        thread = Thread(target=self._run, args=(batch_id,), daemon=True)
        self.threads[batch_id] = thread
        thread.start()

    def _run(self, batch_id: str):
        batch, cancelled = self.batches[batch_id], self.cancellations[batch_id]
        renderer = None
        try:
            with reserve_gpu(f"images:{batch_id}", cancelled), ExitStack() as cleanup:
                from .inference_worker import WARM_INFERENCE
                WARM_INFERENCE.close()
                with self.lock:
                    batch.update(status="running", started_at=time.time(), message="Starting image batch")
                    self._save(batch)
                renderer = self.renderer_factory()
                cleanup.callback(renderer.close)
                settings = ImageSettings(**batch["settings"])
                output_dir = Path(batch["output_dir"])
                count = len(batch["items"])
                for index, item in enumerate(batch["items"]):
                    if cancelled.is_set():
                        raise RenderCancelledError("Image batch cancelled; completed outputs were kept")
                    if item["status"] != "pending":
                        continue
                    with self.lock:
                        item.update(status="running", error=None)
                        batch.update(current_item=item["id"], message=f"Image {index + 1}/{count}: {item['name']}")
                        self._save(batch)
                    try:
                        data = _read_source(Path(item["source"]))
                        if _digest(data) != item["fingerprint"]:
                            raise ValueError("Source image changed since review; create a new review to process the changed file")
                        if output_dir.resolve() != output_dir:
                            raise ValueError("Output folder changed or became a filesystem link")
                        with tempfile.TemporaryDirectory(prefix=".image-work-", dir=output_dir) as temporary:
                            scratch = Path(temporary)
                            source = scratch / ("original" + Path(item["source"]).suffix)
                            source.write_bytes(data)
                            pending = scratch / "finished.png"

                            def report(progress, message):
                                with self.lock:
                                    if progress is not None:
                                        batch["progress"] = (index + min(0.99, max(0.0, progress))) / count
                                    batch["message"] = f"Image {index + 1}/{count}: {message[-800:]}"
                                try:
                                    with (self.state_root / f"{batch_id}.log").open("a", encoding="utf-8") as log:
                                        log.write(f"{time.strftime('%H:%M:%S')} {item['name']}: {message}\n")
                                except OSError:
                                    pass

                            metadata = renderer.render(source, pending, settings, cancelled, report)
                            if cancelled.is_set():
                                raise RenderCancelledError("Image batch cancelled")
                            with Image.open(pending) as result:
                                if result.format != "PNG" or result.size != (item["target_width"], item["target_height"]):
                                    raise ValueError("Inference did not produce the expected PNG dimensions")
                                result.verify()
                            content = pending.read_bytes()
                            destination = output_dir / item["output_name"]
                            # Hard-link publication is atomic and refuses existing
                            # targets. Some filesystems do not support hard links;
                            # exclusive-create fallback still never overwrites.
                            try:
                                os.link(pending, destination)
                            except FileExistsError:
                                raise ValueError("Output already exists; it was not overwritten") from None
                            except OSError:
                                created = False
                                try:
                                    with destination.open("xb") as handle:
                                        created = True
                                        handle.write(content)
                                        handle.flush()
                                        os.fsync(handle.fileno())
                                except Exception:
                                    if created:
                                        destination.unlink(missing_ok=True)
                                    raise
                            with self.lock:
                                item.update(status="complete", error=None, result=metadata,
                                            output_fingerprint=_digest(content), completed_at=time.time())
                    except RenderCancelledError:
                        with self.lock:
                            item.update(status="cancelled", error="Cancelled; no completed output was published")
                        raise
                    except Exception as exc:
                        with self.lock:
                            item.update(status="error", error=str(exc))
                        renderer.close()  # Drop possibly poisoned GPU/model state.
                        if fatal_gpu_error(exc):
                            raise RuntimeError(f"Unsafe GPU/driver error stopped this batch; completed files were kept: {exc}") from exc
                    with self.lock:
                        batch["progress"] = (index + 1) / count
                        self._save(batch)
                with self.lock:
                    succeeded = sum(item["status"] == "complete" for item in batch["items"])
                    failed = count - succeeded
                    batch.update(status="complete" if not failed else "partial" if succeeded else "error", progress=1.0,
                                 message=f"Finished: {succeeded} completed, {failed} failed")
        except RenderCancelledError as exc:
            with self.lock:
                batch.update(status="cancelled", message=str(exc))
        except Exception as exc:
            with self.lock:
                batch.update(status="error", message=str(exc))
        finally:
            with self.lock:
                batch["elapsed_seconds"] = time.time() - batch["attempt_started"]
                batch.pop("current_item", None)
                try:
                    self._save(batch)
                except OSError as exc:
                    batch["message"] += f"; history could not be saved: {exc}"

    def cancel(self, batch_id: str) -> dict:
        with self.lock:
            batch = self.batches[batch_id]
            if batch["status"] in ACTIVE:
                self.cancellations[batch_id].set()
                batch.update(status="cancelling", message="Stopping this image batch; keeping completed outputs")
            return self._public(batch)

    def cancel_all(self):
        with self.lock:
            for batch_id in self.batches:
                if self.batches[batch_id]["status"] in ACTIVE:
                    self.cancel(batch_id)

    def shutdown(self):
        self.cancel_all()
        deadline = time.monotonic() + 8
        for thread in list(self.threads.values()):
            thread.join(timeout=max(0, deadline - time.monotonic()))

    def retry(self, batch_id: str) -> dict:
        with self.lock:
            original = self.batches[batch_id]
            batch = copy.deepcopy(original)
            if batch["status"] in ACTIVE:
                raise ValueError("Wait for this image batch to stop before retrying")
            root = Path(batch["output_dir"])
            if not root.is_dir() or root.resolve() != root:
                raise ValueError("Output folder is unavailable or changed; restore the original location before retrying")
            pending = 0
            for item in batch["items"]:
                if not item.get("output_name") or not item.get("fingerprint"):
                    continue
                path = Path(batch["output_dir"]) / item["output_name"]
                if item["status"] == "complete" and path.is_file():
                    if _digest(path.read_bytes()) != item.get("output_fingerprint"):
                        item.update(status="error", error="Completed output changed; move it aside or start a new batch. It will not be overwritten.")
                    continue
                if path.exists():
                    item.update(status="error", error="Output target exists; move it aside before retrying. It will not be overwritten.")
                    continue
                item.update(status="pending", error=None)
                pending += 1
            if not pending:
                details = next((item["error"] for item in batch["items"] if item.get("error")), "All completed outputs are present and unchanged")
                raise ValueError(f"Nothing can be retried safely: {details}")
            batch.update(status="queued", message="Retrying failed/missing images; keeping completed outputs", attempt_started=time.time(), progress=0.0)
            self.cancellations[batch_id] = Event()
            self.batches[batch_id] = batch
            self._save(batch)
            self._launch(batch_id)
            return self._public(batch)

    def file(self, batch_id: str, item_id: str, *, source: bool = False, review: bool = False) -> Path:
        with self.lock:
            record = (self.reviews if review else self.batches)[batch_id]
            item = next((item for item in record["items"] if item["id"] == item_id), None)
            if item is None:
                raise KeyError("Image was not found")
            if source:
                path = Path(item["source"])
            else:
                if item["status"] != "complete":
                    raise KeyError("Image output is not complete")
                root = Path(record["output_dir"])
                path = root / item["output_name"]
                if path.resolve().parent != root or root.resolve() != root:
                    raise ValueError("Image output path changed")
            if not path.is_file():
                raise FileNotFoundError("Image location is unavailable")
            return path

    def source_preview(self, batch_id: str, item_id: str, *, review: bool = False) -> bytes:
        path = self.file(batch_id, item_id, source=True, review=review)
        data = _read_source(path)
        with self.lock:
            record = (self.reviews if review else self.batches)[batch_id]
            item = next(item for item in record["items"] if item["id"] == item_id)
            if _digest(data) != item.get("fingerprint"):
                raise ValueError("Source image changed since review; review the changed source again")
        image = decode_image(data)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
