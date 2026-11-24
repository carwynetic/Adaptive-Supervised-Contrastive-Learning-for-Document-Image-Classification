from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from urllib3.util.retry import Retry


def _build_ocr_callable(ocr_engine: str) -> Callable[[Image.Image], Tuple[List[str], List[List[int]]]]:
    key = ocr_engine.strip().lower()
    if key == "tesseract":
        from model.modules.ocr_pytesseract import TesseractOCR

        engine = TesseractOCR()
        return engine.extract_words_and_boxes
    if key == "pp_ocrv6":
        from model.modules.ocr_ppdoclayout_ppocrv6small import DocumentOCR

        engine = DocumentOCR()
        return engine.extract_words_and_boxes
    if key == "easyocr":
        import easyocr
        import numpy as np

        reader = easyocr.Reader(["en"], gpu=False)

        def _run_easyocr(img: Image.Image) -> Tuple[List[str], List[List[int]]]:
            rgb = img.convert("RGB")
            w, h = rgb.size
            res = reader.readtext(np.array(rgb))
            words: List[str] = []
            boxes: List[List[int]] = []
            for bbox_pts, text, _ in res:
                xs = [p[0] for p in bbox_pts]
                ys = [p[1] for p in bbox_pts]
                norm_box = [
                    max(0, min(1000, int(1000 * min(xs) / max(1, w)))),
                    max(0, min(1000, int(1000 * min(ys) / max(1, h)))),
                    max(0, min(1000, int(1000 * max(xs) / max(1, w)))),
                    max(0, min(1000, int(1000 * max(ys) / max(1, h)))),
                ]
                for token in str(text).strip().split():
                    if token:
                        words.append(token)
                        boxes.append(norm_box)
            if not words:
                return ["empty"], [[0, 0, 1000, 1000]]
            return words, boxes

        return _run_easyocr
    raise ValueError(f"Unsupported ocr_engine: {ocr_engine}")


def _load_splits_via_datasets(dataset_name: str) -> Dict[str, List[Tuple[Image.Image, int]]]:
    from datasets import load_dataset

    ds_dict = load_dataset(dataset_name)
    out: Dict[str, List[Tuple[Image.Image, int]]] = {}
    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
    for raw_split, ds in ds_dict.items():
        norm_split = split_map.get(raw_split, raw_split)
        items: List[Tuple[Image.Image, int]] = []
        for row in ds:
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            else:
                img = img.convert("RGB")
            items.append((img, int(row["label"])))
        out[norm_split] = items
    return out


def _interleave_by_label(items: List[Tuple[object, int]]) -> List[Tuple[object, int]]:
    buckets: Dict[int, List[Tuple[object, int]]] = {}
    for item in items:
        lbl = int(item[1])
        buckets.setdefault(lbl, []).append(item)
    ordered_labels = sorted(buckets.keys())
    interleaved: List[Tuple[object, int]] = []
    max_len = max((len(buckets[l]) for l in ordered_labels), default=0)
    for pos in range(max_len):
        for lbl in ordered_labels:
            if pos < len(buckets[lbl]):
                interleaved.append(buckets[lbl][pos])
    return interleaved


def _build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        status=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _http_get_with_retry(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, object]] = None,
    timeout: int = 60,
    attempts: int = 5,
) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < attempts - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(1.5 * (2 ** attempt))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed GET request for {url}")


def _load_splits_via_parquet(
    dataset_name: str,
    max_samples_per_split: Optional[int] = None,
) -> Dict[str, List[Tuple[Image.Image, int]]]:
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")
    parquet_files = sorted(f for f in repo_files if f.endswith(".parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {dataset_name}")

    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
    grouped_files: Dict[str, List[str]] = {}
    for pf in parquet_files:
        stem = Path(pf).name.lower()
        raw_split = "train"
        for key in ("train", "validation", "val", "test"):
            if stem.startswith(f"{key}-") or f"/{key}/" in pf.lower():
                raw_split = key
                break
        norm_split = split_map.get(raw_split, raw_split)
        grouped_files.setdefault(norm_split, []).append(pf)

    out: Dict[str, List[Tuple[Image.Image, int]]] = {}
    for norm_split, files in grouped_files.items():
        raw_rows: List[Tuple[bytes, int]] = []
        for pf in files:
            local_pf = hf_hub_download(repo_id=dataset_name, repo_type="dataset", filename=pf)
            table = pq.read_table(local_pf, columns=["image", "label"])
            images_col = table["image"].to_pylist()
            labels_col = table["label"].to_pylist()
            for img_val, lbl in zip(images_col, labels_col):
                img_bytes = img_val["bytes"] if isinstance(img_val, dict) else img_val
                raw_rows.append((img_bytes, int(lbl)))
        raw_rows = _interleave_by_label(raw_rows)
        if max_samples_per_split is not None:
            raw_rows = raw_rows[:max_samples_per_split]
        out[norm_split] = [
            (Image.open(io.BytesIO(img_bytes)).convert("RGB"), lbl)
            for img_bytes, lbl in raw_rows
        ]
    return out


def _fetch_split_sizes(
    dataset_name: str,
    session: Optional[requests.Session] = None,
) -> Dict[Tuple[str, str], int]:
    sess = session or _build_http_session()
    try:
        resp = _http_get_with_retry(
            sess,
            "https://datasets-server.huggingface.co/size",
            params={"dataset": dataset_name},
            timeout=30,
        )
        splits = resp.json().get("size", {}).get("splits", [])
        return {(s["config"], s["split"]): int(s["num_rows"]) for s in splits}
    except Exception:
        return {}


def _load_splits_via_hf_api(
    dataset_name: str,
    max_samples_per_split: Optional[int] = None,
) -> Dict[str, List[Tuple[Image.Image, int]]]:
    session = _build_http_session()
    splits_resp = _http_get_with_retry(
        session,
        "https://datasets-server.huggingface.co/splits",
        params={"dataset": dataset_name},
        timeout=30,
    )
    splits_info = splits_resp.json().get("splits", [])
    split_sizes = _fetch_split_sizes(dataset_name, session=session)
    split_map = {"train": "train", "validation": "val", "val": "val", "test": "test"}
    out: Dict[str, List[Tuple[Image.Image, int]]] = {}

    for entry in splits_info:
        config_name = entry["config"]
        raw_split = entry["split"]
        norm_split = split_map.get(raw_split, raw_split)
        total_rows = split_sizes.get((config_name, raw_split), 0)
        meta_rows: List[Tuple[str, int]] = []

        if max_samples_per_split is not None and total_rows > 2000 and max_samples_per_split < total_rows:
            step = float(total_rows - 1) / float(max(1, max_samples_per_split - 1))
            offsets = sorted({int(round(i * step)) for i in range(max_samples_per_split)})
            for off in offsets:
                resp = _http_get_with_retry(
                    session,
                    "https://datasets-server.huggingface.co/rows",
                    params={
                        "dataset": dataset_name,
                        "config": config_name,
                        "split": raw_split,
                        "offset": off,
                        "length": 1,
                    },
                    timeout=60,
                )
                rows = resp.json().get("rows", [])
                if rows:
                    row = rows[0]["row"]
                    meta_rows.append((row["image"]["src"], int(row["label"])))
        else:
            offset = 0
            page_size = 100
            while True:
                if total_rows > 0 and offset >= total_rows:
                    break
                req_len = min(page_size, total_rows - offset) if total_rows > 0 else page_size
                resp = _http_get_with_retry(
                    session,
                    "https://datasets-server.huggingface.co/rows",
                    params={
                        "dataset": dataset_name,
                        "config": config_name,
                        "split": raw_split,
                        "offset": offset,
                        "length": req_len,
                    },
                    timeout=60,
                )
                body = resp.json()
                if total_rows <= 0 and "num_rows_total" in body:
                    total_rows = int(body["num_rows_total"])
                rows = body.get("rows", [])
                if not rows:
                    break
                for row_wrap in rows:
                    row = row_wrap["row"]
                    meta_rows.append((row["image"]["src"], int(row["label"])))
                offset += len(rows)
                if len(rows) < req_len or (total_rows > 0 and offset >= total_rows):
                    break

        meta_rows = _interleave_by_label(meta_rows)
        if max_samples_per_split is not None:
            meta_rows = meta_rows[:max_samples_per_split]

        def _download_one(row_pair: Tuple[str, int]) -> Tuple[Image.Image, int]:
            src_url, label = row_pair
            img_resp = _http_get_with_retry(session, src_url, timeout=60)
            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
            return img, label

        with ThreadPoolExecutor(max_workers=6) as pool:
            items = list(
                tqdm(
                    pool.map(_download_one, meta_rows),
                    total=len(meta_rows),
                    desc=f"Download [{norm_split}]",
                )
            )
        out[norm_split] = items
    return out


def _ensure_train_val_test_splits(
    raw_splits: Dict[str, List[Tuple[Image.Image, int]]],
    seed: int = 42,
) -> Dict[str, List[Tuple[Image.Image, int]]]:
    if "train" in raw_splits and "val" in raw_splits:
        return raw_splits
    if "train" in raw_splits and "val" not in raw_splits:
        all_items = raw_splits["train"]
        indices = list(range(len(all_items)))
        labels = [item[1] for item in all_items]
        if "test" in raw_splits:
            tr_idx, va_idx = train_test_split(
                indices, test_size=0.15, random_state=seed, stratify=labels if len(set(labels)) > 1 else None
            )
            return {
                "train": [all_items[i] for i in tr_idx],
                "val": [all_items[i] for i in va_idx],
                "test": raw_splits["test"],
            }
        tr_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=labels if len(set(labels)) > 1 else None
        )
        temp_labels = [labels[i] for i in temp_idx]
        va_idx, te_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=temp_labels if len(set(temp_labels)) > 1 else None
        )
        return {
            "train": [all_items[i] for i in tr_idx],
            "val": [all_items[i] for i in va_idx],
            "test": [all_items[i] for i in te_idx],
        }
    return raw_splits


def _load_existing_cache_ids(cache_file: Path) -> set[int]:
    done: set[int] = set()
    if not cache_file.exists():
        return done
    with open(cache_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(int(rec["id"]))
            except Exception:
                continue
    return done


def prepare_rvl_cdip_subset(
    output_dir: Union[str, Path],
    dataset_name: str = "jordyvl/rvl_cdip_100_examples_per_class",
    ocr_engine: str = "tesseract",
    max_samples_per_split: Optional[int] = None,
    seed: int = 42,
    source_local_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, int]:
    out_root = Path(output_dir)
    images_root = out_root / "images"
    labels_root = out_root / "labels"
    ocr_root = out_root / "ocr_cache"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    ocr_root.mkdir(parents=True, exist_ok=True)

    ocr_fn = _build_ocr_callable(ocr_engine)

    if source_local_dir is not None:
        src_root = Path(source_local_dir)
        splits_data: Dict[str, List[Tuple[Image.Image, int]]] = {}
        for sp in ("train", "val", "test"):
            lbl_path = src_root / "labels" / f"{sp}.txt"
            if not lbl_path.exists():
                continue
            entries: List[Tuple[str, int]] = []
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    entries.append((parts[0], int(parts[1])))
            entries = _interleave_by_label(entries)
            if max_samples_per_split is not None:
                entries = entries[:max_samples_per_split]
            splits_data[sp] = [
                (Image.open(src_root / "images" / rel_p).convert("RGB"), lbl)
                for rel_p, lbl in entries
            ]
    else:
        try:
            raw_splits = _load_splits_via_datasets(dataset_name)
        except ImportError:
            try:
                raw_splits = _load_splits_via_parquet(
                    dataset_name=dataset_name,
                    max_samples_per_split=max_samples_per_split,
                )
            except ImportError:
                raw_splits = _load_splits_via_hf_api(
                    dataset_name=dataset_name,
                    max_samples_per_split=max_samples_per_split,
                )
        splits_data = _ensure_train_val_test_splits(raw_splits, seed=seed)

    counts: Dict[str, int] = {}
    for split_name, items in splits_data.items():
        items = _interleave_by_label(items)
        if max_samples_per_split is not None:
            items = items[:max_samples_per_split]

        split_img_dir = images_root / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        cache_file = ocr_root / f"{split_name}.jsonl"
        label_file = labels_root / f"{split_name}.txt"
        existing_ids = _load_existing_cache_ids(cache_file)

        with open(cache_file, "a", encoding="utf-8") as cf, open(label_file, "w", encoding="utf-8") as lf:
            for idx, (pil_img, label) in enumerate(tqdm(items, desc=f"OCR [{split_name}]")):
                rel_img_for_labels = f"{split_name}/{idx:05d}.png"
                rel_img_path = f"images/{rel_img_for_labels}"
                full_img_path = out_root / rel_img_path
                if not full_img_path.exists():
                    pil_img.save(full_img_path)
                lf.write(f"{rel_img_for_labels} {label}\n")
                if idx in existing_ids:
                    continue
                words, boxes = ocr_fn(pil_img)
                record = {
                    "id": idx,
                    "image_path": rel_img_path,
                    "label": int(label),
                    "words": words,
                    "boxes": boxes,
                }
                cf.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split_name] = len(items)
    return counts
