import csv
import json
import math
import os
import re
import sys
from datetime import datetime

try:
    from .tar_streaming import safe_relative_stem_from_tar_name
    from .size_threshold import DEFAULT_SIZE_THRESHOLD_PX, parse_size_threshold
except ImportError:  # pragma: no cover - allows direct test imports without package context
    sys.path.insert(0, os.path.dirname(__file__))
    from tar_streaming import safe_relative_stem_from_tar_name
    from size_threshold import DEFAULT_SIZE_THRESHOLD_PX, parse_size_threshold


DEFAULT_TAR_SAMPLE_RATIO = 1.0


def _parse_blob_url(url: str):
    url = (url or "").strip()
    if not url:
        return None
    url_no_q = url.split("?", 1)[0]
    match = re.match(r"^(https?://[^/]+)(?:/([^/]+)(?:/(.*))?)?$", url_no_q)
    if not match:
        return None
    account_url = match.group(1)
    container = match.group(2)
    path = (match.group(3) or "").lstrip("/")
    return account_url, container, path


def _build_resume_summary_blob_name(previous_job_url: str, tar_blob_name: str) -> str | None:
    parsed = _parse_blob_url(previous_job_url)
    if not parsed:
        return None
    _, container, path = parsed
    if not container:
        return None
    resume_prefix = (path or "").strip().strip("/")
    if resume_prefix.endswith("/summary.json"):
        if "/per_tar/" in resume_prefix:
            resume_prefix = resume_prefix.split("/per_tar/", 1)[0]
        else:
            resume_prefix = resume_prefix.rsplit("/", 1)[0]
    if resume_prefix:
        resume_prefix = f"{resume_prefix}/"
    relative_stem = safe_relative_stem_from_tar_name(tar_blob_name).replace("\\", "/")
    return f"{resume_prefix}per_tar/{relative_stem}/summary.json"


def _resume_run_location(previous_job_url: str):
    parsed = _parse_blob_url(previous_job_url)
    if not parsed:
        raise ValueError("Resume URL must be a valid Azure Blob URL")
    account_url, container, path = parsed
    if not container or not path:
        raise ValueError("Resume URL must point to a run folder or summary.json")

    path = path.strip("/")
    if "/per_tar/" in f"/{path}/":
        path = path.split("/per_tar/", 1)[0]
    elif path.lower().endswith("/run_summary.json") or path.lower().endswith("/summary.json"):
        path = path.rsplit("/", 1)[0]

    parts = [part for part in path.split("/") if part]
    if not parts:
        raise ValueError("Resume URL does not identify a run folder")
    output_slug = parts[-1]
    output_prefix = "/".join(parts[:-1])
    run_blob_prefix = "/".join(parts)
    return account_url, container, output_prefix, output_slug, run_blob_prefix


def load_resume_settings(bsc, previous_job_url: str) -> dict:
    """Load inference defaults and output location from a previous GUI run."""
    account_url, output_container, output_prefix, output_slug, run_blob_prefix = (
        _resume_run_location(previous_job_url)
    )
    container_client = bsc.get_container_client(output_container)
    if previous_job_url.rstrip("/").lower().endswith("/summary.json"):
        _, _, url_path = _parse_blob_url(previous_job_url)
        candidate_names = [url_path.strip("/")]
    else:
        candidate_names = []

    summary_names = [
        str(getattr(blob, "name", "") or "")
        for blob in container_client.list_blobs(
            name_starts_with=f"{run_blob_prefix}/per_tar/"
        )
    ]
    summary_names = sorted(
        name for name in summary_names
        if name.rsplit("/", 1)[-1].lower() == "summary.json"
    )
    candidate_names.extend(name for name in summary_names if name not in candidate_names)

    summary_blob_name = next(
        (name for name in candidate_names if name.rsplit("/", 1)[-1].lower() == "summary.json"),
        None,
    )
    run_summary_name = f"{run_blob_prefix}/run_summary.json"
    run_summary_client = container_client.get_blob_client(run_summary_name)
    summary = None
    if summary_blob_name:
        summary = json.loads(
            container_client.get_blob_client(summary_blob_name).download_blob().readall()
        )
    else:
        if not hasattr(run_summary_client, "exists") or not run_summary_client.exists():
            raise RuntimeError(
                f"No summary.json found in resumed run {output_container}/{run_blob_prefix}"
            )
        summary = json.loads(run_summary_client.download_blob().readall())
        summary_blob_name = run_summary_name

    if not isinstance(summary, dict):
        raise ValueError(f"Resume summary {summary_blob_name} must contain a JSON object")

    if "container" not in summary:
        selected_tars = summary.get("selected_tars") or []
        if selected_tars:
            summary.update(selected_tars[0])

    source_container = str(summary.get("container") or "").strip()
    if not source_container:
        raise ValueError(f"Resume summary {summary_blob_name} has no source container")

    raw_sample_ratio = summary.get("sample_ratio_requested")
    if raw_sample_ratio is None:
        raw_sample_ratio = summary.get("sample_ratio")
    if raw_sample_ratio is None and summary.get("sample_percent_requested") is not None:
        raw_sample_ratio = float(summary["sample_percent_requested"]) / 100.0
    sample_ratio = parse_sample_ratio(raw_sample_ratio)

    try:
        temperature = float(summary["temperature"]) if summary.get("temperature") is not None else None
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Resume summary {summary_blob_name} has an invalid temperature") from exc
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError(f"Resume summary {summary_blob_name} has an invalid temperature")

    run_summary = None
    if summary_blob_name == run_summary_name:
        run_summary = summary
    elif hasattr(run_summary_client, "exists") and run_summary_client.exists():
        run_summary = json.loads(run_summary_client.download_blob().readall())
        if not isinstance(run_summary, dict):
            raise ValueError(f"Resume summary {run_summary_name} must contain a JSON object")

    targets = run_summary.get("targets") if isinstance(run_summary, dict) else None
    if not isinstance(targets, list) or not targets:
        targets = [{
            "container": source_container,
            "prefix": str(summary.get("selected_prefix") or ""),
        }]
    selected_tars = run_summary.get("selected_tars") if isinstance(run_summary, dict) else None
    if not isinstance(selected_tars, list):
        selected_tars = None

    return {
        "account_url": account_url,
        "output_container": output_container,
        "output_prefix": output_prefix,
        "output_slug": output_slug,
        "summary_blob": summary_blob_name,
        "summary": summary,
        "container": source_container,
        "prefix": str(summary.get("selected_prefix") or ""),
        "targets": targets,
        "selected_tars": selected_tars,
        "sort_by_size": run_summary.get("sort_by_size") if isinstance(run_summary, dict) else None,
        "model_blob": summary.get("model_blob"),
        "model_runid": summary.get("model_runid"),
        "sample_ratio": sample_ratio,
        "size_threshold_px": parse_size_threshold(summary.get("size_threshold_px")),
        "temperature": temperature,
    }


def _resume_summary_exists(
    bsc, previous_job_url: str, tar_blob_name: str,
    size_threshold_px=DEFAULT_SIZE_THRESHOLD_PX,
    *,
    sample_ratio: float | None = None,
    source_container: str | None = None,
    selected_prefix: str | None = None,
    model_blob: str | None = None,
    temperature: float | None = None,
) -> bool:
    if not previous_job_url or not tar_blob_name:
        return False
    parsed = _parse_blob_url(previous_job_url)
    if not parsed:
        return False
    _, container, _ = parsed
    if not container:
        return False
    blob_name = _build_resume_summary_blob_name(previous_job_url, tar_blob_name)
    if not blob_name:
        return False

    container_client = bsc.get_container_client(container)
    blob_client = container_client.get_blob_client(blob_name)
    if hasattr(blob_client, "exists"):
        exists = bool(blob_client.exists())
    else:
        exists = any(blob.name == blob_name for blob in container_client.list_blobs(name_starts_with=blob_name))
    if not exists:
        return False
    summary = json.loads(blob_client.download_blob().readall())
    if parse_size_threshold(summary.get("size_threshold_px")) != size_threshold_px:
        return False
    if sample_ratio is not None:
        previous_ratio = summary.get("sample_ratio_requested")
        if previous_ratio is None or not math.isclose(
            float(previous_ratio), sample_ratio, rel_tol=1e-8, abs_tol=1e-9
        ):
            return False
    if source_container is not None and summary.get("container") != source_container:
        return False
    if selected_prefix is not None and str(summary.get("selected_prefix") or "") != str(selected_prefix or ""):
        return False
    if model_blob is not None and summary.get("model_blob") != model_blob:
        return False
    if temperature is not None:
        previous_temperature = summary.get("temperature")
        if previous_temperature is None or not math.isclose(
            float(previous_temperature), float(temperature), rel_tol=1e-8, abs_tol=1e-9
        ):
            return False
    return True


def parse_sample_ratio(
    value,
    default: float = DEFAULT_TAR_SAMPLE_RATIO,
    *,
    allow_whole_number_percent: bool = False,
) -> float:
    """Parse sample ratio values from CSV/UI, accepting fractions and ``%`` suffixed values."""
    if value is None:
        return float(default)

    if isinstance(value, (int, float)):
        ratio = float(value)
    else:
        text = str(value).strip()
        if not text:
            return float(default)
        if text.endswith("%"):
            ratio = float(text[:-1].strip()) / 100.0
        else:
            ratio = float(text)
            if allow_whole_number_percent and ratio > 1.0 and ratio <= 100.0:
                ratio /= 100.0

    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"sample ratio must be between 0 and 1 inclusive, got {value!r}")
    return ratio


def load_tar_selection_csv(csv_path: str):
    """Return selected (container, tar_blob, prefix, sample_ratio, size_threshold_px) rows."""
    selected = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("process", "").strip().upper() == "Y":
                has_sample_ratio = "sample_ratio" in row
                has_sample_percent = "sample_percent" in row
                sample_value = (
                    row.get("sample_ratio")
                    if has_sample_ratio
                    else row.get("sample_percent")
                    if has_sample_percent
                    else None
                )
                try:
                    sample_ratio = parse_sample_ratio(
                        sample_value,
                        allow_whole_number_percent=has_sample_percent and not has_sample_ratio,
                    )
                except ValueError as e:
                    tar_name = row.get("tar_blob") or "(unknown TAR)"
                    raise RuntimeError(f"Invalid sample ratio for {tar_name}: {e}") from e
                try:
                    size_threshold_px = parse_size_threshold(row.get("size_threshold_px"))
                except ValueError as e:
                    tar_name = row.get("tar_blob") or "(unknown TAR)"
                    raise RuntimeError(f"Invalid size threshold for {tar_name}: {e}") from e
                selected.append((
                    row["container"], row["tar_blob"], row.get("prefix") or "",
                    sample_ratio, size_threshold_px,
                ))

    if not selected:
        raise RuntimeError("No TARs marked for processing in CSV")

    return selected


def extract_datetime_from_tar_name(name: str):
    """Attempt to extract YYYYMMDDHHMMSS from tar names. Returns ISO string or None."""
    m = re.search(r"(\d{8})(\d{6})", name)
    if not m:
        return None
    dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_target_specs(targets=None, *, container: str | None = None, prefix: str = ""):
    if targets is None:
        if container:
            return [{"container": container, "prefix": prefix or ""}]
        return []

    normalized = []
    seen = set()
    for entry in targets:
        if isinstance(entry, dict):
            target_container = entry.get("container")
            target_prefix = entry.get("prefix") or entry.get("selected_prefix") or ""
        elif isinstance(entry, (list, tuple)):
            if len(entry) >= 2:
                target_container, target_prefix = entry[0], entry[1]
            elif len(entry) == 1:
                target_container, target_prefix = entry[0], ""
            else:
                raise RuntimeError(f"Unexpected container target spec: {entry!r}")
        else:
            target_container = entry
            target_prefix = ""

        target_container = str(target_container).strip() if target_container is not None else ""
        target_prefix = str(target_prefix or "")
        if not target_container:
            raise RuntimeError(f"Container target missing container name: {entry!r}")
        key = (target_container, target_prefix)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"container": target_container, "prefix": target_prefix})

    return normalized


def generate_tar_selection_csv(
    *,
    bsc,
    container: str | None = None,
    prefix: str = "",
    csv_path: str,
    preselect_ratio: float = 0.10,
    sample_ratio: float = DEFAULT_TAR_SAMPLE_RATIO,
    size_threshold_px: int | str = DEFAULT_SIZE_THRESHOLD_PX,
    targets=None,
    resume_previous_job_url: str | None = None,
    resume_targets=None,
    resume_selected_tars=None,
    model_blob: str | None = None,
    temperature: float | None = None,
    log_cb=None,
):
    """Enumerate TAR blobs and emit editable CSV with Y/N process flag."""
    target_specs = _normalize_target_specs(targets, container=container, prefix=prefix)
    if not target_specs:
        raise RuntimeError("No container targets provided")

    sample_ratio = parse_sample_ratio(sample_ratio)
    size_threshold_px = parse_size_threshold(size_threshold_px)

    rows_written = 0
    preselected_total = 0
    target_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tar_blob", "container", "prefix", "datetime_utc", "process", "sample_ratio", "size_threshold_px"],
        )
        writer.writeheader()

        for target in target_specs:
            target_container = target["container"]
            target_prefix = target["prefix"]
            cc = bsc.get_container_client(target_container)
            saved_target_keys = {
                (str(item.get("container") or "").strip(), str(item.get("prefix") or item.get("selected_prefix") or ""))
                for item in (resume_targets or [])
                if isinstance(item, dict)
            }
            target_key = (target_container, target_prefix)
            saved_tar_entries = [
                item for item in (resume_selected_tars or [])
                if isinstance(item, dict)
                and str(item.get("container") or "").strip() == target_container
                and str(item.get("selected_prefix") or item.get("prefix") or "") == target_prefix
            ]
            if resume_selected_tars is not None and target_key in saved_target_keys:
                tar_names = sorted({
                    str(item.get("source_tar_blob") or item.get("tar_blob") or "")
                    for item in saved_tar_entries
                    if item.get("source_tar_blob") or item.get("tar_blob")
                })
            else:
                tar_names = [
                    b.name for b in cc.list_blobs(name_starts_with=target_prefix)
                    if b.name.lower().endswith((".tar", ".tar.gz", ".tgz"))
                ]

            if not tar_names:
                if len(target_specs) == 1:
                    raise RuntimeError("No TAR blobs found")
                if log_cb:
                    log_cb(f"Skipping target {target_container}/{target_prefix}: no TAR blobs found")
                continue

            tar_rows = []
            for name in tar_names:
                dt = extract_datetime_from_tar_name(name)
                dt_key = dt or "9999-12-31T23:59:59Z"
                tar_rows.append((dt_key, name))

            tar_rows.sort(key=lambda x: x[0])
            N = len(tar_rows)
            preserve_resume_selection = (
                resume_selected_tars is not None and target_key in saved_target_keys
            )
            k = N if preserve_resume_selection else max(1, int(round(N * preselect_ratio)))

            if preserve_resume_selection or k >= N:
                preselected = {name for _, name in tar_rows}
            else:
                step = (N - 1) / max(k - 1, 1)
                selected_indices = {round(i * step) for i in range(k)}
                preselected = {tar_rows[i][1] for i in selected_indices}

            target_count += 1
            for name in sorted(tar_names):
                resume_skipped = False
                if resume_previous_job_url:
                    try:
                        resume_skipped = _resume_summary_exists(
                            bsc,
                            resume_previous_job_url,
                            name,
                            size_threshold_px,
                            sample_ratio=sample_ratio,
                            source_container=target_container,
                            selected_prefix=target_prefix,
                            model_blob=model_blob,
                            temperature=temperature,
                        )
                    except Exception as exc:
                        if log_cb:
                            log_cb(f"WARNING: failed to verify resume output for {name}: {exc}")
                        resume_skipped = False
                process_value = "N"
                if not resume_skipped:
                    process_value = "Y" if name in preselected else "N"
                if resume_skipped and log_cb:
                    log_cb(
                        f"Skipping {target_container}/{name} because a resume summary was found at "
                        f"{_build_resume_summary_blob_name(resume_previous_job_url, name)}"
                    )
                if process_value == "Y":
                    preselected_total += 1
                writer.writerow({
                    "tar_blob": name,
                    "container": target_container,
                    "prefix": target_prefix,
                    "datetime_utc": extract_datetime_from_tar_name(name),
                    "process": process_value,
                    "sample_ratio": sample_ratio,
                    "size_threshold_px": size_threshold_px,
                })
                rows_written += 1

    if rows_written == 0:
        raise RuntimeError("No TAR blobs found")

    if log_cb:
        log_cb(
            f"Wrote TAR selection CSV: {csv_path} "
            f"({preselected_total}/{rows_written} TARs preselected across {target_count} target(s), "
            f"within-TAR sample {sample_ratio:.2%}, size threshold {size_threshold_px} px)"
        )
