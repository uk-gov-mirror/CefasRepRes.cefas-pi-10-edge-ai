import csv
import json
import os
import sys
import tarfile
from io import BytesIO

import pytest

# Allow direct import of the module without triggering GUI dependencies
_module_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "gui"
)
sys.path.insert(0, _module_path)

import csv_utils as csv_utils_module
from csv_utils import (
    DEFAULT_TAR_SAMPLE_RATIO,
    generate_tar_selection_csv,
    load_resume_settings,
    load_tar_selection_csv,
    parse_sample_ratio,
)
from tar_streaming import build_tar_stream_segments, should_sample_tar_member
from tar_streaming import iter_per_tar_prediction_paths, iter_tar_image_bytes_from_blob


class _FakeContainerClient:
    def __init__(self, blob_names, existing_blobs=None):
        self._blob_names = blob_names
        self._existing_blobs = existing_blobs if isinstance(existing_blobs, dict) else set(existing_blobs or [])

    def list_blobs(self, name_starts_with=""):
        return [
            type("Blob", (), {"name": name})()
            for name in self._blob_names
            if name.startswith(name_starts_with)
        ]

    def get_blob_client(self, name):
        if isinstance(self._existing_blobs, dict):
            return _FakeBlobClient(self._existing_blobs.get(name, False))
        return _FakeBlobClient(name in self._existing_blobs)


class _FakeBlobServiceClient:
    def __init__(self, blob_names, existing_blob_names=None):
        self._blob_names = blob_names
        self._existing_blob_names = existing_blob_names or {}

    def get_container_client(self, name):
        if isinstance(self._blob_names, dict):
            blob_names = self._blob_names.get(name, [])
        else:
            blob_names = self._blob_names
        if isinstance(self._existing_blob_names, dict):
            existing_blobs = self._existing_blob_names.get(name, [])
        else:
            existing_blobs = self._existing_blob_names
        return _FakeContainerClient(blob_names, existing_blobs=existing_blobs)


class _FakeDownloader:
    def __init__(self, payload: bytes):
        self._payload = payload

    def chunks(self):
        yield self._payload

    def readall(self):
        return self._payload


class _FakeBlobClient:
    def __init__(self, payload_or_exists):
        if isinstance(payload_or_exists, bool):
            self._exists = payload_or_exists
            self._payload = b"{}"
        else:
            self._exists = True
            self._payload = (
                payload_or_exists
                if isinstance(payload_or_exists, bytes)
                else json.dumps(payload_or_exists).encode("utf-8")
            )
        self.download_calls = []

    def exists(self):
        return self._exists is not None and self._exists

    def download_blob(self, **kwargs):
        self.download_calls.append(kwargs)
        offset = int(kwargs.get("offset", 0) or 0)
        length = kwargs.get("length")
        if length is None:
            return _FakeDownloader(self._payload[offset:])
        return _FakeDownloader(self._payload[offset: offset + int(length)])


def _build_tar_payload(members):
    bio = BytesIO()
    with tarfile.open(fileobj=bio, mode="w") as tf:
        for member_name, payload in members:
            data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
            info = tarfile.TarInfo(name=member_name)
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
    return bio.getvalue()


def test_load_tar_selection_csv_defaults_sample_ratio_when_column_absent(tmp_path):
    csv_path = tmp_path / "tar_selection.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tar_blob", "container", "prefix", "datetime_utc", "process"])
        writer.writeheader()
        writer.writerow({
            "tar_blob": "sample.tar",
            "container": "source",
            "prefix": "prefix/",
            "datetime_utc": "2026-07-15T10:00:00Z",
            "process": "Y",
        })

    assert load_tar_selection_csv(str(csv_path)) == [
        ("source", "sample.tar", "prefix/", DEFAULT_TAR_SAMPLE_RATIO, "inf")
    ]


def test_load_tar_selection_csv_parses_ratio_and_percent_formats(tmp_path):
    ratio_csv = tmp_path / "ratio.csv"
    with open(ratio_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["tar_blob", "container", "prefix", "datetime_utc", "process", "sample_ratio"],
        )
        writer.writeheader()
        writer.writerow({
            "tar_blob": "ratio.tar",
            "container": "source",
            "prefix": "",
            "datetime_utc": "",
            "process": "Y",
            "sample_ratio": "1%",
        })

    percent_csv = tmp_path / "percent.csv"
    with open(percent_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["tar_blob", "container", "prefix", "datetime_utc", "process", "sample_percent"],
        )
        writer.writeheader()
        writer.writerow({
            "tar_blob": "percent.tar",
            "container": "source",
            "prefix": "",
            "datetime_utc": "",
            "process": "Y",
            "sample_percent": "25",
        })

    assert load_tar_selection_csv(str(ratio_csv)) == [("source", "ratio.tar", "", 0.01, "inf")]
    assert load_tar_selection_csv(str(percent_csv)) == [("source", "percent.tar", "", 0.25, "inf")]


@pytest.mark.parametrize("threshold,expected", [("inf", "inf"), ("", "inf"), ("64", 64), ("64.0", 64)])
def test_load_tar_selection_csv_parses_size_threshold(tmp_path, threshold, expected):
    csv_path = tmp_path / "sizes.csv"
    csv_path.write_text(
        f"container,tar_blob,process,sample_ratio,size_threshold_px\nsource,sample.tar,Y,0.25,{threshold}\n"
    )
    assert load_tar_selection_csv(str(csv_path)) == [("source", "sample.tar", "", 0.25, expected)]


def test_load_tar_selection_csv_reports_invalid_threshold_with_tar_name(tmp_path):
    csv_path = tmp_path / "sizes.csv"
    csv_path.write_text("container,tar_blob,process,size_threshold_px\nsource,bad.tar,Y,-1\n")
    with pytest.raises(RuntimeError, match="Invalid size threshold for bad.tar"):
        load_tar_selection_csv(str(csv_path))


def test_generate_tar_selection_csv_writes_sample_ratio_column(tmp_path):
    csv_path = tmp_path / "generated.csv"
    bsc = _FakeBlobServiceClient([
        "prefix/a_20260715100000.tar",
        "prefix/b_20260715110000.tgz",
        "prefix/ignore.txt",
    ])

    generate_tar_selection_csv(
        bsc=bsc,
        container="source",
        prefix="prefix/",
        csv_path=str(csv_path),
        preselect_ratio=0.5,
        sample_ratio=0.25,
    )

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert "sample_ratio" in rows[0]
    assert {row["sample_ratio"] for row in rows} == {"0.25"}
    assert {row["size_threshold_px"] for row in rows} == {"inf"}
    assert sum(1 for row in rows if row["process"] == "Y") == 1


def test_generate_tar_selection_csv_supports_multiple_targets(tmp_path):
    csv_path = tmp_path / "generated_multi.csv"
    bsc = _FakeBlobServiceClient({
        "source_a": ["prefix/a_20260715100000.tar", "prefix/ignore.txt"],
        "source_b": ["other/b_20260715110000.tgz"],
    })

    generate_tar_selection_csv(
        bsc=bsc,
        targets=[
            {"container": "source_a", "prefix": "prefix/"},
            {"container": "source_b", "prefix": "other/"},
        ],
        csv_path=str(csv_path),
        preselect_ratio=1.0,
        sample_ratio=0.5,
        size_threshold_px=64,
    )

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 2
    assert {row["container"] for row in rows} == {"source_a", "source_b"}
    assert [row["prefix"] for row in rows] == ["prefix/", "other/"]
    assert {row["sample_ratio"] for row in rows} == {"0.5"}
    assert {row["size_threshold_px"] for row in rows} == {"64"}


def test_generate_tar_selection_csv_skips_rows_with_resume_summary(tmp_path):
    csv_path = tmp_path / "generated_resume.csv"
    bsc = _FakeBlobServiceClient(
        {
            "source": ["prefix/a_20260715100000.tar"],
            "ml-prediction-results": [],
        },
        existing_blob_names={
            "ml-prediction-results": {
                "runs/slug/per_tar/prefix/a_20260715100000/summary.json": {
                    "container": "source",
                    "size_threshold_px": "inf",
                    "sample_ratio_requested": 0.25,
                    "selected_prefix": "prefix/",
                },
            },
        },
    )

    generate_tar_selection_csv(
        bsc=bsc,
        container="source",
        prefix="prefix/",
        csv_path=str(csv_path),
        preselect_ratio=1.0,
        sample_ratio=0.25,
        resume_previous_job_url="https://example.blob.core.windows.net/ml-prediction-results/runs/slug/",
    )

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert rows[0]["process"] == "N"
    assert rows[0]["sample_ratio"] == "0.25"


def test_generate_tar_selection_csv_reprocesses_when_resume_sample_ratio_differs(tmp_path):
    csv_path = tmp_path / "generated_resume.csv"
    summary_name = "runs/slug/per_tar/prefix/a_20260715100000/summary.json"
    bsc = _FakeBlobServiceClient(
        {
            "source": ["prefix/a_20260715100000.tar"],
            "ml-prediction-results": [],
        },
        existing_blob_names={
            "ml-prediction-results": {
                summary_name: {
                    "container": "source",
                    "size_threshold_px": "inf",
                    "sample_ratio_requested": 0.25,
                    "selected_prefix": "prefix/",
                },
            },
        },
    )

    generate_tar_selection_csv(
        bsc=bsc,
        container="source",
        prefix="prefix/",
        csv_path=str(csv_path),
        preselect_ratio=1.0,
        sample_ratio=0.5,
        resume_previous_job_url="https://example.blob.core.windows.net/ml-prediction-results/runs/slug/",
    )

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["process"] == "Y"


def test_generate_tar_selection_csv_preserves_previous_tar_selection(tmp_path):
    csv_path = tmp_path / "generated_resume.csv"
    bsc = _FakeBlobServiceClient({
        "source": [
            "prefix/selected_20260715100000.tar",
            "prefix/not-selected_20260715110000.tar",
        ],
        "ml-prediction-results": [],
    })

    generate_tar_selection_csv(
        bsc=bsc,
        targets=[{"container": "source", "prefix": "prefix/"}],
        csv_path=str(csv_path),
        preselect_ratio=1.0,
        resume_previous_job_url="https://example.blob.core.windows.net/ml-prediction-results/runs/slug/",
        resume_targets=[{"container": "source", "prefix": "prefix/"}],
        resume_selected_tars=[{
            "container": "source",
            "source_tar_blob": "prefix/selected_20260715100000.tar",
            "selected_prefix": "prefix/",
            "sample_ratio": 1.0,
            "size_threshold_px": "inf",
        }],
    )

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["tar_blob"] for row in rows] == ["prefix/selected_20260715100000.tar"]


def test_incomplete_resume_enumerates_target_and_skips_only_existing_tars(tmp_path):
    summary_name = "runs/slug/per_tar/2025-06-01/0734/summary.json"
    summary = {
        "container": "source",
        "source_tar_blob": "2025-06-01/0734.tar",
        "selected_prefix": "",
        "model_blob": "model-old.pt",
        "sample_ratio_requested": 1.0,
        "size_threshold_px": 80,
        "temperature": 1.25,
    }
    bsc = _FakeBlobServiceClient(
        {
            "source": [
                "2025-06-01/0734.tar",
                "2025-06-01/0744.tar",
                "2025-06-01/0754.tar",
            ],
            "ml-prediction-results": [summary_name],
        },
        existing_blob_names={"ml-prediction-results": {summary_name: summary}},
    )
    resume_url = "https://example.blob.core.windows.net/ml-prediction-results/runs/slug/"
    settings = load_resume_settings(bsc, resume_url)

    assert settings["selected_tars"] is None

    csv_path = tmp_path / "incomplete_resume.csv"
    generate_tar_selection_csv(
        bsc=bsc,
        targets=settings["targets"],
        csv_path=str(csv_path),
        preselect_ratio=1.0,
        sample_ratio=settings["sample_ratio"],
        size_threshold_px=settings["size_threshold_px"],
        resume_previous_job_url=resume_url,
        resume_targets=settings["targets"],
        resume_selected_tars=settings["selected_tars"],
        model_blob=settings["model_blob"],
        temperature=settings["temperature"],
    )

    with open(csv_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [row["tar_blob"] for row in rows] == [
        "2025-06-01/0734.tar",
        "2025-06-01/0744.tar",
        "2025-06-01/0754.tar",
    ]
    assert [row["process"] for row in rows] == ["N", "Y", "Y"]


def test_resume_summary_blob_path_uses_forward_slashes(monkeypatch):
    monkeypatch.setattr(
        csv_utils_module,
        "safe_relative_stem_from_tar_name",
        lambda name: "2025-06-01\\0734",
    )

    assert csv_utils_module._build_resume_summary_blob_name(
        "https://example.blob.core.windows.net/results/runs/slug/",
        "2025-06-01/0734.tar",
    ) == "runs/slug/per_tar/2025-06-01/0734/summary.json"


def test_load_resume_settings_restores_run_defaults_and_output_location():
    summary_name = "runs/guilocalinference-old-run/per_tar/2025-06-01/0734/summary.json"
    run_summary_name = "runs/guilocalinference-old-run/run_summary.json"
    summary = {
        "container": "source",
        "source_tar_blob": "2025-06-01/0734.tar",
        "selected_prefix": "2025-06-01/",
        "model_blob": "model-old.pt",
        "model_runid": "old-model",
        "sample_ratio_requested": 0.25,
        "sample_percent_requested": 25.0,
        "size_threshold_px": 80,
        "temperature": 1.25,
    }
    bsc = _FakeBlobServiceClient(
        {"ml-prediction-results": [summary_name]},
        existing_blob_names={"ml-prediction-results": {
            summary_name: summary,
            run_summary_name: {
                "targets": [
                    {"container": "source", "prefix": "2025-06-01/"},
                    {"container": "source-2", "prefix": "other/"},
                ],
                "sort_by_size": True,
                "selected_tars": [
                    {
                        "container": "source",
                        "source_tar_blob": "2025-06-01/0734.tar",
                        "selected_prefix": "2025-06-01/",
                        "sample_ratio": 0.25,
                        "size_threshold_px": 80,
                    },
                ],
            },
        }},
    )

    settings = load_resume_settings(
        bsc,
        "https://example.blob.core.windows.net/ml-prediction-results/runs/guilocalinference-old-run/",
    )

    assert settings["account_url"] == "https://example.blob.core.windows.net"
    assert settings["output_container"] == "ml-prediction-results"
    assert settings["output_prefix"] == "runs"
    assert settings["output_slug"] == "guilocalinference-old-run"
    assert settings["container"] == "source"
    assert settings["prefix"] == "2025-06-01/"
    assert settings["model_blob"] == "model-old.pt"
    assert settings["model_runid"] == "old-model"
    assert settings["sample_ratio"] == 0.25
    assert settings["size_threshold_px"] == 80
    assert settings["temperature"] == 1.25
    assert settings["targets"][1] == {"container": "source-2", "prefix": "other/"}
    assert settings["selected_tars"][0]["source_tar_blob"] == "2025-06-01/0734.tar"
    assert settings["sort_by_size"] is True


def test_load_resume_settings_falls_back_to_run_summary():
    run_summary_name = "runs/old-run/run_summary.json"
    bsc = _FakeBlobServiceClient(
        {"ml-prediction-results": []},
        existing_blob_names={"ml-prediction-results": {
            run_summary_name: {
                "model_blob": "model-old.pt",
                "model_runid": "old-model",
                "temperature": 1.5,
                "sort_by_size": False,
                "targets": [{"container": "source", "prefix": "prefix/"}],
                "selected_tars": [{
                    "container": "source",
                    "source_tar_blob": "prefix/a.tar",
                    "selected_prefix": "prefix/",
                    "sample_ratio": 0.4,
                    "size_threshold_px": 64,
                }],
            },
        }},
    )

    settings = load_resume_settings(
        bsc,
        "https://example.blob.core.windows.net/ml-prediction-results/runs/old-run/",
    )

    assert settings["summary_blob"] == run_summary_name
    assert settings["model_blob"] == "model-old.pt"
    assert settings["sample_ratio"] == 0.4
    assert settings["size_threshold_px"] == 64
    assert settings["targets"] == [{"container": "source", "prefix": "prefix/"}]


def test_generate_tar_selection_csv_reprocesses_when_resume_threshold_differs(tmp_path):
    csv_path = tmp_path / "generated_resume.csv"
    bsc = _FakeBlobServiceClient(
        {"source": ["prefix/a_20260715100000.tar"]},
        existing_blob_names={
            "ml-prediction-results": ["runs/slug/per_tar/prefix/a_20260715100000/summary.json"],
        },
    )
    generate_tar_selection_csv(
        bsc=bsc, container="source", prefix="prefix/", csv_path=str(csv_path),
        preselect_ratio=1.0, size_threshold_px=64,
        resume_previous_job_url="https://example.blob.core.windows.net/ml-prediction-results/runs/slug/",
    )
    assert load_tar_selection_csv(str(csv_path)) == [
        ("source", "prefix/a_20260715100000.tar", "prefix/", 1.0, 64),
    ]


def test_parse_sample_ratio_accepts_fraction_and_percent_inputs():
    assert parse_sample_ratio("0.01") == pytest.approx(0.01)
    assert parse_sample_ratio("1%") == pytest.approx(0.01)
    assert parse_sample_ratio("25", allow_whole_number_percent=True) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        parse_sample_ratio("25")


def test_should_sample_tar_member_is_deterministic_and_honors_bounds():
    assert should_sample_tar_member("sample.tar", "RawImages/image1.tif", 0.0) is False
    assert should_sample_tar_member("sample.tar", "RawImages/image1.tif", 1.0) is True

    members = [f"RawImages/image_{i}.tif" for i in range(100)]
    first = [should_sample_tar_member("sample.tar", member, 0.5) for member in members]
    second = [should_sample_tar_member("sample.tar", member, 0.5) for member in members]

    assert first == second
    assert any(first)
    assert not all(first)


def test_build_tar_stream_segments_spreads_offsets_across_five_chunks():
    segments = build_tar_stream_segments(blob_size=5120, target_bytes=20, segment_count=5)

    assert segments == [(0, 4), (1024, 4), (2048, 4), (3072, 4), (4096, 4)]


def test_iter_tar_image_bytes_from_blob_applies_member_filter_before_image_read():
    tar_payload = _build_tar_payload([
        ("RawImages/keep.tif", b"keep-bytes"),
        ("RawImages/skip.tif", b"skip-bytes"),
    ])
    blob_client = _FakeBlobClient(tar_payload)

    rows = list(iter_tar_image_bytes_from_blob(
        blob_client,
        "sample.tar",
        member_filter=lambda name: name.endswith("keep.tif"),
    ))

    assert rows == [("RawImages/keep.tif", b"keep-bytes")]


def test_iter_tar_image_bytes_from_blob_supports_prefix_stream_limits():
    tar_payload = _build_tar_payload([
        ("RawImages/one.tif", b"one"),
        ("RawImages/two.tif", b"two"),
    ])
    blob_client = _FakeBlobClient(tar_payload)

    full_rows = list(iter_tar_image_bytes_from_blob(blob_client, "sample.tar"))
    limited_rows = list(iter_tar_image_bytes_from_blob(blob_client, "sample.tar", max_bytes=1024))

    assert len(full_rows) == 2
    assert len(limited_rows) < len(full_rows)
    assert limited_rows == [("RawImages/one.tif", b"one")]
    assert blob_client.download_calls[-1]["length"] == 1024


def test_iter_per_tar_prediction_paths_finds_nested_predictions(tmp_path):
    per_tar_dir = tmp_path / "per_tar"
    nested_dir = per_tar_dir / "2026-08-10" / "1030"
    nested_dir.mkdir(parents=True)
    pred_path = nested_dir / "predictions.json"
    pred_path.write_text("[]", encoding="utf-8")

    assert iter_per_tar_prediction_paths(str(per_tar_dir)) == [str(pred_path)]
