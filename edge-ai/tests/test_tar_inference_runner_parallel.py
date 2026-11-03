"""Tests for parallel TAR processing in tar_inference_runner_by_day.py."""
import importlib
import json
import os
import sys
import threading
import types
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the edge-ai directory and its sub-packages are importable without
# pulling in heavy optional dependencies (torch, pyvips, azure, PySide6).
# We stub out the problematic top-level imports before loading the module.
# ---------------------------------------------------------------------------

EDGE_AI_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EDGE_AI_DIR.parent
GUI_DIR = REPO_ROOT / "gui"
UTILITY_SCRIPTS_DIR = REPO_ROOT / "utilityscripts"

def _ensure_stub(name, attrs=None):
    """Register a lightweight stub module under *name* if not already present."""
    if name in sys.modules:
        return sys.modules[name]
    parts = name.split(".")
    parent_name = ".".join(parts[:-1])
    if parent_name:
        parent = _ensure_stub(parent_name)
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    if parent_name:
        setattr(sys.modules[parent_name], parts[-1], mod)
    return mod


def _stub_dependencies():
    """Stub every import that the runner needs but that isn't installed in CI."""
    # torch
    torch_mod = _ensure_stub("torch")
    torch_mod.no_grad = lambda: MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: None)
    torch_mod.device = lambda x: x
    torch_mod.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        get_device_name=lambda i: "",
    )
    torch_mod.__version__ = "0.0"
    torch_mod.version = types.SimpleNamespace(cuda="0.0")
    torch_mod.load = MagicMock(return_value={})
    torch_mod.stack = MagicMock()
    torch_mod.nn = types.SimpleNamespace(Module=object)

    # pyvips
    pyvips_mod = _ensure_stub("pyvips")
    pyvips_mod.Error = Exception

    # azure stubs
    _ensure_stub("azure")
    _ensure_stub("azure.core")
    _ensure_stub("azure.core.exceptions", {"ResourceNotFoundError": Exception})
    _ensure_stub("azure.identity", {"DefaultAzureCredential": MagicMock})
    _ensure_stub("azure.storage")
    _ensure_stub("azure.storage.blob", {
        "BlobServiceClient": MagicMock,
        "ContentSettings": MagicMock,
    })

    # PySide6 stubs (workers.py imports it; we don't load workers directly but
    # the runner imports GUI helper modules)
    _ensure_stub("PySide6")
    _ensure_stub("PySide6.QtCore", {"QThread": object, "Signal": lambda *a: None})

    # gps / prediction_records stubs
    _ensure_stub("gps")
    _ensure_stub("prediction_records", {
        "build_prediction_record": MagicMock(return_value={}),
        "extract_gps_payload": MagicMock(return_value=None),
    })

    # gui sub-package stubs
    gui_pkg = _ensure_stub("gui")
    gui_pkg.__path__ = [str(GUI_DIR)]

    _ensure_stub("gui.constants", {
        "TAR_PROGRESS_EVERY_IMAGES": 1000,
        "MAX_BLOBS_SAFETY": 100_000,
    })
    _ensure_stub("gui.csv_utils", {
        "parse_sample_ratio": lambda x: float(x),
        "DEFAULT_TAR_SAMPLE_RATIO": 1.0,
    })
    _ensure_stub("gui.model_utils", {
        "build_inference_transform": MagicMock(return_value=lambda b: b),
        "build_model_from_artifact": MagicMock(return_value=MagicMock()),
        "idx_to_label_fn": MagicMock(return_value=lambda i: str(i)),
    })
    _ensure_stub("gui.tar_streaming", {
        "iter_tar_image_bytes_from_blob": MagicMock(return_value=iter([])),
        "per_tar_paths": lambda run_dir, tar_name: (
            str(Path(run_dir) / "per_tar" / tar_name),
            str(Path(run_dir) / "per_tar" / tar_name / "predictions.json"),
            str(Path(run_dir) / "per_tar" / tar_name / "summary.json"),
        ),
        "safe_relative_stem_from_tar_name": lambda t: Path(t).stem,
    })
    _ensure_stub("gui.azure_utils", {
        "get_blob_service_client": MagicMock(),
        "utc_now_iso": lambda: "2000-01-01T00:00:00Z",
        "extract_runid_from_model_blob": MagicMock(return_value=None),
    })
    _ensure_stub("gui.blob_listing", {
        "quick_count_blobs": MagicMock(return_value=0),
    })
    _ensure_stub("gui.workers", {})


_stub_dependencies()

# Insert edge-ai on path so the module resolves its local imports
if str(EDGE_AI_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_AI_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(UTILITY_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_SCRIPTS_DIR))

import tar_inference_runner_by_day as runner  # noqa: E402


def test_settings_blob_argument_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "runner", "--model-blob", "model.pt", "--target-container", "test",
        "--settings-blob", "settings.json",
    ])
    with pytest.raises(SystemExit) as exc:
        runner.parse_args()
    assert exc.value.code == 2
    assert "unrecognized arguments: --settings-blob" in capsys.readouterr().err


@pytest.mark.parametrize("arguments,environment,expected", [
    ([], None, "inf"),
    (["--size-threshold-px", "64"], None, 64),
    ([], "128", 128),
    (["--size-threshold-px", "inf"], "128", "inf"),
])
def test_size_threshold_argument(monkeypatch, arguments, environment, expected):
    monkeypatch.delenv("SIZE_THRESHOLD_PX", raising=False)
    if environment is not None:
        monkeypatch.setenv("SIZE_THRESHOLD_PX", environment)
    monkeypatch.setattr(sys, "argv", [
        "runner", "--model-blob", "model.pt", "--target-container", "test", *arguments,
    ])
    assert runner.parse_args().size_threshold_px == expected


def test_invalid_size_threshold_argument_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "runner", "--model-blob", "model.pt", "--target-container", "test",
        "--size-threshold-px", "-1",
    ])
    with pytest.raises(SystemExit) as exc:
        runner.parse_args()
    assert exc.value.code == 2
    assert "--size-threshold-px" in capsys.readouterr().err


@pytest.mark.parametrize("threshold", [64, "inf", 200])
@pytest.mark.parametrize("sample_ratio", [1.0, 0.25])
def test_runner_filters_before_transform_and_logs_choice(
    monkeypatch, tmp_path, threshold, sample_ratio,
):
    dimensions = {
        b"narrow": (63, 100), b"short": (100, 63),
        b"exact": (64, 64), b"large": (100, 100),
    }
    images = [(key.decode() + ".tif", key) for key in dimensions]
    open_image = MagicMock(side_effect=lambda data, *a, **kw: types.SimpleNamespace(
        width=dimensions[data][0], height=dimensions[data][1],
    ))
    monkeypatch.setattr(runner.pyvips, "Image", types.SimpleNamespace(new_from_buffer=open_image), raising=False)
    monkeypatch.setattr(runner, "iter_tar_image_bytes_from_blob", lambda *a, **kw: iter(images))
    expected_images = [
        data for data, (width, height) in dimensions.items()
        if threshold == "inf" or (width >= threshold and height >= threshold)
    ]
    monkeypatch.setattr(runner.torch, "max", MagicMock(return_value=(
        [types.SimpleNamespace(item=lambda: 0.9) for _ in expected_images],
        [types.SimpleNamespace(item=lambda: 0) for _ in expected_images],
    )), raising=False)
    monkeypatch.setattr(runner.torch, "softmax", MagicMock(), raising=False)
    monkeypatch.setattr(runner, "build_prediction_record", lambda **kwargs: {"tar_member": kwargs["member_name"]})
    transform = MagicMock()
    service = MagicMock()
    service.get_container_client.return_value.get_blob_client.return_value.get_blob_properties.return_value.size = 1024
    output_dir = tmp_path / "per_tar" / "sample.tar"
    output_dir.mkdir(parents=True)

    result = runner.infer_on_tar_archive(
        container="source", tar_name="sample.tar", prefix="prefix/", sample_ratio=sample_ratio,
        size_threshold_px=threshold, model=MagicMock(), tfm=transform,
        idx_to_label=lambda index: "copepod", device="cpu", model_blob="model.pt",
        model_runid="test-model", run_dir=tmp_path, source_blob_service_client=service,
        output_blob_service_client=None, output_container="", output_prefix="runs",
        output_slug="test-run", log_cb=lambda message: None,
    )

    assert result["error_message"] is None
    assert [call.args[0] for call in transform.call_args_list] == expected_images
    assert open_image.call_count == (0 if threshold == "inf" else 4)
    summary = json.loads((output_dir / "summary.json").read_text())
    predictions = json.loads((output_dir / "predictions.json").read_text())
    assert summary["size_threshold_px"] == threshold
    assert summary["size_filtered_image_count"] == 4 - len(expected_images)
    assert summary["image_count"] == len(expected_images) == len(predictions)
    assert summary["class_counts"] == ({"copepod": len(expected_images)} if expected_images else {})
    assert [row["tar_member"] for row in predictions] == [data.decode() + ".tif" for data in expected_images]
    for prediction in predictions:
        assert prediction["size_threshold_px"] == threshold
        assert prediction["model_blob"] == "model.pt"
        assert prediction["tar_sample_ratio"] == sample_ratio


# ---------------------------------------------------------------------------
# Tests for _default_parallel_stream_count
# ---------------------------------------------------------------------------

class TestDefaultParallelStreamCount:
    def test_mobilenet_v3_small_returns_16(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenet_v3_small"}) == 16

    def test_mobilenet_alias_returns_16(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenet"}) == 16

    def test_mobilenetv3_alias_returns_16(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenetv3"}) == 16

    def test_resnet_returns_2(self):
        assert runner._default_parallel_stream_count({"arch": "resnet18"}) == 2

    def test_mobilenet_v2_returns_2(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenet_v2"}) == 2

    def test_mobilenetv2_alias_returns_2(self):
        assert runner._default_parallel_stream_count({"arch": "mobilenetv2"}) == 2

    def test_missing_arch_defaults_to_resnet18(self):
        assert runner._default_parallel_stream_count({}) == 2

    def test_none_artifact_defaults_to_resnet18(self):
        assert runner._default_parallel_stream_count(None) == 2

    def test_hyphen_normalised(self):
        # arch names with hyphens should be normalised to underscores
        assert runner._default_parallel_stream_count({"arch": "mobilenet-v3-small"}) == 16


# ---------------------------------------------------------------------------
# Tests for parallel TAR dispatch in main()
# ---------------------------------------------------------------------------

def _make_fake_infer_on_tar(captured_calls, call_lock):
    """Return a fake infer_on_tar_archive that records calls and returns a valid result dict."""
    def _fake(*, container, tar_name, prefix, sample_ratio, model, tfm,
              idx_to_label, device, model_blob, model_runid, run_dir,
              source_blob_service_client, output_blob_service_client,
              output_container, output_prefix, output_slug, log_cb, size_threshold_px="inf"):
        with call_lock:
            captured_calls.append(tar_name)
        return {
            "container": container,
            "tar_name": tar_name,
            "prefix": prefix,
            "sample_ratio": sample_ratio,
            "tar_class_counts": Counter({"plankton": 1}),
            "error_message": None,
            "skipped_existing": False,
        }
    return _fake


class TestParallelDispatch:
    """Verify that the ThreadPoolExecutor path is exercised with the right concurrency."""

    def _run_main_with_tars(self, tar_names, parallel_streams_arg=0, tmp_path=None,
                            model_blob="model2000-01-01T00-00-00Z.pt", day="2000-01-01",
                            expected_runid="2000-01-01T00-00-00Z",
                            threshold="inf", resume_summary=None):
        """Patch main()'s internals and invoke it; return (selected_tars, captured_calls)."""
        captured_calls = []
        call_lock = threading.Lock()

        fake_artifact = {"arch": "resnet18", "model_state_dict": {"_dummy": None}}

        fake_bsc = MagicMock()
        fake_container_client = MagicMock()
        fake_container_client.list_blobs.return_value = [
            types.SimpleNamespace(name=n) for n in tar_names
        ]
        # model bytes download must return bytes (BytesIO wraps it)
        fake_downloader = MagicMock()
        fake_downloader.readall.return_value = b"fake-model-bytes"
        fake_blob_client = MagicMock()
        fake_blob_client.download_blob.return_value = fake_downloader
        fake_bsc.get_container_client.return_value = fake_container_client
        fake_container_client.get_blob_client.return_value = fake_blob_client
        if resume_summary is not None:
            summary_blob_client = MagicMock()
            summary_blob_client.download_blob.return_value.readall.return_value = json.dumps(resume_summary).encode()
            fake_container_client.get_blob_client.side_effect = (
                lambda name: summary_blob_client if name.endswith("summary.json") else fake_blob_client
            )

        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model
        fake_model.to.return_value = fake_model
        fake_model.load_state_dict = MagicMock()

        # Build a minimal set of args
        args_ns = types.SimpleNamespace(
            account_url="https://fake.blob.core.windows.net",
            models_container="trainedmodels",
            model_blob=model_blob,
            target_container="test-container",
            target_prefix="",
            day=day,
            run_key="test-run",
            sample_ratio="1.0",
            size_threshold_px=threshold,
            output_container="results" if resume_summary is not None else "",
            output_prefix="runs",
            resume_existing=str(resume_summary is not None),
            allow_empty_day="true",
            no_resume=False,
            parallel_streams=parallel_streams_arg,
        )

        work_dir = tmp_path or Path("/tmp/runner_test")
        work_dir.mkdir(parents=True, exist_ok=True)

        infer_mock = MagicMock(side_effect=_make_fake_infer_on_tar(captured_calls, call_lock))
        patches = [
            patch.object(runner, "parse_args", return_value=args_ns),
            patch.object(runner, "build_blob_service_client", return_value=fake_bsc),
            patch.object(runner, "build_model_from_artifact", return_value=fake_model),
            patch.object(runner, "build_inference_transform", return_value=lambda b: b),
            patch.object(runner, "idx_to_label_fn", return_value=lambda i: str(i)),
            patch.object(runner, "infer_on_tar_archive", infer_mock),
            patch("torch.load", return_value=fake_artifact),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.device_count", return_value=0),
            patch.object(Path, "cwd", return_value=work_dir),
        ]

        import contextlib
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            runner.main()

        if resume_summary is None:
            fake_container_client.get_blob_client.assert_called_once_with(model_blob)
        for call in infer_mock.call_args_list:
            assert call.kwargs["model_blob"] == model_blob
            assert call.kwargs["model_runid"] == expected_runid
            assert call.kwargs["size_threshold_px"] == threshold
        assert list((work_dir / "workflow_outputs" / "test-run").iterdir()) == [
            work_dir / "workflow_outputs" / "test-run" / "run_summary.json",
        ]
        return captured_calls

    @pytest.mark.parametrize("day", ["2000-01-01", ""])
    @pytest.mark.parametrize("model_blob, expected_runid", [
        ("model2000-01-01T00-00-00Z.pt", "2000-01-01T00-00-00Z"),
        ("nested/model2000-01-01T00-00-00Z.pt", "2000-01-01T00-00-00Z"),
        ("custom.pt", None),
    ])
    def test_model_metadata_and_summary_name(self, tmp_path, day, model_blob, expected_runid):
        with patch.object(runner, "build_output_blob_name", wraps=runner.build_output_blob_name) as build_name:
            self._run_main_with_tars(
                ["2000-01-01/only.tar"], tmp_path=tmp_path, day=day,
                model_blob=model_blob, expected_runid=expected_runid,
            )
        summary = json.loads(
            (tmp_path / "workflow_outputs" / "test-run" / "run_summary.json").read_text()
        )
        assert summary["model_blob"] == model_blob
        assert summary["model_runid"] == expected_runid
        build_name.assert_called_once_with(
            "runs", "test-run", f"run_summary__{runner.sanitize_slug(day or model_blob)}.json",
        )

    def test_all_tars_processed(self, tmp_path):
        tar_names = [f"2000-01-01/archive_{i:03d}.tar" for i in range(5)]
        calls = self._run_main_with_tars(tar_names, tmp_path=tmp_path)
        assert sorted(calls) == sorted(tar_names)

    def test_parallel_streams_override(self, tmp_path):
        """When --parallel-streams is set explicitly, that value should be used."""
        tar_names = [f"2000-01-01/archive_{i:03d}.tar" for i in range(4)]
        calls = self._run_main_with_tars(tar_names, parallel_streams_arg=4, tmp_path=tmp_path)
        assert len(calls) == 4

    def test_single_tar(self, tmp_path):
        tar_names = ["2000-01-01/only.tar"]
        calls = self._run_main_with_tars(tar_names, tmp_path=tmp_path)
        assert calls == tar_names

    def test_no_tars_writes_empty_day_summary(self, tmp_path):
        """When no TARs are found and allow_empty_day is true, main() should not raise."""
        calls = self._run_main_with_tars([], tmp_path=tmp_path)
        assert calls == []

    @pytest.mark.parametrize("threshold", ["inf", 64])
    @pytest.mark.parametrize("tar_names", [[], ["2000-01-01/only.tar"]])
    def test_run_summary_records_size_threshold(self, tmp_path, threshold, tar_names):
        self._run_main_with_tars(tar_names, tmp_path=tmp_path, threshold=threshold)
        summary = json.loads(
            (tmp_path / "workflow_outputs" / "test-run" / "run_summary.json").read_text()
        )
        assert summary["size_threshold_px"] == threshold
        for selected in summary["selected_tars"]:
            assert selected["size_threshold_px"] == threshold

    @pytest.mark.parametrize("threshold,previous,skipped", [
        ("inf", {}, True),
        (64, {}, False),
        ("inf", {"size_threshold_px": 64}, False),
        (64, {"size_threshold_px": "64.0"}, True),
    ])
    def test_resume_requires_matching_size_threshold(self, tmp_path, threshold, previous, skipped):
        calls = self._run_main_with_tars(
            ["2000-01-01/only.tar"], tmp_path=tmp_path,
            threshold=threshold, resume_summary=previous,
        )
        assert calls == ([] if skipped else ["2000-01-01/only.tar"])
