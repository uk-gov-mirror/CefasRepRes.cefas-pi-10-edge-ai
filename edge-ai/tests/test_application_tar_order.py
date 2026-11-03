import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

EDGE_AI_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EDGE_AI_DIR.parent
GUI_DIR = REPO_ROOT / "gui"
sys.path.insert(0, str(EDGE_AI_DIR))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def worker_module():
    torch = MagicMock()
    torch.load.return_value = {"model_state_dict": {"weight": 1}}
    torch.cuda.is_available.return_value = False
    thread = type("Thread", (), {
        "__init__": lambda self, parent=None: None,
        "isInterruptionRequested": lambda self: False,
    })
    stubs = {
        "pyvips": SimpleNamespace(Error=RuntimeError),
        "torch": torch,
        "torch.nn": torch.nn,
        "torch.nn.functional": torch.nn.functional,
        "PySide6.QtCore": SimpleNamespace(QThread=thread, Signal=lambda *args: MagicMock()),
        "gui.azure_utils": SimpleNamespace(
            get_blob_service_client=MagicMock(),
            utc_now_iso=lambda: "2026-10-27T11:00:00Z",
            extract_runid_from_model_blob=lambda blob: None,
        ),
        "gui.model_utils": SimpleNamespace(
            build_model_from_artifact=MagicMock(),
            build_inference_transform=MagicMock(),
            idx_to_label_fn=MagicMock(),
        ),
    }
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "gui.workers_under_test",
            GUI_DIR / "workers.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module


def _make_worker(module, tmp_path, **kwargs):
    sizes = {("a", "large.tar"): 300, ("b", "small.tar"): 10, ("a", "medium.tar"): 100}
    blobs = {key: MagicMock() for key in sizes}
    for key, blob in blobs.items():
        blob.get_blob_properties.return_value.size = sizes[key]
    containers = {name: MagicMock() for name in ("a", "b", "trainedmodels")}
    for name in ("a", "b"):
        containers[name].list_blobs.return_value.by_page.return_value = []
        containers[name].get_blob_client.side_effect = lambda blob, name=name: blobs[(name, blob)]
    containers["trainedmodels"].get_blob_client.return_value.download_blob.return_value.readall.return_value = b""
    service = module.get_blob_service_client.return_value
    service.get_container_client.side_effect = containers.__getitem__
    worker = module.ApplicationWorker(
        account_url="https://example.blob.core.windows.net",
        selected_tars=[(container, name, "prefix/", 0.0) for container, name in sizes],
        targets=[{"container": name} for name in ("a", "b")],
        model_blob="model.pt",
        output_root=str(tmp_path),
        **kwargs,
    )
    return worker, blobs


@pytest.mark.parametrize("sort_options,expected", [
    ({}, ["large.tar", "small.tar", "medium.tar"]),
    ({"sort_by_size": False}, ["large.tar", "small.tar", "medium.tar"]),
    ({"sort_by_size": True}, ["small.tar", "medium.tar", "large.tar"]),
])
def test_tar_priority_order(worker_module, tmp_path, sort_options, expected):
    worker, blobs = _make_worker(worker_module, tmp_path, **sort_options)

    worker.run()

    worker.failed.emit.assert_not_called()
    result = worker.finished_ok.emit.call_args.args[0]
    summary = json.loads(Path(result["run_summary"]).read_text())
    assert [row["source_tar_blob"] for row in summary["selected_tars"]] == expected
    messages = [call.args[0] for call in worker.log.emit.call_args_list]
    for index, name in enumerate(expected, start=1):
        assert any(f"[{index}/3] Processing TAR " in message and name in message for message in messages)
    for blob in blobs.values():
        assert blob.get_blob_properties.call_count == int(sort_options.get("sort_by_size", False))
    assert [row[1] for row in worker.selected_tars] == ["large.tar", "small.tar", "medium.tar"]


def test_size_lookup_failure_is_reported(worker_module, tmp_path):
    worker, blobs = _make_worker(worker_module, tmp_path, sort_by_size=True)
    blobs[("a", "large.tar")].get_blob_properties.side_effect = RuntimeError("size lookup failed")

    worker.run()

    worker.failed.emit.assert_called_once_with("size lookup failed")
    worker.finished_ok.emit.assert_not_called()


def test_model_class_mapping_supports_both_artifact_formats(worker_module):
    assert worker_module._class_labels_from_artifact({
        "idx_to_class": {"1": "other", "0": "copepod"},
    }) == ["copepod", "other"]
    assert worker_module._class_labels_from_artifact({
        "class_to_idx": {"other": 1, "copepod": 0},
    }) == ["copepod", "other"]


def test_model_classes_worker_loads_selected_model_artifact(worker_module):
    models = MagicMock()
    worker_module.get_blob_service_client.return_value.get_container_client.return_value = models
    models.get_blob_client.return_value.download_blob.return_value.readall.return_value = b"model"
    worker_module.torch.load.return_value = {
        "idx_to_class": {1: "other", 0: "copepod"},
    }
    worker = worker_module.ModelClassesWorker("https://example.blob.core.windows.net", "model.pt")

    worker.run()

    worker.classes_ready.emit.assert_called_once_with(["copepod", "other"])
    worker.failed.emit.assert_not_called()


def test_selected_classes_are_saved_locally_after_classification(
    worker_module, tmp_path, monkeypatch,
):
    worker, _ = _make_worker(
        worker_module,
        tmp_path,
        local_save_classes=["copepod"],
    )
    worker.selected_tars = [("a", "large.tar", "prefix/", 1.0, "inf")]
    monkeypatch.setattr(
        worker_module,
        "iter_tar_image_bytes_from_blob",
        lambda *args, **kwargs: iter([
            ("RawImages/copepod.tif", b"copepod-image"),
            ("RawImages/other.tif", b"other-image"),
        ]),
    )
    worker_module.idx_to_label_fn.return_value = lambda index: ["copepod", "other"][index]
    worker_module.F.softmax.return_value = "probabilities"
    worker_module.torch.max.return_value = (
        [SimpleNamespace(item=lambda: 0.9), SimpleNamespace(item=lambda: 0.8)],
        [SimpleNamespace(item=lambda: 0), SimpleNamespace(item=lambda: 1)],
    )

    worker.run()

    worker.failed.emit.assert_not_called()
    result = worker.finished_ok.emit.call_args.args[0]
    retained_dir = Path(result["run_dir"]) / "retained_classes"
    assert (retained_dir / "copepod" / "large" / "RawImages" / "copepod.tif").read_bytes() == b"copepod-image"
    assert not (retained_dir / "other").exists()
    run_summary = json.loads(Path(result["run_summary"]).read_text())
    assert run_summary["local_retention_classes"] == ["copepod"]
    assert run_summary["locally_retained_image_count"] == 1
    per_tar_summary = json.loads(
        (Path(result["predictions_dir"]) / "large" / "summary.json").read_text()
    )
    assert per_tar_summary["locally_retained_image_count"] == 1


def test_resume_reuses_output_folder_and_temperature(worker_module, tmp_path, monkeypatch):
    output_slug = "guilocalinference-2026-10-27T16-32-25Z-ready-parrot"
    resume_settings = {
        "output_container": "ml-prediction-results",
        "output_prefix": "runs",
        "output_slug": output_slug,
        "model_runid": "original-model",
    }
    worker, _ = _make_worker(
        worker_module,
        tmp_path,
        resume_settings=resume_settings,
        temperature_override=2.5,
    )
    worker.selected_tars = [("a", "large.tar", "prefix/", 0.0)]
    upload = MagicMock()
    monkeypatch.setattr(worker_module, "upload_json_blob", upload)

    worker.run()

    worker.failed.emit.assert_not_called()
    result = worker.finished_ok.emit.call_args.args[0]
    assert Path(result["run_dir"]).name == "application_2026-10-27T16-32-25Z-ready-parrot_modeloriginal-model"
    summary = json.loads((Path(result["predictions_dir"]) / "large" / "summary.json").read_text())
    assert summary["temperature"] == 2.5
    run_summary = json.loads(Path(result["run_summary"]).read_text())
    assert run_summary["sort_by_size"] is False
    uploaded_names = [call.args[2] for call in upload.call_args_list]
    assert f"runs/{output_slug}/per_tar/large/predictions.json" in uploaded_names
    assert f"runs/{output_slug}/per_tar/large/summary.json" in uploaded_names
    assert f"runs/{output_slug}/run_summary.json" in uploaded_names


def test_move_by_class_worker_streams_only_selected_class(worker_module, tmp_path, monkeypatch):
    service = worker_module.get_blob_service_client.return_value
    source_container = service.get_container_client.return_value
    source_container.get_blob_client.side_effect = lambda name: name
    tar_members = {
        "first.tar": [
            ("RawImages/copepod.tif", b"copepod"),
            ("RawImages/other.tif", b"other"),
        ],
        "second.tar": [("RawImages/copepod-2.tif", b"copepod-2")],
    }

    def stream_members(_blob_client, tar_name, **kwargs):
        for member_name, payload in tar_members[tar_name]:
            yield SimpleNamespace(name=member_name), BytesIO(payload)

    stream = MagicMock(side_effect=stream_members)
    monkeypatch.setattr(worker_module, "iter_tar_members_from_blob", stream)
    worker = worker_module.MoveByClassWorker(
        account_url="https://example.blob.core.windows.net",
        predictions=[
            {
                "container": "source",
                "source_tar_blob": "first.tar",
                "tar_member": "RawImages/copepod.tif",
                "predicted_label": "copepods",
            },
            {
                "container": "source",
                "source_tar_blob": "first.tar",
                "tar_member": "RawImages/other.tif",
                "predicted_label": "other",
            },
            {
                "container": "source",
                "source_tar_blob": "second.tar",
                "tar_member": "RawImages/copepod-2.tif",
                "predicted_label": "copepods",
            },
            {
                "container": "source",
                "source_tar_blob": "third.tar",
                "tar_member": "RawImages/other-2.tif",
                "predicted_label": "other",
            },
        ],
        output_run_dir=str(tmp_path),
        class_filter="copepods",
    )

    worker.run()

    worker.failed.emit.assert_not_called()
    stats = worker.finished_ok.emit.call_args.args[0]
    assert stats["downloaded"] == 2
    assert stats["total"] == 2
    assert [call.args[1] for call in stream.call_args_list] == ["first.tar", "second.tar"]
    assert (tmp_path / "by_class" / "copepods" / "first__copepod.tif").read_bytes() == b"copepod"
    assert (tmp_path / "by_class" / "copepods" / "second__copepod-2.tif").read_bytes() == b"copepod-2"
    assert not (tmp_path / "by_class" / "other").exists()


@pytest.mark.parametrize("sample_ratio", [1.0, 0.5])
@pytest.mark.parametrize("threshold", [64, "inf", 200])
def test_image_size_filter_precedes_transform_and_is_logged(
    worker_module, tmp_path, monkeypatch, sample_ratio, threshold,
):
    worker, _ = _make_worker(worker_module, tmp_path)
    worker.selected_tars = [("a", "large.tar", "prefix/", sample_ratio, threshold)]
    dimensions = {
        b"narrow": (63, 100), b"short": (100, 63),
        b"exact": (64, 64), b"large": (100, 100),
    }
    images = [(key.decode() + ".tif", key) for key in dimensions]
    open_image = MagicMock(side_effect=lambda data, *a, **kw: SimpleNamespace(
        width=dimensions[data][0], height=dimensions[data][1],
    ))
    monkeypatch.setattr(worker_module.pyvips, "Image", SimpleNamespace(new_from_buffer=open_image), raising=False)
    if sample_ratio < 1:
        monkeypatch.setattr(worker_module, "build_tar_stream_segments", lambda *a, **kw: [(0, 128), (512, 128)])
        image_stream = MagicMock(side_effect=[iter(images[:2]), iter(images[2:])])
    else:
        image_stream = MagicMock(return_value=iter(images))
    monkeypatch.setattr(worker_module, "iter_tar_image_bytes_from_blob", image_stream)
    expected_images = [
        data for data, (width, height) in dimensions.items()
        if threshold == "inf" or (width >= threshold and height >= threshold)
    ]
    worker_module.torch.max.return_value = (
        [SimpleNamespace(item=lambda: 0.9) for _ in expected_images],
        [SimpleNamespace(item=lambda: 0) for _ in expected_images],
    )
    worker_module.idx_to_label_fn.return_value = lambda index: "copepod"

    worker.run()

    worker.failed.emit.assert_not_called()
    transform = worker_module.build_inference_transform.return_value
    assert [call.args[0] for call in transform.call_args_list] == expected_images
    assert open_image.call_count == (0 if threshold == "inf" else 4)
    result = worker.finished_ok.emit.call_args.args[0]
    per_tar = Path(result["predictions_dir"]) / "large"
    predictions = json.loads((per_tar / "predictions.json").read_text())
    summary = json.loads((per_tar / "summary.json").read_text())
    assert len(predictions) == len(expected_images)
    assert summary["size_threshold_px"] == threshold
    assert summary["size_filtered_image_count"] == 4 - len(expected_images)
    assert summary["class_counts"] == ({"copepod": len(expected_images)} if expected_images else {})
    for prediction in predictions:
        assert prediction["size_threshold_px"] == threshold
        assert prediction["model_blob"] == "model.pt"
        assert prediction["tar_sample_ratio"] == sample_ratio
    run_summary = json.loads(Path(result["run_summary"]).read_text())
    assert run_summary["selected_tars"][0]["size_threshold_px"] == threshold
