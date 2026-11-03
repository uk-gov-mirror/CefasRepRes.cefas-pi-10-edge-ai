import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = REPO_ROOT / "gui"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GUI_DIR))

import gps_tar_images as images_module
from gps_tar_images import (
    TarImagesDialog,
    TarImagesWorker,
    load_tar_prediction_images,
    predictions_blob_for_summary,
)
from test_data_viewer_mosaics import FakeBlob, image_bytes, tar_bytes


SUMMARY = "runs/guilocalinference-2026-09-04T11-34-47Z/per_tar/2023-09-29/0830/summary.json"
PREDICTIONS = SUMMARY.replace("summary.json", "predictions.json")


def prediction(member, label="copepod", tar="2023-09-29/0830.tar"):
    return {"predicted_label": label, "blob_member_uri": f"raw/{tar}!/{member}"}


def service_for(payload, members=(), mode="w"):
    archive = FakeBlob(tar_bytes(members, mode=mode))
    requested = []

    def get_blob(container, blob):
        requested.append((container, blob))
        if container == "ml-prediction-results":
            assert blob == PREDICTIONS
            return SimpleNamespace(
                download_blob=lambda: SimpleNamespace(
                    readall=lambda: json.dumps(payload).encode("utf-8"),
                ),
            )
        assert container == "raw"
        assert blob == "2023-09-29/0830.tar"
        return archive

    service = SimpleNamespace(
        get_container_client=lambda container: SimpleNamespace(
            get_blob_client=lambda blob: get_blob(container, blob),
        ),
    )
    return service, archive, requested


def load(service, output, **kwargs):
    defaults = {
        "image_cb": lambda *args: None,
        "progress_cb": lambda *args: None,
        "is_cancelled": lambda: False,
    }
    defaults.update(kwargs)
    return load_tar_prediction_images(
        service, "raw", "2023-09-29/0830.tar", SUMMARY, "copepod", output, **defaults,
    )


def test_paired_predictions_blob_preserves_run_and_tar_directory():
    assert predictions_blob_for_summary(SUMMARY) == PREDICTIONS
    assert predictions_blob_for_summary("summary.json") == "predictions.json"
    with pytest.raises(ValueError, match="summary.json"):
        predictions_blob_for_summary("runs/another.json")


@pytest.mark.parametrize("wrapper", [lambda x: x, lambda x: {"predictions": x}, lambda x: {"items": x}])
@pytest.mark.parametrize("mode", ["w", "w:gz"])
def test_loads_all_matching_tiffs_once_without_decoding_other_labels_or_tail(
    tmp_path, wrapper, mode,
):
    red = image_bytes("red", size=(400, 200))
    blue = image_bytes("blue")
    records = [
        prediction("images/red.tif"), prediction("images/blue.TIFF"),
        prediction("images/red.tif"), prediction("fish.tif", "fish"),
        prediction("preview.png"),
    ]
    service, archive, requested = service_for(wrapper(records), [
        ("fish.tif", b"unselected image must not be decoded"),
        ("preview.png", b"not a TIFF"),
        ("images/red.tif", red), ("images/blue.TIFF", blue),
        *[(f"tail-{i}.tif", b"tail must not be decoded") for i in range(100)],
    ], mode)
    loaded = []
    progress = []

    assert load(
        service, tmp_path, image_cb=lambda *args: loaded.append(args),
        progress_cb=lambda *args: progress.append(args),
    ) == (2, False)

    assert requested == [
        ("ml-prediction-results", PREDICTIONS), ("raw", "2023-09-29/0830.tar"),
    ]
    assert [member for _, _, member in loaded] == ["images/red.tif", "images/blue.TIFF"]
    assert Path(loaded[0][0]).read_bytes() == red
    assert Path(loaded[1][0]).read_bytes() == blue
    with Image.open(loaded[0][1]) as thumbnail:
        assert thumbnail.size == (160, 80)
    assert progress == [(0, 2), (1, 2), (2, 2)]
    assert archive.calls == [{"max_concurrency": 1}]
    assert archive.downloader.closed
    if mode == "w":
        assert archive.downloader.bytes_yielded < len(archive.downloader.data)


def test_no_matching_predictions_does_not_download_tar(tmp_path):
    service, archive, requested = service_for([prediction("fish.tif", "fish")])
    assert load(service, tmp_path) == (0, False)
    assert requested == [("ml-prediction-results", PREDICTIONS)]
    assert not archive.calls


@pytest.mark.parametrize("payload", [
    {}, {"predictions": "not a list"}, [None],
    [{"predicted_label": "copepod"}],
    [prediction("one.tif", tar="another.tar")],
])
def test_invalid_predictions_fail_explicitly_without_loading_tar(tmp_path, payload):
    service, archive, _ = service_for(payload)
    with pytest.raises(ValueError):
        load(service, tmp_path)
    assert not archive.calls


def test_missing_predicted_member_is_reported_and_stream_closed(tmp_path):
    service, archive, _ = service_for([prediction("missing.tif")])
    with pytest.raises(FileNotFoundError, match="1 predicted TIFF.*missing.tif"):
        load(service, tmp_path)
    assert archive.downloader.closed


def test_selected_corrupt_image_fails_and_stream_closed(tmp_path):
    service, archive, _ = service_for([prediction("bad.tif")], [("bad.tif", b"corrupt")])
    with pytest.raises(ValueError, match="Could not decode TIFF bad.tif"):
        load(service, tmp_path)
    assert archive.downloader.closed


@pytest.mark.parametrize("stage", ["predictions", "tar"])
def test_download_failures_are_not_treated_as_empty_results(tmp_path, stage):
    service, archive, _ = service_for([prediction("one.tif")])

    def fail(**kwargs):
        raise OSError("download unavailable")

    if stage == "tar":
        archive.download_blob = fail
    else:
        service.get_container_client = lambda name: SimpleNamespace(
            get_blob_client=lambda blob: SimpleNamespace(download_blob=fail),
        )
    with pytest.raises(OSError, match="download unavailable"):
        load(service, tmp_path)


def test_malformed_predictions_json_is_reported(tmp_path):
    service = SimpleNamespace(
        get_container_client=lambda name: SimpleNamespace(
            get_blob_client=lambda blob: SimpleNamespace(
                download_blob=lambda: SimpleNamespace(readall=lambda: b"not-json"),
            ),
        ),
    )
    with pytest.raises(json.JSONDecodeError):
        load(service, tmp_path)


def test_predicted_member_names_cannot_escape_temporary_directory(tmp_path):
    service, _, _ = service_for(
        [prediction("../../outside.tif")],
        [("../../outside.tif", image_bytes("red"))],
    )
    loaded = []
    assert load(service, tmp_path, image_cb=lambda *args: loaded.append(args)) == (1, False)
    assert Path(loaded[0][0]).parent == tmp_path
    assert Path(loaded[0][0]).name == "000000.tif"
    assert loaded[0][2] == "../../outside.tif"


def test_cancellation_retains_loaded_images_and_closes_stream(tmp_path):
    service, archive, _ = service_for(
        [prediction("one.tif"), prediction("two.tif")],
        [("one.tif", image_bytes("red")), ("two.tif", image_bytes("blue"))],
    )
    loaded = []
    assert load(
        service, tmp_path, image_cb=lambda *args: loaded.append(args),
        is_cancelled=lambda: bool(loaded),
    ) == (1, True)
    assert len(loaded) == 1
    assert archive.downloader.closed


def test_cancellation_before_loading_makes_no_requests(tmp_path):
    service, archive, requested = service_for([])
    assert load(service, tmp_path, is_cancelled=lambda: True) == (0, True)
    assert requested == []
    assert not archive.calls


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def finish_worker(app, worker):
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(5000)
    if worker.isRunning():
        loop.exec()
    timer.stop()
    assert worker.wait(1000)
    app.processEvents()


def test_gallery_loads_thumbnails_opens_full_size_and_removes_temporary_files(app, monkeypatch):
    service, _, _ = service_for(
        [prediction("one.tif")], [("one.tif", image_bytes("red", size=(300, 200)))],
    )
    monkeypatch.setattr(images_module, "get_blob_service_client", lambda *a, **k: service)
    dialog = TarImagesDialog("https://scanned", "raw", "2023-09-29/0830.tar", SUMMARY, "copepod")
    directory = Path(dialog._directory.name)
    dialog.show()
    finish_worker(app, dialog.worker)
    assert dialog.images.count() == 1
    item = dialog.images.item(0)
    assert item.text() == "one.tif"
    assert not item.icon().isNull()
    assert Path(item.data(Qt.UserRole)).is_file()
    assert dialog.status.text() == "Loaded all 1 matching TIFF(s)."
    assert dialog.stop_button.text() == "Close"
    dialog._show_image(item)
    viewer = dialog.findChild(QDialog)
    assert viewer.isVisible()
    assert viewer.findChild(QLabel).pixmap().size().toTuple() == (300, 200)
    dialog.close()
    assert not directory.exists()
    app.sendPostedEvents(None, QEvent.DeferredDelete)


def test_gps_get_images_filters_summary_then_label_and_opens_gallery(app, monkeypatch):
    from gps_plot_window import GPSPlotWindow

    service, _, requested = service_for(
        [prediction("one.tif"), prediction("fish.tif", "fish")],
        [("fish.tif", b"do not decode"), ("one.tif", image_bytes("red"))],
    )
    accounts = []

    def get_service(account_url, **kwargs):
        accounts.append(account_url)
        return service

    monkeypatch.setattr(images_module, "get_blob_service_client", get_service)
    location = {
        "container": "raw", "tar": "2023-09-29/0830.tar",
        "latitude": 51.5, "longitude": 0.16,
    }
    record = {
        "location": location, "summary_blob": SUMMARY,
        "summary": {"class_counts": {"copepod": 1, "fish": 1}},
    }
    window = GPSPlotWindow(
        [location], {
            "model-one": [record],
            "model-two": [{**record, "summary_blob": "runs/other/summary.json"}],
        }, 2, account_url="https://scanned-account",
    )
    window.tar_selection.setCurrentIndex(1)
    assert not window.get_images_button.isEnabled()
    window.model_filter.setCurrentIndex(window.model_filter.findData("model-one"))
    window.apply_filter()
    window.predicted_label_filter.setCurrentText("copepod")
    assert window.get_images_button.isEnabled()
    window.get_images_button.click()
    dialog = window._image_dialogs[0]
    directory = Path(dialog._directory.name)
    finish_worker(app, dialog.worker)
    assert dialog.isVisible()
    assert dialog.images.count() == 1
    assert dialog.images.item(0).text() == "one.tif"
    assert accounts == ["https://scanned-account"]
    assert requested[0] == ("ml-prediction-results", PREDICTIONS)
    window.close()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not directory.exists()
    assert window._image_dialogs == []


def test_worker_reports_download_errors_instead_of_success(app, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("download unavailable")

    monkeypatch.setattr(images_module, "get_blob_service_client", fail)
    worker = TarImagesWorker(
        "https://scanned", "raw", "2023-09-29/0830.tar", SUMMARY, "copepod", tmp_path,
    )
    errors = []
    loaded = []
    worker.failed.connect(errors.append)
    worker.loaded.connect(lambda *args: loaded.append(args))
    worker.run()
    assert errors == ["OSError: download unavailable"]
    assert loaded == []


def test_gallery_displays_failures_and_no_matches(app, monkeypatch):
    service, _, _ = service_for([])
    monkeypatch.setattr(images_module, "get_blob_service_client", lambda *a, **k: service)
    dialog = TarImagesDialog("https://scanned", "raw", "2023-09-29/0830.tar", SUMMARY, "copepod")
    finish_worker(app, dialog.worker)
    assert dialog.images.count() == 0
    assert "No TIFF predictions match" in dialog.status.text()
    dialog.close()

    def fail(*args, **kwargs):
        raise OSError("predictions unavailable")

    monkeypatch.setattr(images_module, "get_blob_service_client", fail)
    dialog = TarImagesDialog("https://scanned", "raw", "2023-09-29/0830.tar", SUMMARY, "copepod")
    finish_worker(app, dialog.worker)
    assert "Image loading failed: OSError: predictions unavailable" == dialog.status.text()
    assert dialog.progress.maximum() == 1
    assert dialog.stop_button.text() == "Close"
    dialog.close()
    app.sendPostedEvents(None, QEvent.DeferredDelete)


def test_close_and_escape_cancel_worker_before_cleaning_temporary_files(app, monkeypatch):
    import threading

    started = threading.Event()
    release = threading.Event()

    def blocked_load(service, *args, is_cancelled, **kwargs):
        started.set()
        assert release.wait(5)
        assert is_cancelled()
        return 0, True

    monkeypatch.setattr(images_module, "get_blob_service_client", lambda *a, **k: object())
    monkeypatch.setattr(images_module, "load_tar_prediction_images", blocked_load)
    dialog = TarImagesDialog("https://scanned", "raw", "2023-09-29/0830.tar", SUMMARY, "copepod")
    directory = Path(dialog._directory.name)
    dialog.show()
    try:
        assert started.wait(5)
        dialog.reject()
        assert dialog.worker.isInterruptionRequested()
        assert directory.exists()
        assert dialog.isVisible()
    finally:
        release.set()
        finish_worker(app, dialog.worker)
    assert not directory.exists()
    app.sendPostedEvents(None, QEvent.DeferredDelete)


def test_application_shutdown_waits_for_interrupted_worker(app, monkeypatch):
    import threading
    import time

    started = threading.Event()

    def wait_for_cancellation(service, *args, is_cancelled, **kwargs):
        started.set()
        deadline = time.monotonic() + 5
        while not is_cancelled():
            assert time.monotonic() < deadline
            time.sleep(0.001)
        return 0, True

    monkeypatch.setattr(images_module, "get_blob_service_client", lambda *a, **k: object())
    monkeypatch.setattr(images_module, "load_tar_prediction_images", wait_for_cancellation)
    dialog = TarImagesDialog("https://scanned", "raw", "2023-09-29/0830.tar", SUMMARY, "copepod")
    directory = Path(dialog._directory.name)
    assert started.wait(5)
    dialog._shutdown()
    assert not dialog.worker.isRunning()
    assert not directory.exists()
    app.processEvents()
    dialog.close()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
