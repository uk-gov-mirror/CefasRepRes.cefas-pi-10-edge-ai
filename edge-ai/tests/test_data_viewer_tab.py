import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QDateTime, QEventLoop, QTimer, Qt
from PySide6.QtWidgets import QApplication, QWidget

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = REPO_ROOT / "gui"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GUI_DIR))


@pytest.fixture(scope="module")
def tab_module():
    # Container listing otherwise imports unrelated model inference dependencies.
    with patch.dict(sys.modules, {
        "gui.workers": SimpleNamespace(ContainerListWorker=object),
    }):
        spec = importlib.util.spec_from_file_location(
            "data_viewer_tab_under_test", GUI_DIR / "gps_locations_tab.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tab(app, tab_module):
    widget = tab_module.GPSLocationsTab()
    yield widget
    widget.close()


def test_mosaic_controls_unlock_after_scan_even_without_gps(tab):
    assert not tab.mosaic_box.isEnabled()
    tab._populate_tars([("b", "same.tar"), ("a", "same.tar")])
    assert not tab.mosaic_box.isEnabled()
    tab._scan_finished(2, 0, False, [])
    tab._scan_thread_finished()

    assert tab.mosaic_box.isEnabled()
    assert tab.tar_combo.count() == 2
    assert tab.tar_combo.itemText(0) == "a/same.tar"
    assert tab.tar_combo.itemData(1) == ("b", "same.tar")
    assert not tab.load_processed_btn.isEnabled()
    assert not tab.save_plot_btn.isEnabled()


def test_empty_inventory_keeps_export_disabled(tab):
    tab._populate_tars([])
    tab._scan_finished(0, 0, False, [])
    tab._scan_thread_finished()
    assert not tab.mosaic_box.isEnabled()


def test_plot_button_opens_native_window_and_refreshes_loaded_summaries(tab, monkeypatch):
    location = {
        "container": "raw", "tar": "sample.tar", "member": "one.tif",
        "latitude": 51.5, "longitude": 0.16,
    }
    assert not tab.show_plot_btn.isEnabled()
    tab._scan_account_url = "https://scanned-account"
    tab._scan_finished(1, 1, False, [location])
    assert tab.show_plot_btn.isEnabled()
    monkeypatch.setattr(
        tab, "save_plot", lambda: pytest.fail("Native plot must not export HTML"),
    )
    tab.show_plot_btn.click()
    window = tab._plot_window
    assert window.isVisible()
    assert window.isWindow()
    assert window.parent() == tab
    assert window.account_url == "https://scanned-account"
    assert len(window.visible_points) == 1
    tab._processed_loaded({
        "model.pt": [{
            "location": location, "summary": {"class_counts": {"copepod": 7}},
        }],
    }, 1, 1, False)
    assert tab._plot_window is window
    assert [point["count"] for point in window.count_points] == [7]
    window.class_filter.item(0).setCheckState(Qt.Unchecked)
    window.close()
    tab.show_plot_btn.click()
    assert tab._plot_window is window
    assert window.isVisible()
    assert window.count_points == []
    window.close()


def test_new_scan_hides_old_plot_and_disables_plot_button(tab, tab_module, monkeypatch):
    tab._scan_finished(1, 1, False, [{
        "latitude": 51.5, "longitude": 0.16, "container": "raw", "tar": "one.tar",
    }])
    tab.show_plot()
    window = tab._plot_window
    tab._populate_containers(["source"])
    tab.containers_list.item(0).setSelected(True)
    monkeypatch.setattr(tab_module.GPSLocationsWorker, "start", lambda self: None)
    tab.start_scan()
    assert not window.isVisible()
    assert not tab.show_plot_btn.isEnabled()


def test_plot_failure_reports_error_and_processed_failure_unlocks_plot(tab, monkeypatch):
    tab.locations = [{"latitude": 51.5, "longitude": 0.16}]

    def fail(*args, **kwargs):
        raise ImportError("Cartopy is missing")

    with patch.dict(sys.modules, {"gps_plot_window": SimpleNamespace(GPSPlotWindow=fail)}):
        messages = []
        monkeypatch.setattr(
            "PySide6.QtWidgets.QMessageBox.critical",
            lambda parent, title, message: messages.append((title, message)),
        )
        tab.show_plot()
    assert tab._plot_window is None
    assert messages == [("Could not open GPS plot", "ImportError: Cartopy is missing")]
    assert "ERROR opening GPS plot" in tab.log.toPlainText()
    tab.show_plot_btn.setEnabled(False)
    tab._processed_failed("download failed")
    tab._processed_thread_finished()
    assert tab.show_plot_btn.isEnabled()


def test_processed_summaries_use_same_account_as_scanned_tar(tab, tab_module, monkeypatch):
    tab.locations = [{"container": "raw", "tar": "one.tar", "latitude": 51, "longitude": 0}]
    tab._scan_account_url = "https://scanned-account"
    tab.account_url.setText("https://edited-account")
    monkeypatch.setattr(tab_module.ProcessedSummaryWorker, "start", lambda self: None)
    tab.load_processed_summaries()
    assert tab._processed_worker.account_url == "https://scanned-account"
    assert not tab._processed_worker.filter_to_locations


@pytest.mark.parametrize("bounds,indices", [
    ("none", [0, 1, 2, 3]), ("start", [1, 2]), ("end", [0]), ("both", [0]), ("day", [0, 1]),
])
def test_summary_downloads_use_applied_gps_datetime_selection(
    tab, tab_module, monkeypatch, bounds, indices,
):
    locations = [
        {
            "container": "raw", "tar": "2025-07-09/1210.tar",
            "latitude": 51.5, "longitude": 0.16, "datetime": "2025-07-09T12:10:00",
        },
        {
            "container": "raw", "tar": "2025-07-09/1220.tar",
            "latitude": 51.6, "longitude": 0.17, "datetime": "2025-07-09T12:20:00",
        },
        {
            "container": "raw", "tar": "2025-07-10/1210.tar",
            "latitude": 51.7, "longitude": 0.18, "datetime": "2025-07-10T12:10:00",
        },
        {"container": "raw", "tar": "unknown.tar", "latitude": 51.8, "longitude": 0.19},
    ]
    tab._scan_finished(4, 4, False, locations)
    tab.show_plot()
    window = tab._plot_window
    exact = QDateTime.fromString("2025-07-09T12:10:00", Qt.ISODate)
    window.start_enabled.setChecked(bounds in ("start", "both", "day"))
    window.end_enabled.setChecked(bounds in ("end", "both", "day"))
    window.date_start.setDateTime(exact.addSecs(600) if bounds == "start" else exact)
    window.date_end.setDateTime(exact)
    if bounds == "day":
        window.date_start.setDateTime(QDateTime.fromString("2025-07-09T00:00:00", Qt.ISODate))
        window.date_end.setDateTime(QDateTime.fromString("2025-07-09T23:59:59", Qt.ISODate))
    window.apply_filter()
    previous_range = window.applied_datetime_range
    monkeypatch.setattr(tab_module.ProcessedSummaryWorker, "start", lambda self: None)

    tab.load_processed_summaries()

    worker = tab._processed_worker
    assert worker.locations == [locations[index] for index in indices]
    assert worker.filter_to_locations == (bounds != "none")
    assert len(tab.locations) == 4
    tab._processed_loaded({}, 0, 0, False)
    assert window.applied_datetime_range == previous_range
    assert [point["location"] for point in window.visible_points] == worker.locations
    window.close()


def test_empty_gps_datetime_selection_does_not_start_downloads_or_clear_results(
    tab, monkeypatch,
):
    tab._scan_finished(1, 1, False, [{
        "container": "raw", "tar": "2025-07-09/1210.tar",
        "latitude": 51.5, "longitude": 0.16, "datetime": "2025-07-09T12:10:00",
    }])
    tab.show_plot()
    window = tab._plot_window
    window.start_enabled.setChecked(True)
    window.date_start.setDateTime(QDateTime.fromString("2025-07-10T00:00:00", Qt.ISODate))
    window.apply_filter()
    old_results = {"retained": []}
    tab.processed_results = old_results
    messages = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda parent, title, text: messages.append((title, text)),
    )

    tab.load_processed_summaries()

    assert tab._processed_worker is None
    assert tab.processed_results is old_results
    assert messages[0][0] == "No selected GPS locations"
    window.close()


@pytest.mark.parametrize("filtered", [False, True])
def test_summary_worker_filters_per_tar_paths_before_downloading(tab_module, monkeypatch, filtered):
    selected = {
        "container": "raw", "tar": "2025-07-09/1210.tar",
        "latitude": 51.5, "longitude": 0.16, "datetime": "2025-07-09T12:10:00",
    }
    summary = {
        "container": "raw", "source_tar_blob": selected["tar"], "model_blob": "model.pt",
        "class_counts": {"copepod": 7}, "generated_utc": "2026-10-22T16:18:30Z",
    }
    summaries = {
        "runs/guilocalinference-2026-10-22T16-18-30Z-tender-quail/per_tar/2025-07-09/1210/summary.json": summary,
        "runs/another-run/per_tar/2025-07-09/1210/summary.json": {
            **summary, "model_blob": "another-model.pt",
        },
        "runs/another-run/per_tar/2025-07-09/1220/summary.json": {
            **summary, "source_tar_blob": "2025-07-09/1220.tar",
        },
        "runs/another-run/per_tar/2025-07-10/1210/summary.json": {
            **summary, "source_tar_blob": "2025-07-10/1210.tar",
        },
        "runs/another-run/per_tar/2025-07-09/12100/summary.json": {
            **summary, "source_tar_blob": "2025-07-09/12100.tar",
        },
        "runs/legacy/summary.json": summary,
    }
    container = SimpleNamespace(
        list_blobs=lambda: [
            {"name": name} for name in [
                *summaries, "runs/another-run/per_tar/2025-07-09/1210/predictions.json",
            ]
        ],
    )
    service = SimpleNamespace(get_container_client=lambda name: container)
    monkeypatch.setattr(tab_module, "get_blob_service_client", lambda *a, **k: service)
    worker = tab_module.ProcessedSummaryWorker(
        "https://example", [selected], filter_to_locations=filtered,
    )
    downloaded = []
    expected_names = list(summaries)[:2] if filtered else list(summaries)

    def download(client, name):
        assert name in expected_names
        downloaded.append(name)
        return summaries[name]

    monkeypatch.setattr(worker, "_download_summary", download)
    completed = []
    inventories = []
    errors = []
    worker.processed_ready.connect(lambda *args: completed.append(args))
    worker.inventory_ready.connect(inventories.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert errors == []
    assert sorted(downloaded) == sorted(expected_names)
    assert inventories == [len(expected_names)]
    merged, count, matches, cancelled = completed[0]
    assert count == len(expected_names)
    assert matches == 2
    assert not cancelled
    assert all(record["location"] == selected for records in merged.values() for record in records)


@pytest.mark.parametrize("tar", ["1210.tar", "1210.TAR", "1210.tar.gz", "1210.tgz"])
def test_summary_path_filter_handles_root_paths_and_compressed_tars(
    tab_module, monkeypatch, tar,
):
    name = "per_tar/2025-07-09/1210/SUMMARY.JSON"
    container = SimpleNamespace(list_blobs=lambda: [SimpleNamespace(name=name)])
    service = SimpleNamespace(get_container_client=lambda name: container)
    monkeypatch.setattr(tab_module, "get_blob_service_client", lambda *a, **k: service)
    worker = tab_module.ProcessedSummaryWorker(
        "https://example",
        [{"container": "raw", "tar": f"2025-07-09/{tar}"}],
        filter_to_locations=True,
    )
    downloaded = []
    monkeypatch.setattr(
        worker, "_download_summary",
        lambda client, blob: downloaded.append(blob) or {},
    )
    worker.run()
    assert downloaded == [name]


def test_no_matching_summary_paths_downloads_nothing(tab_module, monkeypatch):
    container = SimpleNamespace(list_blobs=lambda: [
        SimpleNamespace(name="runs/run/per_tar/2025-07-10/1210/summary.json"),
    ])
    service = SimpleNamespace(get_container_client=lambda name: container)
    monkeypatch.setattr(tab_module, "get_blob_service_client", lambda *a, **k: service)
    worker = tab_module.ProcessedSummaryWorker(
        "https://example", [{"container": "raw", "tar": "2025-07-09/1210.tar"}],
        filter_to_locations=True,
    )
    monkeypatch.setattr(
        worker, "_download_summary",
        lambda *args: pytest.fail("No unmatched summaries should be downloaded"),
    )
    completed = []
    worker.processed_ready.connect(lambda *args: completed.append(args))
    worker.run()
    assert completed == [({}, 0, 0, False)]


def test_scan_worker_inventories_all_tars_not_only_gps_matches(tab_module, monkeypatch):
    blobs = ["yes.tar", "no-gps.TAR", "nested/compressed.tgz", "ignored.txt"]
    container = SimpleNamespace(
        list_blobs=lambda **kwargs: [SimpleNamespace(name=name) for name in blobs],
        get_blob_client=lambda name: name,
    )
    service = SimpleNamespace(get_container_client=lambda name: container)
    monkeypatch.setattr(tab_module, "get_blob_service_client", lambda *a, **kw: service)
    worker = tab_module.GPSLocationsWorker("https://example", ["source"], "")
    monkeypatch.setattr(
        worker, "_find_first_tif_location",
        lambda blob, container, name: {"tar": name} if name == "yes.tar" else None,
    )
    inventory = []
    completed = []
    worker.tars_ready.connect(inventory.append)
    worker.scan_finished.connect(lambda *args: completed.append(args))

    worker.run()

    assert inventory == [[("source", name) for name in blobs[:3]]]
    assert completed == [(3, 1, False, [{"tar": "yes.tar"}])]


def test_processed_worker_logs_duplicate_drop_and_exports_only_latest_counts(
    tab_module, monkeypatch,
):
    summary = {
        "container": "raw-images",
        "source_tar_blob": "sample.tar",
        "model_blob": "model.pt",
        "sample_ratio_requested": 0.25,
        "sample_percent_requested": 25.0,
    }
    summaries = {
        "runs/a-new/summary.json": {
            **summary,
            "generated_utc": "2026-10-27T08:00:00Z",
            "class_counts": {"copepod": 7},
        },
        "runs/z-old/summary.json": {
            **summary,
            "generated_utc": "2026-10-26T08:00:00Z",
            "class_counts": {"copepod": 100},
        },
    }
    container = SimpleNamespace(
        list_blobs=lambda: [SimpleNamespace(name=name) for name in summaries],
    )
    service = SimpleNamespace(get_container_client=lambda name: container)
    monkeypatch.setattr(tab_module, "get_blob_service_client", lambda *a, **kw: service)
    worker = tab_module.ProcessedSummaryWorker(
        "https://example",
        [{"container": "raw-images", "tar": "sample.tar",
          "latitude": 51.5, "longitude": 0.16}],
    )
    monkeypatch.setattr(worker, "_download_summary", lambda client, name: summaries[name])
    completed = []
    messages = []
    errors = []
    worker.processed_ready.connect(lambda *args: completed.append(args))
    worker.log.connect(messages.append)
    worker.failed.connect(errors.append)

    worker.run()

    assert errors == []
    assert len(completed) == 1
    merged, downloaded_count, matched_count, cancelled = completed[0]
    assert (downloaded_count, matched_count, cancelled) == (2, 1, False)
    assert merged["model.pt"][0]["summary_blob"] == "runs/a-new/summary.json"
    assert merged["model.pt"][0]["summary"]["class_counts"] == {"copepod": 7}
    warnings = [message for message in messages if "WARNING" in message]
    assert len(warnings) == 1
    assert "oldest run was dropped before merge" in warnings[0]
    assert "runs/z-old/summary.json" in warnings[0]
    html = tab_module.build_gps_plot_html(worker.locations, merged, downloaded_count)
    assert '"class_counts": {"copepod": 7}' in html
    assert '"copepod": 100' not in html
    assert "runs/z-old/summary.json" not in html


def test_new_scan_clears_old_inventory_and_locks_exports(tab, tab_module, monkeypatch):
    tab._populate_containers(["source"])
    tab.containers_list.item(0).setSelected(True)
    tab._populate_tars([("old", "previous.tar")])
    tab._scan_thread_finished()
    tab.account_url.setText("https://scan-account")
    monkeypatch.setattr(tab_module.GPSLocationsWorker, "start", lambda self: None)

    tab.start_scan()

    assert tab.tar_combo.count() == 0
    assert not tab.mosaic_box.isEnabled()
    assert tab._scan_worker.account_url == "https://scan-account"


def test_export_uses_selected_tar_scan_account_and_requested_settings(tab, tab_module, monkeypatch, tmp_path):
    tab._scan_account_url = "https://scan-account"
    tab.account_url.setText("https://changed-account")
    tab._populate_tars([("source-a", "one.tar"), ("source-b", "two.tar")])
    tab._scan_thread_finished()
    tab.tar_combo.setCurrentIndex(1)
    tab.mosaic_directory.setText(str(tmp_path))
    tab.mosaic_count.setValue(3)
    tab.mosaic_width.setValue(321)
    tab.mosaic_height.setValue(123)
    tab.mosaic_skip.setValue(4)
    monkeypatch.setattr(tab_module.MosaicExportWorker, "start", lambda self: None)

    tab.export_mosaics()

    worker = tab._mosaic_worker
    assert worker.account_url == "https://scan-account"
    assert (worker.container, worker.tar_name) == ("source-b", "two.tar")
    assert worker.output_directory == str(tmp_path)
    assert (worker.count, worker.width, worker.height, worker.skip_every) == (3, 321, 123, 4)
    assert not tab.mosaic_box.isEnabled()
    assert not tab.scan_btn.isEnabled()
    assert not tab.load_processed_btn.isEnabled()
    assert tab.stop_btn.isEnabled()
    assert tab.progress.maximum() == 3

    monkeypatch.setattr(worker, "isRunning", lambda: True)
    tab.stop_scan()
    assert not tab.stop_btn.isEnabled()
    assert tab.status.text() == "Stopping mosaic export..."
    worker.finished.emit()
    assert tab.mosaic_box.isEnabled()
    assert tab.scan_btn.isEnabled()
    assert not tab.stop_btn.isEnabled()


def test_worker_uses_unique_directories_and_propagates_progress(tab_module, monkeypatch, tmp_path):
    from data_viewer_mosaics import MosaicProgress

    service = SimpleNamespace(
        get_container_client=lambda name: SimpleNamespace(
            get_blob_client=lambda name: SimpleNamespace(
                get_blob_properties=lambda: SimpleNamespace(size=1000)
            )
        )
    )
    monkeypatch.setattr(tab_module, "get_blob_service_client", lambda *a, **kw: service)
    outputs = []
    result = MosaicProgress(saved=2, seen=4, included=4)

    def export(blob, tar_name, directory, count, width, height, **kwargs):
        assert tar_name == "prefix/same.tar"
        assert (count, width, height, kwargs["skip_every"]) == (2, 4, 4, 0)
        assert kwargs["blob_size"] == 1000
        kwargs["progress_cb"](result)
        outputs.append(Path(directory))
        return result

    monkeypatch.setattr(tab_module, "export_tar_mosaics", export)
    worker = tab_module.MosaicExportWorker(
        "https://account", "source", "prefix/same.tar", str(tmp_path), 2, 4, 4, 0
    )
    completed = []
    progress = []
    worker.exported.connect(lambda *args: completed.append(args))
    worker.progress.connect(progress.append)
    worker.run()
    worker.run()

    assert len(set(outputs)) == 2
    assert all(path.parent == tmp_path and path.is_dir() for path in outputs)
    assert completed == [(result, str(path)) for path in outputs]
    assert progress == [result, result]


def test_worker_failure_emits_error_not_success(tab_module, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("Azure unavailable")

    monkeypatch.setattr(tab_module, "get_blob_service_client", fail)
    worker = tab_module.MosaicExportWorker(
        "https://account", "source", "sample.tar", str(tmp_path), 2, 4, 4, 0
    )
    errors = []
    completed = []
    worker.failed.connect(errors.append)
    worker.exported.connect(lambda *args: completed.append(args))
    worker.run()

    assert errors == ["OSError: Azure unavailable"]
    assert completed == []


def test_full_scan_to_streamed_export_flow(tab, tab_module, app, monkeypatch, tmp_path):
    from PIL import Image
    from test_data_viewer_mosaics import FakeBlob, image_bytes, tar_bytes

    payload = tar_bytes([
        (f"{index}.tif", image_bytes(color))
        for index, color in enumerate(("red", "green", "blue", "yellow", "cyan", "magenta"))
    ])

    class Blob(FakeBlob):
        def get_blob_properties(self):
            return SimpleNamespace(size=len(payload))

    downloads = []

    def get_blob(name):
        blob = Blob(payload)
        downloads.append(blob)
        return blob

    service = SimpleNamespace(
        get_container_client=lambda name: SimpleNamespace(
            list_blobs=lambda **kwargs: [SimpleNamespace(name="images.tar")],
            get_blob_client=get_blob,
        )
    )
    monkeypatch.setattr(tab_module, "get_blob_service_client", lambda *a, **kw: service)
    monkeypatch.setattr(tab_module, "extract_gps_metadata_from_tiff", lambda data: None)

    def finish(worker):
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        worker.finished.connect(loop.quit)
        timer.start(5000)
        if not worker.isFinished():
            loop.exec()
        app.processEvents()
        assert worker.isFinished()
        timer.stop()

    tab._populate_containers(["source"])
    tab.containers_list.item(0).setSelected(True)
    tab.start_scan()
    finish(tab._scan_worker)
    assert tab.tar_combo.currentData() == ("source", "images.tar")
    assert tab.mosaic_box.isEnabled()
    assert not tab.locations

    tab.mosaic_directory.setText(str(tmp_path))
    tab.mosaic_count.setValue(2)
    tab.mosaic_width.setValue(4)
    tab.mosaic_height.setValue(2)
    tab.mosaic_skip.setValue(3)
    tab.export_mosaics_btn.click()
    finish(tab._mosaic_worker)

    outputs = sorted(tmp_path.glob("mosaics_images_*/composite_*.png"))
    assert len(outputs) == 2
    for output in outputs:
        with Image.open(output) as image:
            assert image.size == (4, 2)
    with Image.open(outputs[1]) as image:
        assert image.getpixel((0, 0)) == (255, 255, 0)
        assert image.getpixel((2, 0)) == (0, 255, 255)
    assert tab.progress.value() == 2
    assert "reached image 5" in tab.status.text()
    assert "4 included, 1 skipped" in tab.status.text()
    assert "TAR byte" in tab.log.toPlainText()
    assert tab.mosaic_box.isEnabled()
    assert all(blob.downloader.closed for blob in downloads)


def test_app_labels_tab_data_viewer(tab_module, app):
    with patch.dict(sys.modules, {
        "gui.edge_tab": SimpleNamespace(EdgeAITab=QWidget),
        "gui.train_tab": SimpleNamespace(ModelTrainingTab=QWidget),
        "gui.application_validation_tab": SimpleNamespace(ApplicationValidationTab=QWidget),
        "gui.gps_locations_tab": tab_module,
    }):
        spec = importlib.util.spec_from_file_location("data_viewer_app_under_test", GUI_DIR / "app.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        window = module.MainWindow()
    assert window.centralWidget().tabText(3) == "Data viewer"
    window.close()
