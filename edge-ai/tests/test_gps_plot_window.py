import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = REPO_ROOT / "gui"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GUI_DIR))

from gps_plot_window import GPSPlotWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def plot_data():
    locations = [
        {
            "latitude": 51.5, "longitude": 0.16,
            "datetime": "2025-01-02T10:20:30",
            "container": "raw", "tar": "sample.tar", "member": "one.tif",
        },
        {
            "latitude": 52.0, "longitude": 0.2,
            "datetime": "2025-01-03T10:20:30",
            "container": "other", "tar": "other.tar", "member": "two.tif",
        },
        {
            "latitude": -50.0, "longitude": -1.0,
            "container": "raw", "tar": "unprocessed.tar",
        },
    ]

    def record(index, percentage, size, counts):
        return {
            "location": locations[index],
            "summary_blob": f"runs/{index}-{percentage}-{size}/per_tar/summary.json",
            "summary": {
                "sample_percent_requested": percentage, "size_threshold_px": size,
                "class_counts": counts,
            },
        }

    merged = {
        "model-one.pt": [
            record(0, "25", "64.0", {"copepod": 3, "fish": 2}),
            record(0, 100, None, {"copepod": 7, "fish": "not-a-count"}),
        ],
        "model-two.pt": [
            record(0, 25.0, 128, {"copepod": 11}),
            record(1, "0.0", 128, {"copepod": 4}),
        ],
    }
    return locations, merged


@pytest.fixture
def window(app, plot_data):
    locations, merged = plot_data
    widget = GPSPlotWindow(locations, merged, processed_summary_count=4)
    yield widget
    widget.close()


def counts(window):
    return [point["count"] for point in window.count_points]


def choose(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_native_map_renders_offline_with_area_scaled_circles(window, monkeypatch):
    from cartopy.io import Downloader

    def reject_download(*args, **kwargs):
        pytest.fail("Native map must not download any background data")

    monkeypatch.setattr(Downloader, "path", reject_download)
    window.canvas.draw()
    assert len(window.axes.images) == 1
    assert counts(window) == [23, 4]
    sizes = window.count_markers.get_sizes()
    assert sizes.tolist() == pytest.approx([800, 800 * 4 / 23])
    assert window.location_markers.get_offsets().tolist() == [
        [0.16, 51.5], [0.2, 52.0], [-1.0, -50.0],
    ]
    assert window.result_count.text() == "3 of 3 locations; 4 matched summary record(s)"
    assert window.toolbar.actions()


@pytest.mark.parametrize("value,expected", [("64", [5]), ("inf", [7]), ("128", [11, 4])])
def test_size_filters_normalize_values_and_counts(window, value, expected):
    choose(window.size_threshold_filter, value)
    window.apply_filter()
    assert counts(window) == expected


def test_model_sampling_and_container_filters_combine_and_clear(window):
    choose(window.model_filter, "model-two.pt")
    choose(window.sample_percent_filter, 0.0)
    choose(window.container_filter, "other")
    window.apply_filter()
    assert counts(window) == [4]
    assert window.result_count.text() == "1 of 3 locations; 1 matched summary record(s)"
    choose(window.model_filter, "model-one.pt")
    window.apply_filter()
    assert not window.visible_points
    assert window.location_markers.get_offsets().shape == (0, 2)
    assert window.count_markers.get_sizes().size == 0
    window.clear_filter()
    assert counts(window) == [23, 4]
    assert len(window.visible_points) == 3


def test_datetime_filters_include_endpoints_hide_unknown_and_reject_reversed_range(window):
    window.start_enabled.setChecked(True)
    window.end_enabled.setChecked(True)
    exact = QDateTime.fromString("2025-01-02T10:20:30", Qt.ISODate)
    window.date_start.setDateTime(exact)
    window.date_end.setDateTime(exact)
    window.apply_filter()
    assert counts(window) == [23]
    assert len(window.visible_points) == 1
    window.date_end.setDateTime(exact.addSecs(-1))
    window.apply_filter()
    assert "End datetime must be on or after start datetime." == window.filter_message.text()
    assert counts(window) == [23]
    window.clear_filter()
    assert window.filter_message.text() == ""
    assert len(window.visible_points) == 3


def test_class_checkboxes_update_immediately_and_clear_restores_all(window):
    for index in range(window.class_filter.count()):
        item = window.class_filter.item(index)
        if item.text() == "fish":
            item.setCheckState(Qt.Unchecked)
    assert counts(window) == [21, 4]
    assert window.class_selection_count.text() == "1 of 2 classes selected"
    for index in range(window.class_filter.count()):
        window.class_filter.item(index).setCheckState(Qt.Unchecked)
    assert counts(window) == []
    assert len(window.visible_points) == 3
    window.clear_filter()
    assert counts(window) == [23, 4]
    assert window.class_selection_count.text() == "2 of 2 classes selected"


def test_hover_includes_capture_source_and_counts(window):
    from matplotlib.backend_bases import MouseEvent

    window.canvas.draw()
    x, y = window.axes.transData.transform((0.16, 51.5))
    event = MouseEvent("motion_notify_event", window.canvas, x, y)
    window._show_hover(event)
    assert window.hover.get_visible()
    for detail in (
        "Latitude: 51.50000", "Longitude: 0.16000",
        "Datetime: 2025-01-02T10:20:30", "Container: raw",
        "TAR: sample.tar", "TIFF: one.tif",
        "Matched processed summaries: 3", "Selected-class count: 23",
    ):
        assert detail in window.hover.get_text()
    window._show_hover(MouseEvent("motion_notify_event", window.canvas, 0, 0))
    assert not window.hover.get_visible()


def test_hover_does_not_change_map_geometry_or_extent(window):
    from matplotlib.backend_bases import MouseEvent

    window.visible_points[0]["location"]["tar"] = (
        "2025-01-02/1210/" + "long-archive-name-" * 3 + ".tar"
    )
    window.canvas.draw()
    window.canvas.draw()
    original_bounds = window.axes.bbox.bounds
    original_extent = window.axes.get_xlim(), window.axes.get_ylim()
    for _ in range(3):
        x, y = window.axes.transData.transform((0.16, 51.5))
        window._show_hover(MouseEvent("motion_notify_event", window.canvas, x, y))
        window.canvas.draw()
        assert window.hover.get_visible()
        assert window.axes.bbox.bounds == pytest.approx(original_bounds, abs=1e-6)
        assert window.axes.get_xlim() == pytest.approx(original_extent[0])
        assert window.axes.get_ylim() == pytest.approx(original_extent[1])
        window._show_hover(MouseEvent("motion_notify_event", window.canvas, 0, 0))
        window.canvas.draw()
        assert window.axes.bbox.bounds == pytest.approx(original_bounds, abs=1e-6)


@pytest.mark.parametrize("bounds", ["start", "end", "both"])
def test_applied_datetime_range_survives_summary_refresh(window, plot_data, bounds):
    locations, merged = plot_data
    exact = QDateTime.fromString("2025-01-02T10:20:30", Qt.ISODate)
    window.start_enabled.setChecked(bounds in ("start", "both"))
    window.end_enabled.setChecked(bounds in ("end", "both"))
    window.date_start.setDateTime(exact)
    window.date_end.setDateTime(exact)
    window.apply_filter()
    previous_range = window.applied_datetime_range
    previous_locations = [point["location"] for point in window.visible_points]
    window.set_data(locations, merged, 4)
    assert window.applied_datetime_range == previous_range
    assert [point["location"] for point in window.visible_points] == previous_locations
    window.set_data([{**locations[0], "tar": "new-scan.tar"}])
    assert window.applied_datetime_range == (None, None)


def test_invalid_datetime_edit_does_not_replace_applied_download_selection(window):
    window.start_enabled.setChecked(True)
    window.end_enabled.setChecked(True)
    window.apply_filter()
    previous_range = window.applied_datetime_range
    previous_points = window.visible_points
    window.date_end.setDateTime(window.date_start.dateTime().addSecs(-1))
    window.apply_filter()
    assert window.filter_message.text()
    assert window.applied_datetime_range == previous_range
    assert window.visible_points == previous_points


def test_toolbar_navigation_restores_map_extent(window):
    original = window.axes.get_xlim(), window.axes.get_ylim()
    window.toolbar.push_current()
    window.axes.set_xlim(0, 1)
    window.axes.set_ylim(51, 53)
    window.toolbar.push_current()
    window.toolbar.back()
    assert window.axes.get_xlim() == pytest.approx(original[0])
    assert window.axes.get_ylim() == pytest.approx(original[1])
    window.toolbar.forward()
    assert window.axes.get_xlim() == pytest.approx((0, 1))
    assert window.axes.get_ylim() == pytest.approx((51, 53))
    window.toolbar.home()
    assert window.axes.get_xlim() == pytest.approx(original[0])


def test_raw_locations_and_empty_processed_results_remain_visible(app, plot_data):
    locations, _ = plot_data
    widget = GPSPlotWindow(locations)
    assert widget.processed_box.isHidden()
    assert len(widget.visible_points) == 3
    assert counts(widget) == []
    widget.set_data(locations, {}, 0)
    assert not widget.processed_box.isHidden()
    assert len(widget.visible_points) == 3
    assert counts(widget) == []
    widget.close()


def test_missing_and_nonfinite_class_counts_do_not_create_circles(window, plot_data):
    locations, _ = plot_data
    record = {
        "location": locations[0],
        "summary": {"class_counts": {
            "nan": float("nan"), "infinity": float("inf"), "negative": -3,
            "missing": None, "string": "4", "zero": 0,
        }},
    }
    window.set_data(locations, {"model": [record]}, 1)
    assert counts(window) == [4]


def test_empty_locations_report_error(app):
    with pytest.raises(ValueError, match="no GPS locations"):
        GPSPlotWindow([])


def click_location(window, index=0, button=1):
    from matplotlib.backend_bases import MouseEvent

    window.canvas.draw()
    location = window.visible_points[index]["location"]
    x, y = window.axes.transData.transform((location["longitude"], location["latitude"]))
    window._select_tar(MouseEvent("button_press_event", window.canvas, x, y, button=button))


def test_map_selection_requires_one_summary_not_just_one_model(window):
    assert not window.get_images_button.isEnabled()
    click_location(window)
    assert window.tar_selection.currentData() == "raw/sample.tar"
    assert window.selection_marker.get_offsets().tolist() == [[0.16, 51.5]]
    assert not window.get_images_button.isEnabled()
    assert "exactly one" in window.tar_summary_status.text()
    choose(window.model_filter, "model-one.pt")
    window.apply_filter()
    assert not window.get_images_button.isEnabled()
    choose(window.sample_percent_filter, 25.0)
    window.apply_filter()
    assert window.get_images_button.isEnabled()
    assert window.predicted_label_filter.count() == 2
    assert window._selected_summary_blob.endswith("0-25-64.0/per_tar/summary.json")
    window.clear_filter()
    assert not window.get_images_button.isEnabled()
    assert window.predicted_label_filter.count() == 0


def test_all_models_can_load_images_if_selected_tar_has_one_summary(window):
    choose(window.tar_selection, "other/other.tar")
    assert window.model_filter.currentData() is None
    assert window.get_images_button.isEnabled()
    assert window.predicted_label_filter.currentText() == "copepod"
    choose(window.tar_selection, "raw/unprocessed.tar")
    assert not window.get_images_button.isEnabled()
    assert "No processed summary" in window.tar_summary_status.text()


def test_filters_drop_hidden_selection_and_class_counts_do_not_resolve_ambiguity(window):
    choose(window.tar_selection, "raw/sample.tar")
    for index in range(window.class_filter.count()):
        window.class_filter.item(index).setCheckState(Qt.Unchecked)
    assert window._selected_point() is not None
    assert not window.get_images_button.isEnabled()
    choose(window.container_filter, "other")
    window.apply_filter()
    assert window.tar_selection.currentData() is None
    assert window.selection_marker.get_offsets().shape == (0, 2)


def test_duplicate_location_records_share_one_summary_and_overlapping_tars_are_selectable(
    window, plot_data,
):
    locations, merged = plot_data
    locations = [locations[0], {**locations[0], "tar": "overlap.tar"}]
    record = merged["model-one.pt"][0]
    window.set_data(locations, {"model": [record, record]}, 1)
    click_location(window)
    assert window.get_images_button.isEnabled()
    assert window.tar_selection.count() == 3
    choose(window.tar_selection, "raw/overlap.tar")
    assert not window.get_images_button.isEnabled()
    assert window.selection_marker.get_offsets().tolist() == [[0.16, 51.5]]


def test_raw_or_missing_summary_blob_disables_images(window, plot_data):
    locations, merged = plot_data
    record = {**merged["model-one.pt"][0], "summary_blob": None}
    window.set_data(locations, {"model": [record]}, 1)
    choose(window.tar_selection, "raw/sample.tar")
    assert not window.get_images_button.isEnabled()
    record["summary_blob"] = "runs/not-a-summary.json"
    window.set_data(locations, {"model": [record]}, 1)
    assert not window.get_images_button.isEnabled()
    assert "summary.json blob" in window.tar_summary_status.text()
    window.set_data(locations)
    choose(window.tar_selection, "raw/sample.tar")
    assert not window.get_images_button.isEnabled()


def test_map_selection_ignores_toolbar_navigation_and_right_click(window):
    click_location(window, button=3)
    assert window.tar_selection.currentData() is None
    window.toolbar.pan()
    click_location(window)
    assert window.tar_selection.currentData() is None
    window.toolbar.pan()
    click_location(window)
    assert window.tar_selection.currentData() == "raw/sample.tar"


def test_get_images_passes_selected_tar_summary_label_and_account(window, monkeypatch):
    from PySide6.QtWidgets import QDialog
    import gps_plot_window

    opened = []

    class Gallery(QDialog):
        def __init__(self, *args, parent=None):
            super().__init__(parent)
            opened.append(args)

    monkeypatch.setattr(gps_plot_window, "TarImagesDialog", Gallery)
    window.account_url = "https://scanned-account"
    choose(window.tar_selection, "raw/sample.tar")
    choose(window.model_filter, "model-one.pt")
    choose(window.sample_percent_filter, 25.0)
    window.apply_filter()
    window.predicted_label_filter.setCurrentText("fish")
    window.get_images_button.click()
    assert opened == [(
        "https://scanned-account", "raw", "sample.tar",
        window._selected_summary_blob, "fish",
    )]
    assert window._image_dialogs[0].isVisible()
