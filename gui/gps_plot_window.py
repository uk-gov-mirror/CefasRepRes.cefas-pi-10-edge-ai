import math
from datetime import datetime

import cartopy.crs as ccrs
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.constants import DEFAULT_ACCOUNT_URL
from gui.size_threshold import parse_size_threshold
from gui.gps_tar_images import TarImagesDialog, predictions_blob_for_summary


class GPSPlotWindow(QMainWindow):
    """Native GPS map with the former HTML viewer's summary filters."""

    def __init__(
        self, locations, merged_results=None, processed_summary_count=0, parent=None,
        account_url=DEFAULT_ACCOUNT_URL,
    ):
        super().__init__(parent, Qt.Window)
        self.account_url = account_url
        self._image_dialogs = []
        self.setWindowTitle("GPS locations")
        self.resize(1200, 800)
        content = QWidget()
        self.setCentralWidget(content)
        layout = QHBoxLayout(content)
        controls = QWidget()
        controls.setMaximumWidth(360)
        filters = QVBoxLayout(controls)

        dates = QGroupBox("TIFF capture time")
        date_layout = QFormLayout(dates)
        self.start_enabled = QCheckBox("From datetime")
        self.end_enabled = QCheckBox("To datetime")
        self.date_start = QDateTimeEdit()
        self.date_end = QDateTimeEdit()
        for checkbox, editor in (
            (self.start_enabled, self.date_start),
            (self.end_enabled, self.date_end),
        ):
            editor.setCalendarPopup(True)
            editor.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            editor.setEnabled(False)
            checkbox.toggled.connect(editor.setEnabled)
            date_layout.addRow(checkbox, editor)
        filters.addWidget(dates)
        note = QLabel(
            "Locations without a capture time are hidden while a datetime filter is active. "
            "Applied datetime filters also limit Load processed summaries to visible TARs."
        )
        note.setWordWrap(True)
        filters.addWidget(note)

        self.processed_box = QGroupBox("Processed summaries")
        processed_layout = QFormLayout(self.processed_box)
        processed_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        self.model_filter = QComboBox()
        self.sample_percent_filter = QComboBox()
        self.size_threshold_filter = QComboBox()
        self.container_filter = QComboBox()
        for label, combo in (
            ("Model", self.model_filter),
            ("Sample percent requested", self.sample_percent_filter),
            ("Size threshold (px; inf = not applied)", self.size_threshold_filter),
            ("Raw container", self.container_filter),
        ):
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            processed_layout.addRow(label, combo)
        self.class_selection_count = QLabel()
        self.class_filter = QListWidget()
        processed_layout.addRow(self.class_selection_count)
        processed_layout.addRow(self.class_filter)
        filters.addWidget(self.processed_box)
        self.summary_status = QLabel()
        self.summary_status.setWordWrap(True)
        filters.addWidget(self.summary_status)

        apply_button = QPushButton("Apply filter")
        clear_button = QPushButton("Clear filter")
        filters.addWidget(apply_button)
        filters.addWidget(clear_button)
        self.filter_message = QLabel()
        self.filter_message.setWordWrap(True)
        self.filter_message.setStyleSheet("color: #a12622")
        filters.addWidget(self.filter_message)
        images_box = QGroupBox("Selected TAR images")
        images_layout = QFormLayout(images_box)
        self.tar_selection = QComboBox()
        self.tar_selection.setMinimumContentsLength(12)
        self.tar_selection.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.tar_selection.setToolTip(
            "Click a map marker or select a TAR here. "
            "Use this list to choose between TARs at the same GPS position."
        )
        self.tar_summary_status = QLabel()
        self.tar_summary_status.setTextFormat(Qt.PlainText)
        self.tar_summary_status.setWordWrap(True)
        self.predicted_label_filter = QComboBox()
        self.get_images_button = QPushButton("Get images")
        images_layout.addRow("TAR", self.tar_selection)
        images_layout.addRow(self.tar_summary_status)
        images_layout.addRow("Predicted label", self.predicted_label_filter)
        images_layout.addRow(self.get_images_button)
        filters.addWidget(images_box)
        filters.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(310)
        layout.addWidget(scroll)

        plot_layout = QVBoxLayout()
        self.result_count = QLabel()
        plot_layout.addWidget(self.result_count)
        self.figure = Figure(layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(projection=ccrs.PlateCarree())
        # Cartopy bundles this raster, so the map needs no tile or coastline downloads.
        self.axes.stock_img(interpolation="bilinear")
        self.axes.gridlines(draw_labels=["bottom", "left"])
        self.axes.set_title("GPS locations")
        self.location_markers = self.axes.scatter(
            [], [], s=30, color="#1677a8", alpha=0.8, label="TAR locations",
        )
        self.count_markers = self.axes.scatter(
            [], [], s=[], facecolor="#e4572e", edgecolor="#9d301b",
            alpha=0.65, label="Selected class counts",
        )
        self.selection_marker = self.axes.scatter(
            [], [], s=100, facecolor="none", edgecolor="#222222", linewidth=2, zorder=5,
        )
        self.axes.legend(loc="upper right")
        self.hover = self.axes.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.95},
            zorder=10,
        )
        self.hover.set_in_layout(False)
        self.hover.set_visible(False)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        layout.addLayout(plot_layout, 1)
        self.canvas.mpl_connect("motion_notify_event", self._show_hover)
        self.canvas.mpl_connect("button_press_event", self._select_tar)

        apply_button.clicked.connect(self.apply_filter)
        clear_button.clicked.connect(self.clear_filter)
        self.class_filter.itemChanged.connect(self.apply_filter)
        self.tar_selection.currentIndexChanged.connect(self._update_tar_selection)
        self.predicted_label_filter.currentIndexChanged.connect(self._update_image_button)
        self.get_images_button.clicked.connect(self._get_images)
        self.set_data(locations, merged_results, processed_summary_count)

    def set_data(
        self, locations, merged_results=None, processed_summary_count=0, account_url=None,
    ):
        if not locations:
            raise ValueError("There are no GPS locations to plot.")
        previous_range = (
            self.applied_datetime_range
            if getattr(self, "locations", None) == locations else (None, None)
        )
        if account_url is not None:
            self.account_url = account_url
        self.locations = list(locations)
        self.processed_loaded = merged_results is not None
        self.results_by_location = {}
        options = [set() for _ in range(4)]
        classes = set()
        for model, records in (merged_results or {}).items():
            for record in records:
                summary = record.get("summary") or {}
                location = record.get("location") or {}
                container = str(location.get("container") or summary.get("container") or "")
                tar = str(location.get("tar") or summary.get("source_tar_blob") or "")
                percentage = summary.get("sample_percent_requested")
                percentage = float(percentage) if percentage is not None else None
                size = str(parse_size_threshold(summary.get("size_threshold_px")))
                counts = summary.get("class_counts")
                counts = counts if isinstance(counts, dict) else {}
                result = {
                    "filters": (model, percentage, size, container),
                    "class_counts": counts,
                    "summary_blob": record.get("summary_blob"),
                }
                self.results_by_location.setdefault((container, tar), []).append(result)
                for values, value in zip(options, result["filters"]):
                    if value is not None:
                        values.add(value)
                classes.update(str(name) for name in counts)

        for combo, values in zip(self._processed_filters(), options):
            combo.clear()
            combo.addItem("All", None)
            for value in sorted(values, key=str):
                combo.addItem(str(value), value)
        self.class_filter.blockSignals(True)
        self.class_filter.clear()
        for name in sorted(classes):
            item = QListWidgetItem(name, self.class_filter)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
        self.class_filter.blockSignals(False)
        self.processed_box.setVisible(self.processed_loaded)
        self.summary_status.setVisible(self.processed_loaded)
        self.summary_status.setText(
            f"{processed_summary_count:,} summary file(s) downloaded. "
            "Circle area represents the selected-class total for each matched TAR."
        )
        capture_times = [
            QDateTime.fromString(location.get("datetime") or "", Qt.ISODate)
            for location in self.locations
        ]
        capture_times = [value for value in capture_times if value.isValid()]
        self.date_start.setDateTime(
            min(capture_times) if capture_times else QDateTime.currentDateTime()
        )
        self.date_end.setDateTime(
            max(capture_times) if capture_times else QDateTime.currentDateTime()
        )
        self.clear_filter()
        if any(value is not None for value in previous_range):
            for value, checkbox, editor in (
                (previous_range[0], self.start_enabled, self.date_start),
                (previous_range[1], self.end_enabled, self.date_end),
            ):
                if value is not None:
                    checkbox.setChecked(True)
                    editor.setDateTime(QDateTime.fromString(value.isoformat(), Qt.ISODate))
            self.apply_filter()
        self._fit_locations()
        self.processed_results = merged_results
        self.processed_summary_count = processed_summary_count

    def _processed_filters(self):
        return (
            self.model_filter, self.sample_percent_filter,
            self.size_threshold_filter, self.container_filter,
        )

    def clear_filter(self):
        self.start_enabled.setChecked(False)
        self.end_enabled.setChecked(False)
        for combo in self._processed_filters():
            combo.setCurrentIndex(0)
        self.class_filter.blockSignals(True)
        for index in range(self.class_filter.count()):
            self.class_filter.item(index).setCheckState(Qt.Checked)
        self.class_filter.blockSignals(False)
        self.apply_filter()

    @staticmethod
    def _capture_time(value):
        # TIFF times have no timezone; match the HTML viewer's wall-clock filtering.
        return datetime.fromisoformat(value).replace(tzinfo=None)

    def apply_filter(self):
        start = (
            self.date_start.dateTime().toPython().replace(tzinfo=None)
            if self.start_enabled.isChecked() else None
        )
        end = (
            self.date_end.dateTime().toPython().replace(tzinfo=None)
            if self.end_enabled.isChecked() else None
        )
        if start is not None and end is not None and end < start:
            self.filter_message.setText("End datetime must be on or after start datetime.")
            return
        self.filter_message.clear()
        selected_filters = tuple(combo.currentData() for combo in self._processed_filters())
        classes = [
            self.class_filter.item(index).text()
            for index in range(self.class_filter.count())
            if self.class_filter.item(index).checkState() == Qt.Checked
        ]
        self.class_selection_count.setText(
            f"{len(classes)} of {self.class_filter.count()} classes selected"
        )
        visible = []
        matched_count = 0
        for location in self.locations:
            if start is not None or end is not None:
                value = location.get("datetime")
                if not value:
                    continue
                try:
                    captured = self._capture_time(value)
                except ValueError:
                    self.filter_message.setText(
                        f"Invalid capture datetime for {location.get('tar', '')}: {value!r}"
                    )
                    return
                if (
                    (start is not None and captured < start)
                    or (end is not None and captured > end)
                ):
                    continue
            results = [
                result for result in self.results_by_location.get(
                    (location.get("container"), location.get("tar")), ()
                )
                if all(
                    selected is None or selected == actual
                    for selected, actual in zip(selected_filters, result["filters"])
                )
            ]
            if any(value is not None for value in selected_filters) and not results:
                continue
            total = 0.0
            for result in results:
                for name in classes:
                    try:
                        count = float(result["class_counts"].get(name, 0))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(count) and count > 0 and math.isfinite(total + count):
                        total += count
            visible.append({
                "location": location, "result_count": len(results),
                "count": total, "results": results,
            })
            matched_count += len(results)
        self.applied_datetime_range = (start, end)
        self.visible_points = visible
        self.count_points = [point for point in visible if point["count"] > 0]
        self._set_offsets(self.location_markers, self.visible_points)
        self._set_offsets(self.count_markers, self.count_points)
        maximum = max((point["count"] for point in self.count_points), default=1)
        self.count_markers.set_sizes([
            800 * (point["count"] / maximum) for point in self.count_points
        ])
        text = f"{len(visible):,} of {len(self.locations):,} locations"
        if self.processed_loaded:
            text += f"; {matched_count:,} matched summary record(s)"
        self.result_count.setText(text)
        selected_tar = self.tar_selection.currentData()
        self.tar_selection.blockSignals(True)
        self.tar_selection.clear()
        self.tar_selection.addItem("Select a TAR or click the map", None)
        seen = set()
        for point in visible:
            location = point["location"]
            key = f"{location.get('container')}/{location.get('tar')}"
            if key not in seen:
                self.tar_selection.addItem(key, key)
                seen.add(key)
        index = self.tar_selection.findData(selected_tar)
        self.tar_selection.setCurrentIndex(max(0, index))
        self.tar_selection.blockSignals(False)
        self._update_tar_selection()
        self.hover.set_visible(False)
        self.canvas.draw_idle()

    def _selected_point(self):
        key = self.tar_selection.currentData()
        return next((
            point for point in self.visible_points
            if f"{point['location'].get('container')}/{point['location'].get('tar')}" == key
        ), None)

    def _update_tar_selection(self):
        point = self._selected_point()
        self._set_offsets(self.selection_marker, [point] if point else [])
        previous_label = self.predicted_label_filter.currentText()
        self.predicted_label_filter.blockSignals(True)
        self.predicted_label_filter.clear()
        self._selected_summary_blob = None
        if point is None:
            message = "Select a TAR to view its predicted images."
        else:
            results = point["results"]
            summaries = {result["summary_blob"] for result in results}
            if not results:
                message = "No processed summary matches this TAR."
            elif len(summaries) != 1 or None in summaries or "" in summaries:
                message = (
                    "Filter this TAR down to exactly one associated summary.json "
                    "(model, sampling percentage and size threshold)."
                )
            else:
                self._selected_summary_blob = next(iter(summaries))
                try:
                    predictions_blob_for_summary(self._selected_summary_blob)
                except ValueError as exc:
                    self._selected_summary_blob = None
                    message = str(exc)
                else:
                    message = f"Summary: {self._selected_summary_blob}"
                    labels = sorted({
                        str(name) for result in results for name in result["class_counts"]
                    })
                    self.predicted_label_filter.addItems(labels)
                    index = self.predicted_label_filter.findText(previous_label)
                    self.predicted_label_filter.setCurrentIndex(max(0, index))
        self.tar_summary_status.setText(message)
        self.predicted_label_filter.blockSignals(False)
        self._update_image_button()
        self.canvas.draw_idle()

    def _update_image_button(self):
        self.get_images_button.setEnabled(bool(
            self._selected_summary_blob and self.predicted_label_filter.currentText()
        ))
        self.predicted_label_filter.setEnabled(bool(self._selected_summary_blob))

    def _select_tar(self, event):
        if event.button != 1 or event.inaxes != self.axes or self.toolbar.mode:
            return
        for markers, points in (
            (self.count_markers, self.count_points),
            (self.location_markers, self.visible_points),
        ):
            contains, indices = markers.contains(event)
            if contains:
                location = points[indices["ind"][0]]["location"]
                key = f"{location.get('container')}/{location.get('tar')}"
                self.tar_selection.setCurrentIndex(self.tar_selection.findData(key))
                return

    def _get_images(self):
        point = self._selected_point()
        if point is None or not self.get_images_button.isEnabled():
            return
        location = point["location"]
        dialog = TarImagesDialog(
            self.account_url, location["container"], location["tar"],
            self._selected_summary_blob, self.predicted_label_filter.currentText(),
            parent=self,
        )
        self._image_dialogs.append(dialog)
        dialog.destroyed.connect(lambda: self._image_dialogs.remove(dialog))
        dialog.show()

    def closeEvent(self, event):
        for dialog in list(self._image_dialogs):
            dialog.close()
        super().closeEvent(event)

    @staticmethod
    def _set_offsets(markers, points):
        markers.set_offsets(np.array([
            [point["location"]["longitude"], point["location"]["latitude"]]
            for point in points
        ], dtype=float).reshape(-1, 2))

    def _fit_locations(self):
        longitudes = [float(location["longitude"]) for location in self.locations]
        latitudes = [float(location["latitude"]) for location in self.locations]
        lon_pad = max(0.1, (max(longitudes) - min(longitudes)) * 0.1)
        lat_pad = max(0.1, (max(latitudes) - min(latitudes)) * 0.1)
        self.axes.set_extent((
            max(-180, min(longitudes) - lon_pad),
            min(180, max(longitudes) + lon_pad),
            max(-90, min(latitudes) - lat_pad),
            min(90, max(latitudes) + lat_pad),
        ), crs=ccrs.PlateCarree())
        self.toolbar.update()
        self.canvas.draw_idle()

    def _show_hover(self, event):
        if event.inaxes == self.axes:
            for markers, points in (
                (self.count_markers, self.count_points),
                (self.location_markers, self.visible_points),
            ):
                contains, indices = markers.contains(event)
                if contains:
                    point = points[indices["ind"][0]]
                    location = point["location"]
                    self.hover.xy = (location["longitude"], location["latitude"])
                    self.hover.set_text(
                        f"Latitude: {location['latitude']:.5f}\n"
                        f"Longitude: {location['longitude']:.5f}\n"
                        f"Datetime: {location.get('datetime') or 'Datetime unavailable'}\n"
                        f"Container: {location.get('container', '')}\n"
                        f"TAR: {location.get('tar', '')}\n"
                        f"TIFF: {location.get('member', '')}\n"
                        f"Matched processed summaries: {point['result_count']}\n"
                        f"Selected-class count: {point['count']:g}"
                    )
                    self.hover.set_visible(True)
                    self.canvas.draw_idle()
                    return
        if self.hover.get_visible():
            self.hover.set_visible(False)
            self.canvas.draw_idle()
