import json
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtCore import QUrl  # DEPRECATED: legacy HTML export.
from PySide6.QtGui import QDesktopServices  # DEPRECATED: legacy HTML export.
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.azure_utils import get_blob_service_client
from gui.constants import DEFAULT_ACCOUNT_URL, PROCESSED_RESULTS_CONTAINER
from gui.tar_streaming import (
    iter_tar_members_from_blob,
    safe_relative_stem_from_tar_name,
    safe_stem_from_tar_name,
)
from gui.workers import ContainerListWorker
from gui.data_viewer_mosaics import describe_mosaic_progress, export_tar_mosaics
from gui.gps_locations_utils import (
    build_gps_plot_html,  # DEPRECATED: legacy HTML export.
    extract_gps_metadata_from_tiff,
    is_tar_blob,
    merge_processed_summaries,
)


MAX_TAR_STREAM_BYTES = 0.1 * 1024 * 1024
MAX_PARALLEL_TAR_STREAMS = 128
MAX_PARALLEL_SUMMARY_DOWNLOADS = 128


class GPSLocationsWorker(QThread):
    log = Signal(str)
    inventory_ready = Signal(int)
    tars_ready = Signal(list)
    progress = Signal(int, int, int)
    failed = Signal(str)
    scan_finished = Signal(int, int, bool, list)

    def __init__(self, account_url: str, containers: list, prefix: str, parent=None):
        super().__init__(parent)
        self.account_url = account_url
        self.containers = list(containers)
        self.prefix = prefix
        self.tar_count = 0
        self.location_count = 0
        self.error_message = None

    def _find_first_tif_location(self, blob_client, container, tar_name):
        stream_error = False

        def report_stream_error(message):
            nonlocal stream_error
            stream_error = True
            self.log.emit(f"WARNING {container}/{tar_name}: {message}")

        members = iter_tar_members_from_blob(
            blob_client,
            tar_name,
            log_cb=report_stream_error,
            max_bytes=MAX_TAR_STREAM_BYTES,
        )
        try:
            for member, extracted in members:
                if self.isInterruptionRequested():
                    return None
                if not member.name.lower().endswith((".tif", ".tiff")):
                    extracted.close()
                    continue

                try:
                    image_bytes = extracted.read()
                finally:
                    extracted.close()
                metadata = extract_gps_metadata_from_tiff(image_bytes)
                if metadata is None:
                    return None
                return {
                    "container": container,
                    "tar": tar_name,
                    "member": member.name,
                    **metadata,
                }
            if not stream_error:
                self.log.emit(f"No TIFF member found in {container}/{tar_name}")
            return None
        finally:
            members.close()

    def run(self):
        locations = []
        try:
            blob_service_client = get_blob_service_client(
                self.account_url, log_cb=self.log.emit
            )
            tar_blobs = []
            for container in self.containers:
                if self.isInterruptionRequested():
                    break
                container_client = blob_service_client.get_container_client(container)
                for blob in container_client.list_blobs(name_starts_with=self.prefix):
                    if self.isInterruptionRequested():
                        break
                    name = getattr(blob, "name", None)
                    if name and is_tar_blob(name):
                        tar_blobs.append((container, name))

            total = len(tar_blobs)
            self.tars_ready.emit(tar_blobs)
            self.inventory_ready.emit(total)
            self.log.emit(
                f"Found {total:,} TAR files across {len(self.containers)} container(s)."
            )
            container_clients = {
                name: blob_service_client.get_container_client(name)
                for name in self.containers
            }

            def scan_tar(container, tar_name):
                if self.isInterruptionRequested():
                    return None
                blob_client = container_clients[container].get_blob_client(tar_name)
                return self._find_first_tif_location(blob_client, container, tar_name)

            max_workers = min(MAX_PARALLEL_TAR_STREAMS, total) if total else 1
            self.log.emit(f"Scanning with up to {max_workers} parallel TAR streams.")

            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="gps-location-tar",
            )
            future_to_tar = {}
            try:
                for container, tar_name in tar_blobs:
                    if self.isInterruptionRequested():
                        break
                    future = executor.submit(scan_tar, container, tar_name)
                    future_to_tar[future] = (container, tar_name)

                for future in as_completed(future_to_tar):
                    container, tar_name = future_to_tar[future]
                    if self.isInterruptionRequested():
                        break

                    try:
                        location = future.result()
                        if location is not None:
                            locations.append(location)
                            self.location_count += 1
                    except Exception as exc:
                        self.log.emit(
                            f"WARNING failed to read {container}/{tar_name}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    self.tar_count += 1
                    self.progress.emit(
                        self.tar_count,
                        total,
                        self.location_count,
                    )
            finally:
                if self.isInterruptionRequested():
                    for future in future_to_tar:
                        future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

            self.scan_finished.emit(
                self.tar_count,
                self.location_count,
                self.isInterruptionRequested(),
                locations,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.error_message = message
            self.failed.emit(message)
            self.log.emit(f"ERROR scanning TAR files: {message}")
            self.scan_finished.emit(
                self.tar_count,
                self.location_count,
                self.isInterruptionRequested(),
                locations,
            )


class ProcessedSummaryWorker(QThread):
    log = Signal(str)
    inventory_ready = Signal(int)
    progress = Signal(int, int)
    processed_ready = Signal(object, int, int, bool)
    failed = Signal(str)

    def __init__(
        self, account_url: str, locations: list, parent=None, filter_to_locations=False,
    ):
        super().__init__(parent)
        self.account_url = account_url
        self.locations = list(locations)
        self.filter_to_locations = filter_to_locations

    def _download_summary(self, container_client, blob_name):
        raw = container_client.get_blob_client(blob_name).download_blob().readall()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        summary = json.loads(raw)
        if not isinstance(summary, dict):
            raise ValueError("summary JSON must contain an object")
        return summary

    def run(self):
        try:
            blob_service_client = get_blob_service_client(
                self.account_url, log_cb=self.log.emit
            )
            container_client = blob_service_client.get_container_client(
                PROCESSED_RESULTS_CONTAINER
            )
            selected_stems = {
                safe_relative_stem_from_tar_name(location["tar"]).replace("\\", "/")
                for location in self.locations
            } if self.filter_to_locations else set()
            summary_names = []
            skipped = 0
            for blob in container_client.list_blobs():
                if self.isInterruptionRequested():
                    break
                name = getattr(blob, "name", None)
                if name is None and isinstance(blob, dict):
                    name = blob.get("name")
                if name and str(name).rsplit("/", 1)[-1].lower() == "summary.json":
                    if self.filter_to_locations:
                        relative = f"/{name}".partition("/per_tar/")[2]
                        if relative.rpartition("/")[0] not in selected_stems:
                            skipped += 1
                            continue
                    summary_names.append(str(name))
            summary_names.sort()
            total = len(summary_names)
            self.inventory_ready.emit(total)
            self.log.emit(
                f"Found {total:,} summary.json file(s) in "
                f"{PROCESSED_RESULTS_CONTAINER}."
            )
            if self.filter_to_locations:
                self.log.emit(
                    f"Limited downloads to {len(self.locations):,} selected TIFF GPS "
                    f"location(s); skipped {skipped:,} summaries outside their "
                    "per_tar paths (including summaries without a per_tar path)."
                )
            if not summary_names:
                self.processed_ready.emit({}, 0, 0, self.isInterruptionRequested())
                return

            summaries = []
            completed = 0
            max_workers = min(MAX_PARALLEL_SUMMARY_DOWNLOADS, total)
            self.log.emit(
                f"Downloading summary files with up to {max_workers} parallel request(s)."
            )
            executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="processed-summary",
            )
            names = iter(summary_names)
            pending = {}

            def submit_next():
                if self.isInterruptionRequested():
                    return False
                try:
                    blob_name = next(names)
                except StopIteration:
                    return False
                pending[executor.submit(
                    self._download_summary, container_client, blob_name
                )] = blob_name
                return True

            try:
                for _ in range(max_workers):
                    if not submit_next():
                        break
                while pending and not self.isInterruptionRequested():
                    finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        blob_name = pending.pop(future)
                        try:
                            summaries.append((blob_name, future.result()))
                        except Exception as exc:
                            self.log.emit(
                                f"WARNING failed to download {PROCESSED_RESULTS_CONTAINER}/"
                                f"{blob_name}: {type(exc).__name__}: {exc}"
                            )
                        completed += 1
                        self.progress.emit(completed, total)
                        submit_next()
            finally:
                if self.isInterruptionRequested():
                    for future in pending:
                        future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)

            merged = merge_processed_summaries(
                self.locations, summaries, log_cb=self.log.emit
            )
            matched_count = sum(len(records) for records in merged.values())
            self.log.emit(
                f"Downloaded {len(summaries):,} valid summary file(s); matched "
                f"{matched_count:,} record(s) across {len(merged):,} model(s)."
            )
            self.processed_ready.emit(
                merged,
                len(summaries),
                matched_count,
                self.isInterruptionRequested(),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.failed.emit(message)
            self.log.emit(f"ERROR loading processed summaries: {message}")


class MosaicExportWorker(QThread):
    log = Signal(str)
    progress = Signal(object)
    exported = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self, account_url, container, tar_name, output_directory,
        count, width, height, skip_every, parent=None,
    ):
        super().__init__(parent)
        self.account_url = account_url
        self.container = container
        self.tar_name = tar_name
        self.output_directory = output_directory
        self.count = count
        self.width = width
        self.height = height
        self.skip_every = skip_every
        self.blob_size = None

    def run(self):
        try:
            client = get_blob_service_client(self.account_url, log_cb=self.log.emit)
            blob = client.get_container_client(self.container).get_blob_client(
                self.tar_name
            )
            self.blob_size = blob.get_blob_properties().size
            root = Path(self.output_directory).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            output = tempfile.mkdtemp(
                prefix=f"mosaics_{safe_stem_from_tar_name(self.tar_name)[:80]}_",
                dir=root,
            )
            self.log.emit(
                f"Exporting {self.container}/{self.tar_name} to {output}; "
                f"up to {self.count:,} mosaics of {self.width} x {self.height} pixels; "
                f"skip every {self.skip_every} image(s) (0 = no skipping)."
            )
            result = export_tar_mosaics(
                blob, self.tar_name, output, self.count, self.width, self.height,
                skip_every=self.skip_every,
                blob_size=self.blob_size,
                log_cb=self.log.emit,
                progress_cb=self.progress.emit,
                is_cancelled=self.isInterruptionRequested,
            )
            self.exported.emit(result, output)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.log.emit(f"ERROR exporting mosaics: {message}")
            self.failed.emit(message)


class GPSLocationsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        storage_box = QGroupBox("Azure Blob Storage")
        storage_layout = QHBoxLayout(storage_box)
        self.account_url = QLineEdit(DEFAULT_ACCOUNT_URL)
        self.refresh_containers_btn = QPushButton("List containers")
        storage_layout.addWidget(QLabel("Account URL:"))
        storage_layout.addWidget(self.account_url)
        storage_layout.addWidget(self.refresh_containers_btn)
        layout.addWidget(storage_box)

        targets_box = QGroupBox("TAR sources")
        targets_layout = QVBoxLayout(targets_box)
        target_controls = QHBoxLayout()
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Optional blob-name prefix")
        target_controls.addWidget(QLabel("Prefix:"))
        target_controls.addWidget(self.prefix_edit)
        targets_layout.addLayout(target_controls)
        self.containers_list = QListWidget()
        self.containers_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.containers_list.setMaximumHeight(125)
        targets_layout.addWidget(self.containers_list)
        layout.addWidget(targets_box)

        actions = QHBoxLayout()
        self.scan_btn = QPushButton("Scan TARs for first TIFF GPS")
        self.load_processed_btn = QPushButton("Load processed summaries")
        self.load_processed_btn.setToolTip(
            "Optional: apply TIFF capture-time filters in the GPS plot first to "
            "download summaries only for its visible TAR locations."
        )
        self.load_processed_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.show_plot_btn = QPushButton("Show GPS plot")
        self.show_plot_btn.setEnabled(False)
        self.save_plot_btn = QPushButton("Save Plotly HTML (deprecated)")  # DEPRECATED
        self.save_plot_btn.setEnabled(False)  # DEPRECATED
        actions.addWidget(self.scan_btn)
        actions.addWidget(self.load_processed_btn)
        actions.addWidget(self.stop_btn)
        actions.addWidget(self.show_plot_btn)
        actions.addWidget(self.save_plot_btn)  # DEPRECATED: legacy HTML export.
        self.status = QLabel("Select one or more containers, then scan.")
        self.status.setWordWrap(True)
        actions.addWidget(self.status, 1)
        layout.addLayout(actions)
        summary_filter_note = QLabel(
            "Optional summary date/datetime filter: Show GPS plot, set From/To datetime "
            "and Apply filter, then Load processed summaries. Only the visible TARs' "
            "per_tar summaries will be downloaded."
        )
        summary_filter_note.setWordWrap(True)
        layout.addWidget(summary_filter_note)

        self.mosaic_box = QGroupBox("Stream a polled TAR to composite mosaics")
        mosaic_layout = QVBoxLayout(self.mosaic_box)
        self.tar_combo = QComboBox()
        self.tar_combo.setMinimumContentsLength(30)
        self.tar_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.tar_combo.setToolTip("Includes TARs without GPS coordinates.")
        mosaic_layout.addWidget(self.tar_combo)
        dimensions = QHBoxLayout()
        self.mosaic_count = QSpinBox()
        self.mosaic_count.setRange(1, 1000000)
        self.mosaic_count.setValue(10)
        self.mosaic_width = QSpinBox()
        self.mosaic_width.setRange(1, 16384)
        self.mosaic_width.setValue(2048)
        self.mosaic_height = QSpinBox()
        self.mosaic_height.setRange(1, 16384)
        self.mosaic_height.setValue(2048)
        self.mosaic_skip = QSpinBox()
        self.mosaic_skip.setRange(0, 1000000)
        self.mosaic_skip.setSpecialValueText("No skipping")
        self.mosaic_skip.setToolTip(
            "Drop every Sth image: 2 drops images 2, 4, 6...; "
            "0 keeps all; 1 skips all. Skipped images are not decoded."
        )
        for label, control in (
            ("Mosaics (N):", self.mosaic_count),
            ("Width (px):", self.mosaic_width),
            ("Height (px):", self.mosaic_height),
            ("Skip every Sth:", self.mosaic_skip),
        ):
            dimensions.addWidget(QLabel(label))
            dimensions.addWidget(control)
        mosaic_layout.addLayout(dimensions)
        destination = QHBoxLayout()
        self.mosaic_directory = QLineEdit(str(Path.home() / "plankton_mosaics"))
        self.mosaic_browse_btn = QPushButton("Browse...")
        self.export_mosaics_btn = QPushButton("Export mosaics")
        destination.addWidget(QLabel("Output directory:"))
        destination.addWidget(self.mosaic_directory, 1)
        destination.addWidget(self.mosaic_browse_btn)
        destination.addWidget(self.export_mosaics_btn)
        mosaic_layout.addLayout(destination)
        self.mosaic_box.setEnabled(False)
        layout.addWidget(self.mosaic_box)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.locations = []
        self.processed_results = None
        self.processed_summary_count = 0
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(180)
        layout.addWidget(self.log, 1)

        self._container_worker = None
        self._scan_worker = None
        self._processed_worker = None
        self._mosaic_worker = None
        self._plot_window = None
        self._scan_account_url = None
        self.refresh_containers_btn.clicked.connect(self.refresh_containers)
        self.scan_btn.clicked.connect(self.start_scan)
        self.load_processed_btn.clicked.connect(self.load_processed_summaries)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.show_plot_btn.clicked.connect(self.show_plot)
        self.save_plot_btn.clicked.connect(self.save_plot)  # DEPRECATED
        self.mosaic_browse_btn.clicked.connect(self._browse_mosaic_directory)
        self.export_mosaics_btn.clicked.connect(self.export_mosaics)

    def _append_log(self, message):
        self.log.append(str(message))

    def refresh_containers(self):
        if self._container_worker is not None and self._container_worker.isRunning():
            return
        self.refresh_containers_btn.setEnabled(False)
        self._append_log("Listing Azure containers...")
        self._container_worker = ContainerListWorker(self.account_url.text().strip())
        self._container_worker.log.connect(self._append_log)
        self._container_worker.containers_ready.connect(self._populate_containers)
        self._container_worker.finished.connect(
            lambda: self.refresh_containers_btn.setEnabled(True)
        )
        self._container_worker.start()

    def _populate_containers(self, containers):
        self.containers_list.clear()
        for container in containers:
            self.containers_list.addItem(QListWidgetItem(container))

    def start_scan(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        containers = [
            item.text().strip() for item in self.containers_list.selectedItems()
        ]
        if not containers:
            QMessageBox.warning(
                self, "No containers selected", "Select at least one container to scan."
            )
            return

        self.locations = []
        self.tar_combo.clear()
        self.mosaic_box.setEnabled(False)
        self._scan_account_url = self.account_url.text().strip()
        self.processed_results = None
        self.processed_summary_count = 0
        self.load_processed_btn.setEnabled(False)
        self.show_plot_btn.setEnabled(False)
        if self._plot_window is not None:
            self._plot_window.close()
        self.save_plot_btn.setEnabled(False)  # DEPRECATED
        self.log.clear()
        self.progress.setRange(0, 0)
        self.status.setText("Listing TAR files...")
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._scan_worker = GPSLocationsWorker(
            self._scan_account_url,
            containers,
            self.prefix_edit.text().strip(),
        )
        self._scan_worker.log.connect(self._append_log)
        self._scan_worker.inventory_ready.connect(self._inventory_ready)
        self._scan_worker.tars_ready.connect(self._populate_tars)
        self._scan_worker.progress.connect(self._update_progress)
        self._scan_worker.failed.connect(
            lambda message: self.status.setText(f"Scan failed: {message}")
        )
        self._scan_worker.scan_finished.connect(self._scan_finished)
        self._scan_worker.finished.connect(self._scan_thread_finished)
        self._scan_worker.start()

    def stop_scan(self):
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._scan_worker.requestInterruption()
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopping scan...")
        elif self._processed_worker is not None and self._processed_worker.isRunning():
            self._processed_worker.requestInterruption()
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopping processed summary downloads...")
        elif self._mosaic_worker is not None and self._mosaic_worker.isRunning():
            self._mosaic_worker.requestInterruption()
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopping mosaic export...")

    def _populate_tars(self, tars):
        self.tar_combo.clear()
        for container, tar_name in sorted(tars):
            self.tar_combo.addItem(f"{container}/{tar_name}", (container, tar_name))

    def _browse_mosaic_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Mosaic output directory", self.mosaic_directory.text()
        )
        if directory:
            self.mosaic_directory.setText(directory)

    def export_mosaics(self):
        if self._mosaic_worker is not None and self._mosaic_worker.isRunning():
            return
        selected = self.tar_combo.currentData()
        directory = self.mosaic_directory.text().strip()
        if not selected or not directory:
            QMessageBox.warning(
                self, "Missing export settings",
                "Select a polled TAR and an output directory.",
            )
            return
        container, tar_name = selected
        self.mosaic_box.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.load_processed_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setRange(0, self.mosaic_count.value())
        self.progress.setValue(0)
        self.status.setText("Streaming TAR for mosaics...")
        self._mosaic_worker = MosaicExportWorker(
            self._scan_account_url, container, tar_name, directory,
            self.mosaic_count.value(), self.mosaic_width.value(),
            self.mosaic_height.value(), self.mosaic_skip.value(),
        )
        self._mosaic_worker.log.connect(self._append_log)
        self._mosaic_worker.progress.connect(self._mosaic_progress)
        self._mosaic_worker.exported.connect(self._mosaics_exported)
        self._mosaic_worker.failed.connect(
            lambda message: self.status.setText(f"Mosaic export failed: {message}")
        )
        self._mosaic_worker.finished.connect(self._mosaic_thread_finished)
        self._mosaic_worker.start()

    def _mosaic_progress(self, progress):
        self.progress.setValue(progress.saved)
        self.status.setText(
            describe_mosaic_progress(
                progress, self._mosaic_worker.tar_name, self._mosaic_worker.blob_size
            )
        )

    def _mosaics_exported(self, progress, directory):
        self._mosaic_progress(progress)
        reason = (
            "Stopped" if progress.cancelled
            else "End of TAR" if progress.exhausted
            else "Finished"
        )
        self.status.setText(f"{reason}: {self.status.text()}; output: {directory}")

    def _mosaic_thread_finished(self):
        self.scan_btn.setEnabled(True)
        self.load_processed_btn.setEnabled(bool(self.locations))
        self.mosaic_box.setEnabled(bool(self.tar_combo.count()))
        self.stop_btn.setEnabled(False)

    def _inventory_ready(self, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        if total == 0:
            self.status.setText("No TAR files found.")
        else:
            self.status.setText(f"Scanning 0 of {total:,} TAR files...")

    def _update_progress(self, processed, total, location_count):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(processed)
        self.status.setText(
            f"Scanned {processed:,} of {total:,} TAR files; "
            f"found {location_count:,} GPS locations."
        )

    def _scan_finished(self, processed, location_count, cancelled, locations):
        self.locations = locations
        self.load_processed_btn.setEnabled(bool(locations))
        self.show_plot_btn.setEnabled(bool(locations))
        self.save_plot_btn.setEnabled(bool(locations))  # DEPRECATED
        self.progress.setValue(processed)
        if self._scan_worker and self._scan_worker.error_message:
            return
        if cancelled:
            self.status.setText(
                f"Stopped after {processed:,} TAR files; "
                f"found {location_count:,} GPS locations."
            )
        else:
            self.status.setText(
                f"Finished: scanned {processed:,} TAR files; "
                f"found {location_count:,} GPS locations."
            )

    def load_processed_summaries(self):
        if self._processed_worker is not None and self._processed_worker.isRunning():
            return
        if not self.locations:
            QMessageBox.warning(
                self,
                "No raw locations",
                "Scan raw TAR files for GPS locations before loading processed summaries.",
            )
            return

        locations = self.locations
        filter_to_locations = (
            self._plot_window is not None
            and self._plot_window.locations == self.locations
            and any(value is not None for value in self._plot_window.applied_datetime_range)
        )
        if filter_to_locations:
            locations = [point["location"] for point in self._plot_window.visible_points]
            if not locations:
                QMessageBox.warning(
                    self, "No selected GPS locations",
                    "No GPS locations match the applied map filters. "
                    "Adjust or clear the filters before loading summaries.",
                )
                return
            self._append_log(
                f"Using {len(locations):,} visible GPS location(s) from the applied "
                "TIFF capture-time filter."
            )

        self.processed_results = None
        self.processed_summary_count = 0
        self.mosaic_box.setEnabled(False)
        self.show_plot_btn.setEnabled(False)
        self.save_plot_btn.setEnabled(False)  # DEPRECATED
        self.load_processed_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Listing processed summary files...")
        self._append_log(
            f"Loading processed summaries from {PROCESSED_RESULTS_CONTAINER}..."
        )

        self._processed_worker = ProcessedSummaryWorker(
            self._scan_account_url or self.account_url.text().strip(),
            locations,
            filter_to_locations=filter_to_locations,
        )
        self._processed_worker.log.connect(self._append_log)
        self._processed_worker.inventory_ready.connect(
            self._processed_inventory_ready
        )
        self._processed_worker.progress.connect(self._processed_progress)
        self._processed_worker.processed_ready.connect(self._processed_loaded)
        self._processed_worker.failed.connect(self._processed_failed)
        self._processed_worker.finished.connect(self._processed_thread_finished)
        self._processed_worker.start()

    def _processed_inventory_ready(self, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(0)
        if total == 0:
            self.status.setText("No processed summary files found.")
        else:
            self.status.setText(f"Downloading 0 of {total:,} processed summaries...")

    def _processed_progress(self, completed, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(completed)
        self.status.setText(
            f"Downloaded {completed:,} of {total:,} processed summaries..."
        )

    def _processed_loaded(self, results, summary_count, matched_count, cancelled):
        self.processed_results = results
        self.processed_summary_count = summary_count
        self.show_plot_btn.setEnabled(bool(self.locations))
        self.save_plot_btn.setEnabled(bool(self.locations))  # DEPRECATED
        if self._plot_window is not None and self._plot_window.isVisible():
            self.show_plot()
        if cancelled:
            self.status.setText(
                f"Stopped after downloading {summary_count:,} valid summaries; "
                f"matched {matched_count:,} record(s)."
            )
        else:
            self.status.setText(
                f"Loaded {summary_count:,} summaries; matched {matched_count:,} "
                f"record(s) across {len(results):,} model(s)."
            )

    def _processed_failed(self, message):
        self.status.setText(f"Processed summary loading failed: {message}")

    def _processed_thread_finished(self):
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        self.scan_btn.setEnabled(True)
        self.load_processed_btn.setEnabled(bool(self.locations))
        self.show_plot_btn.setEnabled(bool(self.locations))
        self.save_plot_btn.setEnabled(bool(self.locations))  # DEPRECATED
        self.mosaic_box.setEnabled(bool(self.tar_combo.count()))
        self.stop_btn.setEnabled(False)

    def show_plot(self):
        if not self.locations:
            QMessageBox.warning(
                self, "No GPS locations", "Scan TAR files for GPS locations first."
            )
            return
        try:
            if self._plot_window is None:
                from gui.gps_plot_window import GPSPlotWindow

                self._plot_window = GPSPlotWindow(
                    self.locations, self.processed_results,
                    self.processed_summary_count, parent=self,
                    account_url=self._scan_account_url or self.account_url.text().strip(),
                )
            elif (
                self._plot_window.locations != self.locations
                or self._plot_window.processed_results is not self.processed_results
                or self._plot_window.processed_summary_count != self.processed_summary_count
            ):
                self._plot_window.set_data(
                    self.locations, self.processed_results, self.processed_summary_count,
                    account_url=self._scan_account_url or self.account_url.text().strip(),
                )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._append_log(f"ERROR opening GPS plot: {message}")
            QMessageBox.critical(self, "Could not open GPS plot", message)
            return
        self._plot_window.show()
        self._plot_window.raise_()
        self._plot_window.activateWindow()

    def save_plot(self):  # DEPRECATED
        if not self.locations:  # DEPRECATED
            return  # DEPRECATED
        filename, _ = QFileDialog.getSaveFileName(  # DEPRECATED
            self,  # DEPRECATED
            "Save GPS locations plot",  # DEPRECATED
            str(Path.home() / "gps_locations.html"),  # DEPRECATED
            "HTML files (*.html)",  # DEPRECATED
        )  # DEPRECATED
        if not filename:  # DEPRECATED
            return  # DEPRECATED
# DEPRECATED
        output_path = Path(filename)  # DEPRECATED
        if output_path.suffix.lower() != ".html":  # DEPRECATED
            output_path = output_path.with_suffix(".html")  # DEPRECATED
        try:  # DEPRECATED
            output_path.write_text(  # DEPRECATED
                build_gps_plot_html(  # DEPRECATED
                    self.locations,  # DEPRECATED
                    self.processed_results,  # DEPRECATED
                    self.processed_summary_count,  # DEPRECATED
                ),  # DEPRECATED
                encoding="utf-8",  # DEPRECATED
            )  # DEPRECATED
        except Exception as exc:  # DEPRECATED
            QMessageBox.critical(  # DEPRECATED
                self,  # DEPRECATED
                "Could not save Plotly HTML",  # DEPRECATED
                f"{type(exc).__name__}: {exc}",  # DEPRECATED
            )  # DEPRECATED
            return  # DEPRECATED
# DEPRECATED
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_path.resolve()))):  # DEPRECATED
            QMessageBox.warning(  # DEPRECATED
                self,  # DEPRECATED
                "Plot saved",  # DEPRECATED
                f"The plot was saved to {output_path}, but could not be opened automatically.",  # DEPRECATED
            )  # DEPRECATED
        self.status.setText(f"Plotly HTML saved to {output_path}")  # DEPRECATED
# DEPRECATED
    def _scan_thread_finished(self):
        self.scan_btn.setEnabled(True)
        self.mosaic_box.setEnabled(bool(self.tar_combo.count()))
        self.stop_btn.setEnabled(False)
