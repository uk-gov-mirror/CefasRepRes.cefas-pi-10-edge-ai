import json
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QCoreApplication, QSize, Qt, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from gui.azure_utils import get_blob_service_client
from gui.constants import PROCESSED_RESULTS_CONTAINER
from gui.prediction_sources import _prediction_records_from_payload
from gui.tar_streaming import iter_tar_members_from_blob


def predictions_blob_for_summary(summary_blob):
    directory, separator, filename = str(summary_blob or "").rpartition("/")
    if filename.lower() != "summary.json":
        raise ValueError("The selected summary must reference a summary.json blob.")
    return f"{directory}{separator}predictions.json"


def load_tar_prediction_images(
    service, container, tar_name, summary_blob, predicted_label, output_directory,
    image_cb, progress_cb, is_cancelled, log_cb=None,
):
    """Stream one TAR once, saving only TIFFs predicted as the requested label."""
    if not predicted_label:
        raise ValueError("Select a predicted label.")
    if is_cancelled():
        return 0, True
    predictions_blob = predictions_blob_for_summary(summary_blob)
    if log_cb:
        log_cb(f"Downloading {PROCESSED_RESULTS_CONTAINER}/{predictions_blob}")
    raw = (
        service.get_container_client(PROCESSED_RESULTS_CONTAINER)
        .get_blob_client(predictions_blob).download_blob().readall()
    )
    records = _prediction_records_from_payload(json.loads(raw))
    if records is None:
        raise ValueError(f"Unexpected predictions payload in {predictions_blob}")
    prefix = f"{container}/{tar_name}!/"
    pending = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid prediction record in {predictions_blob}")
        if record.get("predicted_label") != predicted_label:
            continue
        uri = record.get("blob_member_uri")
        if not isinstance(uri, str) or not uri.startswith(prefix):
            raise ValueError(f"Prediction does not reference the selected TAR: {uri!r}")
        member_name = uri[len(prefix):]
        if member_name.lower().endswith((".tif", ".tiff")):
            pending.add(member_name)
    total = len(pending)
    progress_cb(0, total)
    if is_cancelled():
        return 0, True
    if not pending:
        return 0, False

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    blob = service.get_container_client(container).get_blob_client(tar_name)
    members = iter_tar_members_from_blob(
        blob, tar_name, log_cb=log_cb, raise_on_open_error=True,
    )
    saved = 0
    try:
        for member, extracted in members:
            try:
                if is_cancelled():
                    return saved, True
                if member.name not in pending:
                    continue
                payload = extracted.read()
            finally:
                extracted.close()
            image_path = output / f"{saved:06d}.tif"
            thumbnail_path = output / f"{saved:06d}.png"
            try:
                with Image.open(BytesIO(payload)) as source:
                    thumbnail = source.convert("RGB")
                try:
                    thumbnail.thumbnail((160, 160), Image.Resampling.LANCZOS)
                    thumbnail.save(thumbnail_path)
                finally:
                    thumbnail.close()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Could not decode TIFF {member.name}: {exc}") from exc
            image_path.write_bytes(payload)
            pending.remove(member.name)
            saved += 1
            image_cb(str(image_path), str(thumbnail_path), member.name)
            progress_cb(saved, total)
            if not pending:
                break
        if is_cancelled():
            return saved, True
        if pending:
            missing = ", ".join(sorted(pending)[:5])
            raise FileNotFoundError(
                f"{len(pending)} predicted TIFF member(s) not found in {tar_name}: {missing}"
            )
        return saved, False
    finally:
        members.close()


class TarImagesWorker(QThread):
    image_ready = Signal(str, str, str)
    progress = Signal(int, int)
    loaded = Signal(int, bool)
    failed = Signal(str)
    log = Signal(str)

    def __init__(
        self, account_url, container, tar_name, summary_blob, predicted_label,
        output_directory, parent=None,
    ):
        super().__init__(parent)
        self.account_url = account_url
        self.arguments = (
            container, tar_name, summary_blob, predicted_label, output_directory,
        )

    def run(self):
        try:
            service = get_blob_service_client(self.account_url, log_cb=self.log.emit)
            count, cancelled = load_tar_prediction_images(
                service, *self.arguments,
                image_cb=self.image_ready.emit,
                progress_cb=self.progress.emit,
                is_cancelled=self.isInterruptionRequested,
                log_cb=self.log.emit,
            )
            self.loaded.emit(count, cancelled)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class TarImagesDialog(QDialog):
    def __init__(
        self, account_url, container, tar_name, summary_blob, predicted_label, parent=None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"{predicted_label} - {container}/{tar_name}")
        self.resize(900, 650)
        self._directory = tempfile.TemporaryDirectory(prefix="gps_tar_images_")
        self._close_requested = False
        layout = QVBoxLayout(self)
        source = QLabel(f"Summary: {summary_blob}\nPredicted label: {predicted_label}")
        source.setWordWrap(True)
        source.setTextFormat(Qt.PlainText)
        layout.addWidget(source)
        self.status = QLabel("Loading predictions...")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)
        self.images = QListWidget()
        self.images.setViewMode(QListWidget.IconMode)
        self.images.setResizeMode(QListWidget.Adjust)
        self.images.setMovement(QListWidget.Static)
        self.images.setIconSize(QSize(160, 160))
        self.images.setGridSize(QSize(190, 210))
        self.images.setWordWrap(True)
        self.images.itemDoubleClicked.connect(self._show_image)
        layout.addWidget(self.images, 1)
        layout.addWidget(QLabel("Double-click an image to view it at full size."))
        self.stop_button = QPushButton("Stop")
        layout.addWidget(self.stop_button)
        self.worker = TarImagesWorker(
            account_url, container, tar_name, summary_blob, predicted_label,
            self._directory.name, parent=self,
        )
        self.worker.image_ready.connect(self._add_image)
        self.worker.progress.connect(self._progress)
        self.worker.loaded.connect(self._loaded)
        self.worker.failed.connect(self._failed)
        self.worker.log.connect(self.status.setText)
        self.worker.finished.connect(self._finished)
        self.stop_button.clicked.connect(self._stop)
        QCoreApplication.instance().aboutToQuit.connect(self._shutdown)
        self.worker.start()

    def _add_image(self, image_path, thumbnail_path, member_name):
        item = QListWidgetItem(QIcon(thumbnail_path), member_name, self.images)
        item.setToolTip(member_name)
        item.setData(Qt.UserRole, image_path)

    def _progress(self, completed, total):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(completed)
        self.status.setText(f"Loaded {completed:,} of {total:,} matching TIFF(s).")

    def _loaded(self, count, cancelled):
        if cancelled:
            self.status.setText(f"Stopped; {count:,} TIFF(s) loaded.")
        elif not count:
            self.status.setText("No TIFF predictions match this label in the selected TAR.")
        else:
            self.status.setText(f"Loaded all {count:,} matching TIFF(s).")

    def _failed(self, message):
        self.progress.setRange(0, 1)
        self.status.setText(f"Image loading failed: {message}")

    def _stop(self):
        if self.worker.isRunning():
            self.worker.requestInterruption()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping image loading...")
        else:
            self.close()

    def _finished(self):
        self.stop_button.setText("Close")
        self.stop_button.setEnabled(True)
        if self._close_requested:
            self.close()

    def _show_image(self, item):
        try:
            with Image.open(item.data(Qt.UserRole)) as source:
                image = source.convert("RGB")
            try:
                pixmap = QPixmap.fromImage(ImageQt(image))
            finally:
                image.close()
        except (OSError, ValueError) as exc:
            self._failed(f"Could not open {item.text()}: {exc}")
            return
        viewer = QDialog(self)
        viewer.setAttribute(Qt.WA_DeleteOnClose)
        viewer.setWindowTitle(item.text())
        viewer.resize(800, 600)
        layout = QVBoxLayout(viewer)
        scroll = QScrollArea()
        label = QLabel()
        label.setPixmap(pixmap)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        viewer.show()

    def reject(self):
        if self.worker.isRunning():
            self._close_requested = True
            self._stop()
            return
        self._directory.cleanup()
        super().reject()

    def _shutdown(self):
        self.worker.requestInterruption()
        self.worker.wait()
        self._directory.cleanup()

    def closeEvent(self, event):
        if self.worker.isRunning():
            self._close_requested = True
            self._stop()
            event.ignore()
            return
        super().closeEvent(event)
