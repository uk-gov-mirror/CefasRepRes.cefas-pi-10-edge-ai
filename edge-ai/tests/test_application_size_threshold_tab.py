import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_DIR = REPO_ROOT / "gui"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GUI_DIR))
sys.path.insert(0, str(REPO_ROOT / "edge-ai"))


@pytest.fixture(scope="module")
def tab_module():
    def parse_blob_url(url):
        parsed = urlsplit(url.split("?", 1)[0])
        parts = parsed.path.lstrip("/").split("/", 1)
        return (
            f"{parsed.scheme}://{parsed.netloc}",
            parts[0] if parts and parts[0] else None,
            parts[1] if len(parts) > 1 else "",
        )

    with patch.dict(sys.modules, {
        "pyvips": SimpleNamespace(Error=RuntimeError),
        "plot_class_counts_timeseries": SimpleNamespace(run_bayesian_timeseries_analysis=MagicMock()),
        "gui.azure_utils": SimpleNamespace(
            get_blob_service_client=MagicMock(),
            parse_blob_url=parse_blob_url,
            parse_model_input=lambda value: value if value and value.lower().endswith(".pt") else None,
            utc_now_iso=lambda: "2026-10-27T11:00:00Z",
        ),
        "gui.prediction_sources": SimpleNamespace(
            load_predictions_from_source=MagicMock(),
        ),
        "gui.workers": SimpleNamespace(
            ApplicationWorker=MagicMock(), ContainerListWorker=MagicMock(),
            ModelClassesWorker=MagicMock(), ModelListWorker=MagicMock(),
            MoveByClassWorker=MagicMock(),
        ),
    }):
        spec = importlib.util.spec_from_file_location(
            "gui.tab_under_test", GUI_DIR / "tab.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tab(app, tab_module):
    widget = tab_module.ApplicationValidationTab()
    yield widget
    widget.close()


@pytest.mark.parametrize("value", ["64", "inf"])
def test_generate_csv_prompts_for_size_and_passes_choice_to_worker(
    tab, tab_module, monkeypatch, tmp_path, value,
):
    csv_path = tmp_path / "selection.csv"
    targets = [{"container": "source", "prefix": ""}]
    service = SimpleNamespace(get_container_client=lambda name: SimpleNamespace(
        list_blobs=lambda **kwargs: [SimpleNamespace(name="sample.tar")],
    ))
    monkeypatch.setattr(tab_module.QFileDialog, "getSaveFileName", lambda *a: (str(csv_path), ""))
    monkeypatch.setattr(tab_module.QInputDialog, "getDouble", lambda *a: (1.0, True))
    get_text = MagicMock(return_value=(value, True))
    monkeypatch.setattr(tab_module.QInputDialog, "getText", get_text)
    monkeypatch.setattr(tab, "_tar_csv_targets_from_ui", lambda: targets)
    monkeypatch.setattr(tab, "_targets_from_list", lambda: targets)
    monkeypatch.setattr(tab, "_storage_client", lambda: service)
    monkeypatch.setattr(tab, "_get_selected_model_blob", lambda: "model.pt")
    tab.output_dir.setText(str(tmp_path))

    tab.gen_csv_btn.click()

    get_text.assert_called_once()
    assert get_text.call_args.kwargs["text"] == "inf"
    threshold = "inf" if value == "inf" else 64
    assert tab.selected_tars == [("source", "sample.tar", "", 1.0, threshold)]
    assert f"size threshold(s): {value} px" in tab.log.toPlainText()
    tab.run_btn.click()
    assert tab_module.ApplicationWorker.call_args.kwargs["selected_tars"] == tab.selected_tars


@pytest.mark.parametrize("text,accepted", [("-1", True), ("64", False)])
def test_invalid_or_cancelled_threshold_does_not_start_storage_access(
    tab, tab_module, monkeypatch, tmp_path, text, accepted,
):
    csv_path = tmp_path / "selection.csv"
    monkeypatch.setattr(tab_module.QFileDialog, "getSaveFileName", lambda *a: (str(csv_path), ""))
    monkeypatch.setattr(tab_module.QInputDialog, "getDouble", lambda *a: (1.0, True))
    monkeypatch.setattr(tab_module.QInputDialog, "getText", lambda *a, **kw: (text, accepted))
    warning = MagicMock()
    storage_client = MagicMock()
    monkeypatch.setattr(tab_module.QMessageBox, "warning", warning)
    monkeypatch.setattr(tab, "_tar_csv_targets_from_ui", lambda: [{"container": "source"}])
    monkeypatch.setattr(tab, "_storage_client", storage_client)

    tab.gen_csv_btn.click()

    assert not csv_path.exists()
    storage_client.assert_not_called()
    assert warning.call_count == int(accepted)


def test_generate_resume_csv_restores_settings_and_targets(tab, tab_module, monkeypatch, tmp_path):
    settings = {
        "account_url": "https://example.blob.core.windows.net",
        "output_container": "ml-prediction-results",
        "output_prefix": "runs",
        "output_slug": "guilocalinference-old-run",
        "summary_blob": "runs/guilocalinference-old-run/per_tar/2025-06-01/0734/summary.json",
        "summary": {
            "source_tar_blob": "2025-06-01/0734.tar",
            "temperature": 1.25,
        },
        "container": "source",
        "prefix": "2025-06-01/",
        "targets": [{"container": "source", "prefix": "2025-06-01/"}],
        "selected_tars": [{
            "container": "source",
            "source_tar_blob": "2025-06-01/0734.tar",
            "selected_prefix": "2025-06-01/",
            "sample_ratio": 0.25,
            "size_threshold_px": 80,
        }],
        "sort_by_size": True,
        "model_blob": "model-old.pt",
        "model_runid": "old-model",
        "sample_ratio": 0.25,
        "size_threshold_px": 80,
        "temperature": 1.25,
    }
    monkeypatch.setattr(tab_module.QFileDialog, "getSaveFileName", lambda *a: (str(tmp_path / "selection.csv"), ""))
    monkeypatch.setattr(tab_module, "load_resume_settings", lambda *a: settings)
    monkeypatch.setattr(tab, "_storage_client", lambda: object())
    get_double = MagicMock(side_effect=[(1.0, True), (0.25, True)])
    monkeypatch.setattr(tab_module.QInputDialog, "getDouble", get_double)
    monkeypatch.setattr(
        tab_module.QInputDialog,
        "getText",
        lambda *a, **kw: (kw["text"], True),
    )
    generate_csv = MagicMock()
    monkeypatch.setattr(tab_module, "generate_tar_selection_csv", generate_csv)
    monkeypatch.setattr(
        tab_module,
        "load_tar_selection_csv",
        lambda path: [("source", "2025-06-01/0734.tar", "2025-06-01/", 0.25, 80)],
    )
    tab.resume_previous_job_edit.setText(
        "https://example.blob.core.windows.net/ml-prediction-results/runs/guilocalinference-old-run/"
    )

    tab.generate_tar_csv()

    assert get_double.call_args_list[0].args[3] == 1.0
    assert get_double.call_args_list[1].args[3] == 0.25
    assert tab.model_url_edit.text() == "model-old.pt"
    assert tab.selected_targets_list.item(0).text() == "source | 2025-06-01/"
    assert tab.sort_tars_by_size.isChecked()
    assert generate_csv.call_args.kwargs["targets"] == settings["targets"]
    assert generate_csv.call_args.kwargs["sample_ratio"] == 0.25
    assert generate_csv.call_args.kwargs["size_threshold_px"] == 80
    assert generate_csv.call_args.kwargs["model_blob"] == "model-old.pt"
    assert generate_csv.call_args.kwargs["temperature"] == 1.25
    assert generate_csv.call_args.kwargs["resume_selected_tars"] == settings["selected_tars"]


def test_move_by_class_prompts_and_passes_selected_class(tab, tab_module, monkeypatch):
    tab._last_predictions = [
        {"predicted_label": "copepods"},
        {"predicted_label": "other"},
        {"predicted_label": "ERROR"},
    ]
    tab._last_account_url = "https://example.blob.core.windows.net"
    tab._last_run_dir = "results/run"
    get_item = MagicMock(return_value=("copepods", True))
    monkeypatch.setattr(tab_module.QInputDialog, "getItem", get_item)

    tab.move_by_class()

    assert get_item.call_args.args[3] == ["All classes (no filter)", "copepods", "other"]
    assert tab_module.MoveByClassWorker.call_args.kwargs["class_filter"] == "copepods"


def test_local_retention_dialog_allows_selecting_model_classes(tab_module):
    dialog = tab_module.ClassRetentionDialog(["copepod", "other"])
    dialog.class_list.item(0).setCheckState(tab_module.Qt.Checked)

    assert dialog.selected_classes() == ["copepod"]


def test_local_retention_selection_is_cleared_when_account_changes(tab, tab_module):
    tab._available_local_save_model_blob = "model.pt"
    tab._available_local_save_account_url = tab.account_url.text().strip()
    tab._available_local_save_classes = ["copepod"]
    tab._local_save_model_blob = "model.pt"
    tab._local_save_account_url = tab.account_url.text().strip()
    tab._local_save_classes = ["copepod"]

    tab.account_url.setText("https://other.blob.core.windows.net")

    assert tab._local_save_classes == []
    assert tab._available_local_save_classes == []


def test_run_application_passes_selected_retention_classes(
    tab, tab_module, monkeypatch, tmp_path,
):
    monkeypatch.setattr(tab, "_load_resume_settings_if_needed", lambda: None)
    monkeypatch.setattr(tab, "_targets_from_list", lambda: [{"container": "source"}])
    monkeypatch.setattr(tab, "_get_selected_model_blob", lambda: "model.pt")
    tab.output_dir.setText(str(tmp_path))
    tab.selected_tars = [("source", "sample.tar", "", 1.0, "inf")]
    tab._local_save_classes = ["copepod"]
    tab._local_save_model_blob = "model.pt"
    tab._local_save_account_url = tab.account_url.text().strip()

    tab.run_application()

    assert tab_module.ApplicationWorker.call_args.kwargs["local_save_classes"] == ["copepod"]
