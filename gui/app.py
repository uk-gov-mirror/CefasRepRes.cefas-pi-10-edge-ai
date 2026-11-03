import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.application_validation_tab import ApplicationValidationTab
from gui.edge_tab import EdgeAITab
from gui.gps_locations_tab import GPSLocationsTab
from gui.train_tab import ModelTrainingTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plankton Imager – Edge AI")
        self.resize(1000, 700)

        tabs = QTabWidget()
        tabs.addTab(EdgeAITab(), "Edge AI")
        tabs.addTab(ModelTrainingTab(), "Model Training")
        tabs.addTab(ApplicationValidationTab(), "Apply trained model and validate")
        tabs.addTab(GPSLocationsTab(), "Data viewer")

        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()