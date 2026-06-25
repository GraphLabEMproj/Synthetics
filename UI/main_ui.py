import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Must be set before other Qt imports
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QUrl
from PySide6.QtQuickControls2 import QQuickStyle

from backend import Backend, SyntheticsImageProvider


def main():
    QQuickStyle.setStyle("Material")

    app = QGuiApplication(sys.argv)
    app.setApplicationName("Synthetics Generator UI")
    app.setOrganizationName("GraphLabEM")

    provider = SyntheticsImageProvider()
    backend = Backend(provider)

    engine = QQmlApplicationEngine()
    engine.addImageProvider("synthetics", provider)
    engine.rootContext().setContextProperty("backend", backend)

    qml_file = str(SCRIPT_DIR / "qml" / "main.qml")
    engine.load(QUrl.fromLocalFile(qml_file))

    if not engine.rootObjects():
        sys.exit(-1)

    # Load config AFTER QML is ready so ParameterPanel's Connections can catch the signal
    from PySide6.QtCore import QTimer
    default_config = str(PROJECT_ROOT / "setting.json")
    if os.path.exists(default_config):
        QTimer.singleShot(100, lambda: backend.loadConfig(default_config))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
