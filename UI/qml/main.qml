import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts
import QtQuick.Dialogs

ApplicationWindow {
    id: root
    visible: true
    width: 1460
    height: 880
    minimumWidth: 1000
    minimumHeight: 640
    title: "Synthetics Generator UI — " + configNameLabel.text

    // ── theme ────────────────────────────────────────────────────────────────
    property bool darkMode: true
    Material.theme:      darkMode ? Material.Dark  : Material.Light
    Material.primary:    darkMode ? "#3d5afe"      : "#1565c0"
    Material.accent:     darkMode ? "#40c4ff"      : "#0288d1"
    Material.background: darkMode ? "#1a1a2e"      : "#f5f5f5"
    Material.foreground: darkMode ? "#e0e0ff"      : "#1a1a2e"

    // ── state ────────────────────────────────────────────────────────────────
    property bool generationRunning: false

    // ── signals from backend ─────────────────────────────────────────────────
    Connections {
        target: backend
        function onErrorOccurred(msg)       { snackbar.show(msg) }
        function onConfigPathChanged(name)  { configNameLabel.text = name }
        function onGenerationBusy(b)        { root.generationRunning = b }
        function onLogMessage(msg) {
            logArea.append(msg)
            centerTabs.currentIndex = 1   // switch to log during generation
        }
        function onGenerationDone() {
            centerTabs.currentIndex = 0   // switch back to visualizer when done
        }
    }

    // ── file dialogs ─────────────────────────────────────────────────────────
    FileDialog {
        id: openDialog
        title: "Открыть конфиг"
        nameFilters: ["JSON files (*.json)", "All files (*)"]
        onAccepted: backend.loadConfig(selectedFile.toString())
    }

    FileDialog {
        id: saveDialog
        title: "Сохранить конфиг"
        fileMode: FileDialog.SaveFile
        nameFilters: ["JSON files (*.json)"]
        defaultSuffix: "json"
        onAccepted: backend.saveConfig(selectedFile.toString())
    }

    // ── toolbar ──────────────────────────────────────────────────────────────
    header: ToolBar {
        height: 52
        Material.background: root.darkMode ? "#12122a" : "#1565c0"

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            spacing: 8

            Label {
                text: "Synthetics Generator"
                font.pixelSize: 16
                font.weight: Font.Medium
                color: "white"
                opacity: 0.95
            }

            Item { Layout.fillWidth: true }

            RowLayout {
                spacing: 6
                Label {
                    text: "Конфиг:"
                    font.pixelSize: 11
                    color: "white"
                    opacity: 0.7
                }
                Rectangle {
                    height: 24
                    width: configNameLabel.implicitWidth + 16
                    radius: 12
                    color: Qt.rgba(1, 1, 1, 0.15)
                    Label {
                        id: configNameLabel
                        anchors.centerIn: parent
                        text: "setting.json"
                        font.pixelSize: 11
                        color: "white"
                        opacity: 0.9
                    }
                }
                ToolButton {
                    text: "Открыть"
                    font.pixelSize: 11
                    display: AbstractButton.TextOnly
                    onClicked: openDialog.open()
                    contentItem: Label {
                        text: parent.text
                        color: "#80d8ff"
                        font: parent.font
                        verticalAlignment: Text.AlignVCenter
                    }
                    background: Rectangle {
                        color: parent.hovered ? Qt.rgba(1,1,1,0.12) : "transparent"
                        radius: 4
                    }
                }
            }

            ToolButton {
                id: themeBtn
                implicitWidth: 40
                implicitHeight: 40
                onClicked: root.darkMode = !root.darkMode
                contentItem: Label {
                    text: root.darkMode ? "☀" : "🌙"
                    font.pixelSize: 18
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    radius: 20
                    color: themeBtn.hovered ? Qt.rgba(1,1,1,0.15) : "transparent"
                }
                ToolTip.visible: hovered
                ToolTip.text: root.darkMode ? "Светлая тема" : "Тёмная тема"
            }
        }
    }

    // ── main three-column layout ─────────────────────────────────────────────
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ════════════════════════════════════════════════════════════════════
        // LEFT PANEL — parameters + action buttons (380 px)
        // ════════════════════════════════════════════════════════════════════
        Rectangle {
            Layout.preferredWidth: 380
            Layout.minimumWidth:   320
            Layout.fillHeight: true
            color: root.darkMode ? "#1e1e32" : "#ffffff"

            ColumnLayout {
                anchors.fill: parent
                spacing: 0

                TabBar {
                    id: leftTabs
                    Layout.fillWidth: true
                    Material.accent: root.Material.accent

                    TabButton {
                        text: "Параметры"
                        font.pixelSize: 12
                        implicitHeight: 40
                    }
                    TabButton {
                        text: "Настройки"
                        font.pixelSize: 12
                        implicitHeight: 40
                    }
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: 1
                    color: root.darkMode ? "#2a2a48" : "#e0e0e0"
                }

                StackLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    currentIndex: leftTabs.currentIndex

                    ParameterPanel { id: paramPanel }
                    GenerationPanel {}
                }

                // ── vertical action buttons ───────────────────────────────
                Rectangle {
                    Layout.fillWidth: true
                    height: 1
                    color: root.darkMode ? "#2a2a48" : "#e0e0e0"
                }

                Column {
                    Layout.fillWidth: true
                    Layout.margins: 10
                    spacing: 6

                    // Сгенерировать основу
                    Button {
                        width: parent.width
                        text: root.generationRunning ? "Генерация…" : "Сгенерировать основу"
                        Material.background: root.darkMode ? "#1a3a5c" : "#bbdefb"
                        Material.foreground: root.darkMode ? "#64b5f6" : "#0d47a1"
                        font.pixelSize: 12
                        font.weight: Font.Medium
                        enabled: !root.generationRunning
                        onClicked: backend.generateStructure()
                        ToolTip.visible: hovered
                        ToolTip.text: "Заглушка ветки generator_3d — выполняет полную генерацию"
                    }

                    // Перерисовать
                    Button {
                        width: parent.width
                        text: root.generationRunning ? "Генерация…" : "Перерисовать"
                        Material.background: Material.Blue
                        Material.foreground: "white"
                        font.pixelSize: 12
                        font.weight: Font.Medium
                        enabled: !root.generationRunning
                        onClicked: backend.redraw()
                    }

                    // Сохранить ▼  (dropdown)
                    Button {
                        width: parent.width
                        text: "Сохранить ▾"
                        Material.background: root.darkMode ? "#2d2d4a" : "#e8eaf6"
                        font.pixelSize: 12
                        enabled: !root.generationRunning
                        onClicked: saveMenu.open()

                        Menu {
                            id: saveMenu
                            y: -height

                            MenuItem {
                                text: "Сохранить конфиг (тот же файл)"
                                onTriggered: backend.saveConfigInPlace()
                            }
                            MenuItem {
                                text: "Сохранить конфиг как…"
                                onTriggered: saveDialog.open()
                            }
                        }
                    }
                }
            }
        }

        // vertical divider
        Rectangle {
            Layout.fillHeight: true
            width: 1
            color: root.darkMode ? "#2a2a48" : "#e0e0e0"
        }

        // ════════════════════════════════════════════════════════════════════
        // CENTER PANEL — Visualizer / Log (fills remaining space)
        // ════════════════════════════════════════════════════════════════════
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            TabBar {
                id: centerTabs
                Layout.fillWidth: true
                Material.accent: root.Material.accent

                TabButton {
                    text: "Визуализация"
                    font.pixelSize: 12
                    implicitHeight: 40
                }
                TabButton {
                    text: "Журнал генерации"
                    font.pixelSize: 12
                    implicitHeight: 40
                }
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: root.darkMode ? "#2a2a48" : "#e0e0e0"
            }

            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: centerTabs.currentIndex

                // — Visualizer tab —
                Visualizer {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                // — Log tab —
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: root.darkMode ? "#0f0f1e" : "#fafafa"

                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: 8
                        clip: true

                        TextArea {
                            id: logArea
                            readOnly: true
                            wrapMode: TextArea.Wrap
                            font.family: "Consolas, monospace"
                            font.pixelSize: 11
                            color: root.darkMode ? "#a9b1d6" : "#1a1a2e"
                            background: null
                            text: "[ Журнал генерации ]\n"

                            function append(msg) {
                                logArea.text += msg + "\n"
                                // auto-scroll to bottom
                                logArea.cursorPosition = logArea.length
                            }
                        }
                    }

                    // clear button
                    ToolButton {
                        anchors.top: parent.top
                        anchors.right: parent.right
                        anchors.margins: 4
                        implicitWidth: 28
                        implicitHeight: 28
                        text: "✕"
                        font.pixelSize: 11
                        onClicked: logArea.text = "[ Журнал генерации ]\n"
                        ToolTip.visible: hovered
                        ToolTip.text: "Очистить журнал"
                        contentItem: Label {
                            text: parent.text
                            color: root.darkMode ? "#6a6a8a" : "#888"
                            font: parent.font
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                        }
                        background: Rectangle {
                            radius: 4
                            color: parent.hovered ? Qt.rgba(1,1,1,0.1) : "transparent"
                        }
                    }
                }
            }
        }

        // vertical divider
        Rectangle {
            Layout.fillHeight: true
            width: 1
            color: root.darkMode ? "#2a2a48" : "#e0e0e0"
        }

        // ════════════════════════════════════════════════════════════════════
        // RIGHT PANEL — Histograms (380 px)
        // ════════════════════════════════════════════════════════════════════
        HistogramPanel {
            Layout.preferredWidth: 380
            Layout.minimumWidth:   300
            Layout.fillHeight: true
            darkMode: root.darkMode
        }
    }

    // ── snackbar ──────────────────────────────────────────────────────────────
    Snackbar { id: snackbar }
}
