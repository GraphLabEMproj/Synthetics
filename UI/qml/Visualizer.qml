import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Item {
    id: root

    property string imageCounter: "0"
    property int    maskIndex:    0     // 0=Оригинал … 6=Везикулы
    property bool   busy:         false

    // Provider key for the selected mask/layer
    readonly property var maskKeys: [
        "layer", "mask_psd", "mask_axon", "mask_mem",
        "mask_mito", "mask_mito_b", "mask_ves"
    ]
    property string currentKey: root.maskKeys[root.maskIndex] ?? "layer"

    Connections {
        target: backend
        function onImageReady(counter)  { root.imageCounter = counter }
        function onGenerationBusy(b)    { root.busy = b }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        // ── image area ────────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Material.theme === Material.Dark ? "#12121f" : "#1a1a2e"
            radius: 6
            clip: true

            Image {
                id: img
                anchors.fill: parent
                anchors.margins: 4
                fillMode: Image.PreserveAspectFit
                smooth: true
                asynchronous: true
                cache: false

                source: root.imageCounter === "0"
                    ? ""
                    : "image://synthetics/" + root.currentKey + "/" + root.imageCounter

                Label {
                    visible: img.status !== Image.Ready && !root.busy
                    anchors.centerIn: parent
                    text: "Нажмите «Перерисовать»\nдля генерации изображения"
                    horizontalAlignment: Text.AlignHCenter
                    opacity: 0.35
                    font.pixelSize: 13
                    color: "white"
                }

                BusyIndicator {
                    visible: root.busy
                    anchors.centerIn: parent
                    running: root.busy
                    Material.accent: Material.Blue
                }
            }

            // resolution badge
            Label {
                visible: img.status === Image.Ready
                anchors.bottom: parent.bottom
                anchors.right:  parent.right
                anchors.margins: 6
                text: img.sourceSize.width + "×" + img.sourceSize.height
                font.pixelSize: 9
                opacity: 0.45
                color: "white"
            }
        }

        // ── controls row ──────────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 4
            Layout.rightMargin: 4
            spacing: 8

            // Layer slider — only shown for 3D stacks (to > 0).
            // Current generator produces 2D images, so this stays hidden.
            Label {
                visible: layerSlider.to > 0
                text: "Слой:"
                font.pixelSize: 11
                opacity: 0.75
            }

            Slider {
                id: layerSlider
                visible: to > 0
                Layout.fillWidth: true
                from: 0; to: 0; stepSize: 1
                ToolTip.visible: pressed
                ToolTip.text: "Слой " + Math.round(value) + " / " + (to + 1)
            }

            Label {
                visible: layerSlider.to > 0
                text: Math.round(layerSlider.value) + " / " + (layerSlider.to + 1)
                font.pixelSize: 11
                opacity: 0.75
                Layout.preferredWidth: 46
            }

            // mask selector ComboBox — 7 options
            Label {
                text: "Вид:"
                font.pixelSize: 11
                opacity: 0.75
            }

            ComboBox {
                id: maskCombo
                font.pixelSize: 11
                implicitHeight: 32
                model: ["Оригинал", "PSD", "Аксон", "Мембраны",
                        "Митохондрии", "Границы мито", "Везикулы"]
                onCurrentIndexChanged: root.maskIndex = currentIndex

                background: Rectangle {
                    color: Material.theme === Material.Dark
                        ? Qt.rgba(1,1,1,0.08) : Qt.rgba(0,0,0,0.06)
                    radius: 4
                    border.color: Qt.rgba(1,1,1,0.15)
                    border.width: 1
                }

                contentItem: Label {
                    leftPadding: 8
                    rightPadding: maskCombo.indicator.width + 4
                    text: maskCombo.displayText
                    font: maskCombo.font
                    color: Material.foreground
                    verticalAlignment: Text.AlignVCenter
                    elide: Text.ElideRight
                }
            }
        }

        // ── dataset progress bar ──────────────────────────────────────────
        RowLayout {
            id: progressRow
            Layout.fillWidth: true
            visible: false
            spacing: 8

            Label { id: progressLabel; text: "0 / 0"; font.pixelSize: 11; opacity: 0.75 }
            ProgressBar {
                id: progressBar
                Layout.fillWidth: true
                from: 0; to: 1; value: 0
            }
        }

        Connections {
            target: backend
            function onGenerationProgress(current, total) {
                progressRow.visible = true
                progressLabel.text  = current + " / " + total
                progressBar.value   = current / total
            }
            function onGenerationDone() {
                progressRow.visible = false
                progressBar.value   = 0
            }
        }
    }
}
