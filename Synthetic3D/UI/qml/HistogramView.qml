import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Item {
    id: root

    property string histCounter: "0"

    Connections {
        target: backend
        function onHistogramReady(counter) { root.histCounter = counter }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 8

        // ── histogram image ─────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: Material.theme === Material.Dark ? "#12121f" : "#f5f5f5"
            radius: 6

            Image {
                id: histImg
                anchors.fill: parent
                anchors.margins: 4
                fillMode: Image.PreserveAspectFit
                smooth: true
                asynchronous: true
                cache: false

                source: root.histCounter === "0"
                    ? ""
                    : "image://synthetics/histogram/" + root.histCounter

                Label {
                    visible: histImg.status !== Image.Ready
                    anchors.centerIn: parent
                    text: "Нажмите «Построить гистограммы»"
                    opacity: 0.35
                    font.pixelSize: 13
                    color: Material.theme === Material.Dark ? "white" : "#333"
                    horizontalAlignment: Text.AlignHCenter
                }
            }
        }

        // ── build button ────────────────────────────────────────────────────
        Button {
            Layout.alignment: Qt.AlignHCenter
            text: "Построить гистограммы"
            Material.background: Material.Blue
            Material.foreground: "white"
            font.pixelSize: 12
            onClicked: backend.buildHistograms()
        }
    }
}
