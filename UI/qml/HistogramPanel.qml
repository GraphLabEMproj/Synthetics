import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Rectangle {
    id: root

    property bool   darkMode: true
    property int    maxHists: 5
    property var    histKeys: []     // list of provider keys, newest first

    color: darkMode ? "#1a1a30" : "#f0f0f8"

    Connections {
        target: backend
        function onHistListChanged(jsonStr) {
            root.histKeys = JSON.parse(jsonStr)
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // ── header ────────────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            height: 40
            color: root.darkMode ? "#16162a" : "#e8e8f8"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 8
                spacing: 6

                Label {
                    text: "Гистограммы"
                    font.pixelSize: 12
                    font.weight: Font.Medium
                    opacity: 0.9
                    Layout.fillWidth: true
                }

                Label {
                    text: "Слотов:"
                    font.pixelSize: 11
                    opacity: 0.6
                }

                SpinBox {
                    id: slotSpin
                    from: 1; to: 20; value: root.maxHists
                    implicitWidth: 72
                    implicitHeight: 28
                    font.pixelSize: 11
                    onValueChanged: {
                        root.maxHists = value
                        backend.setMaxHistograms(value)
                    }
                }

                // "Добавить" button — append newest histogram
                Button {
                    text: "＋"
                    implicitWidth: 28
                    implicitHeight: 28
                    font.pixelSize: 14
                    Material.background: Material.Blue
                    Material.foreground: "white"
                    ToolTip.visible: hovered
                    ToolTip.text: "Добавить гистограмму текущего слоя"
                    onClicked: backend.buildHistograms()
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: root.darkMode ? "#2a2a48" : "#d0d0e8"
        }

        // ── histogram list ────────────────────────────────────────────────
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            Column {
                id: histColumn
                width: root.width
                spacing: 0

                Repeater {
                    model: root.maxHists
                    delegate: HistSlot {
                        width: root.width
                        slotIndex: index
                        providerKey: index < root.histKeys.length ? root.histKeys[index] : ""
                        darkMode: root.darkMode
                    }
                }

                // empty-state hint
                Label {
                    visible: root.histKeys.length === 0
                    width: root.width
                    height: 80
                    text: "Нажмите «＋» чтобы\nдобавить гистограмму"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                    font.pixelSize: 11
                    opacity: 0.35
                    color: root.darkMode ? "white" : "black"
                    wrapMode: Text.WordWrap
                }
            }
        }
    }
}
