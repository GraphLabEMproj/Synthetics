import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Bottom-anchored error/info snackbar (Material Design)
Item {
    id: root
    anchors.fill: parent
    anchors.bottomMargin: 16

    function show(message, durationMs) {
        msgLabel.text = message
        bar.opacity = 1
        hideTimer.interval = durationMs || 5000
        hideTimer.restart()
    }

    Timer {
        id: hideTimer
        onTriggered: bar.opacity = 0
    }

    Rectangle {
        id: bar
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        width: Math.min(parent.width - 32, msgLabel.implicitWidth + 48)
        height: 44
        radius: 6
        color: "#323248"
        opacity: 0
        z: 999

        Behavior on opacity { NumberAnimation { duration: 200 } }

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 16
            anchors.rightMargin: 8
            spacing: 8

            Label {
                id: msgLabel
                Layout.fillWidth: true
                font.pixelSize: 12
                color: "#e0e0ff"
                wrapMode: Text.WordWrap
                maximumLineCount: 3
                elide: Text.ElideRight
            }

            ToolButton {
                text: "✕"
                implicitWidth: 32
                implicitHeight: 32
                font.pixelSize: 11
                contentItem: Label {
                    text: parent.text
                    color: "#a0a0cc"
                    font: parent.font
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Item {}
                onClicked: { hideTimer.stop(); bar.opacity = 0 }
            }
        }
    }
}
