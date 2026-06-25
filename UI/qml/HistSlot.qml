import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Rectangle {
    id: root

    property int    slotIndex:   0
    property string providerKey: ""
    property bool   darkMode:    true

    height: providerKey !== "" ? 190 : 52
    color: "transparent"

    // separator
    Rectangle {
        anchors.bottom: parent.bottom
        anchors.left:   parent.left
        anchors.right:  parent.right
        height: 1
        color: root.darkMode ? "#25253a" : "#d8d8ec"
    }

    // ── filled slot: histogram image ─────────────────────────────────────
    ColumnLayout {
        visible: root.providerKey !== ""
        anchors.fill: parent
        anchors.margins: 6
        spacing: 4

        // slot label
        Label {
            text: "#" + (root.slotIndex + 1)
            font.pixelSize: 9
            opacity: 0.45
            color: root.darkMode ? "white" : "black"
        }

        Image {
            Layout.fillWidth: true
            Layout.fillHeight: true
            fillMode: Image.PreserveAspectFit
            smooth: true
            asynchronous: true
            cache: false
            // providerKey doubles as cache-buster suffix after "/"
            source: root.providerKey !== ""
                ? "image://synthetics/" + root.providerKey + "/" + root.providerKey
                : ""
        }
    }

    // ── empty slot ────────────────────────────────────────────────────────
    Item {
        visible: root.providerKey === ""
        anchors.fill: parent

        Label {
            anchors.centerIn: parent
            text: "#" + (root.slotIndex + 1) + " / пусто"
            font.pixelSize: 10
            opacity: 0.25
            color: root.darkMode ? "white" : "black"
        }
    }
}
