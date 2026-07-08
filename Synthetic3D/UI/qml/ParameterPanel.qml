import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Scrollable panel with auto-generated slider groups from JSON config
Item {
    id: root

    property var groups: []   // parsed from backend.paramsChanged JSON

    Connections {
        target: backend
        function onParamsChanged(jsonStr) {
            root.groups = JSON.parse(jsonStr)
        }
    }

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true

        Column {
            id: content
            width: parent.width
            spacing: 0

            Repeater {
                model: root.groups

                delegate: Column {
                    id: groupCol
                    width: content.width
                    spacing: 0

                    property var groupData: modelData

                    // ── group header ────────────────────────────────────────
                    Rectangle {
                        width: groupCol.width
                        height: 32
                        color: Material.theme === Material.Dark ? "#2d2d4a" : "#e8eaf6"

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 12
                            anchors.rightMargin: 8

                            Label {
                                text: groupData.name
                                font.pixelSize: 11
                                font.weight: Font.Medium
                                font.letterSpacing: 0.8
                                opacity: 0.72
                                color: Material.theme === Material.Dark ? "#7b8cde" : "#3949ab"
                                Layout.fillWidth: true
                            }

                            Label {
                                text: groupData.params.length + " param"
                                font.pixelSize: 9
                                opacity: 0.45
                            }
                        }

                        Rectangle {
                            anchors.bottom: parent.bottom
                            width: parent.width
                            height: 1
                            color: Material.theme === Material.Dark ? "#3a3a58" : "#c5cae9"
                        }
                    }

                    // ── sliders ─────────────────────────────────────────────
                    Repeater {
                        model: groupData.params

                        delegate: Column {
                            width: groupCol.width
                            spacing: 0

                            SliderRow {
                                id: slRow
                                width: groupCol.width
                                paramKey: modelData.key
                                label:    modelData.label
                                value:    modelData.value
                                minVal:   modelData.min
                                maxVal:   modelData.max
                                step:     modelData.step
                                decimals: modelData.decimals

                                onValueEdited: (key, val) => backend.updateParam(key, val)
                            }

                            Rectangle {
                                width: groupCol.width
                                height: 1
                                color: Material.theme === Material.Dark ? "#28283e" : "#eeeeee"
                            }
                        }
                    }

                    // spacing between groups
                    Item { width: 1; height: 4 }
                }
            }

            // bottom padding
            Item { width: 1; height: 16 }
        }
    }
}
