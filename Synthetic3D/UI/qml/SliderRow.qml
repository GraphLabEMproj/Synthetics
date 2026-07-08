import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

// Single parameter row: [Label] [Slider ─────●─────] [TextField]
Item {
    id: root

    property string paramKey: ""
    property string label: ""
    property real   value: 0
    property real   minVal: 0
    property real   maxVal: 255
    property real   step: 1
    property int    decimals: 1

    signal valueEdited(string key, real newValue)

    implicitHeight: 36
    implicitWidth: parent ? parent.width : 400

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 8
        anchors.rightMargin: 8
        spacing: 6

        Label {
            id: lbl
            text: root.label
            Layout.preferredWidth: 160
            Layout.minimumWidth: 110
            elide: Text.ElideRight
            font.pixelSize: 11
            opacity: 0.87

            MouseArea { id: lblHover; anchors.fill: parent; hoverEnabled: true }
            ToolTip.visible: lblHover.containsMouse && lbl.truncated
            ToolTip.text: root.label
            ToolTip.delay: 600
        }

        Slider {
            id: slider
            Layout.fillWidth: true
            from: root.minVal
            to: root.maxVal
            stepSize: root.step
            // declarative binding — only changed by root.value externally
            value: root.value

            onMoved: {
                root.value = value
                root.valueEdited(root.paramKey, value)
            }

            background: Rectangle {
                x: slider.leftPadding
                y: slider.topPadding + slider.availableHeight / 2 - height / 2
                implicitWidth: 200; implicitHeight: 4
                width: slider.availableWidth; height: implicitHeight
                radius: 2
                color: Material.theme === Material.Dark ? "#3d3d5c" : "#c0c0d0"

                Rectangle {
                    width: slider.visualPosition * parent.width
                    height: parent.height
                    color: Material.accent
                    radius: 2
                }
            }

            handle: Rectangle {
                x: slider.leftPadding + slider.visualPosition * (slider.availableWidth - width)
                y: slider.topPadding + slider.availableHeight / 2 - height / 2
                implicitWidth: 16; implicitHeight: 16
                radius: 8
                color: slider.pressed ? Qt.lighter(Material.accent, 1.2) : Material.accent
                border.color: Qt.darker(Material.accent, 1.3)
                border.width: 1
            }
        }

        TextField {
            id: field
            Layout.preferredWidth: 64
            // only update text when not being edited
            text: root.value.toFixed(root.decimals)
            font.pixelSize: 11
            horizontalAlignment: TextInput.AlignRight
            padding: 3; leftPadding: 5; rightPadding: 5

            background: Rectangle {
                radius: 4
                color: Material.theme === Material.Dark ? "#2a2a3e" : "#f5f5f5"
                border.color: field.activeFocus
                    ? Material.accent
                    : (Material.theme === Material.Dark ? "#44445e" : "#c0c0c0")
                border.width: field.activeFocus ? 2 : 1
            }

            validator: DoubleValidator {
                bottom: root.minVal; top: root.maxVal
                notation: DoubleValidator.StandardNotation
            }

            onEditingFinished: {
                var v = parseFloat(text)
                if (!isNaN(v)) {
                    v = Math.max(root.minVal, Math.min(root.maxVal, v))
                    if (root.step > 0)
                        v = Math.round(v / root.step) * root.step
                    // only set root.value; the slider binding 'value: root.value' handles sync
                    root.value = v
                    text = v.toFixed(root.decimals)
                    root.valueEdited(root.paramKey, v)
                }
            }
        }
    }

    // Sync field text when root.value is changed externally (not via this component)
    // Do NOT set slider.value here — it would break the declarative binding above
    onValueChanged: {
        if (!field.activeFocus)
            field.text = root.value.toFixed(root.decimals)
    }
}
