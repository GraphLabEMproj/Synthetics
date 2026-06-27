import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

Item {
    id: root

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true

        ColumnLayout {
            width: parent.width
            spacing: 0

            // ── mode ──────────────────────────────────────────────────────────
            SectionHeader { title: "Режим генерации" }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.margins: 12
                spacing: 4

                RadioButton {
                    id: rbSingle
                    text: "Один слой (быстрый превью)"
                    checked: true
                    font.pixelSize: 12
                    onCheckedChanged: if (checked) backend.setMode("single")
                }
                RadioButton {
                    id: rbDataset
                    text: "Полный датасет"
                    font.pixelSize: 12
                    onCheckedChanged: if (checked) backend.setMode("dataset")
                }
            }

            // ── organelle counts ───────────────────────────────────────────────
            SectionHeader { title: "Количество органелл" }

            Column {
                Layout.fillWidth: true
                Layout.leftMargin: 12
                Layout.rightMargin: 12
                Layout.topMargin: 6
                Layout.bottomMargin: 6
                spacing: 6

                CountRow { label: "PSD";          genKey: "count_psd";      defVal: 3; maxVal: 10 }
                CountRow { label: "Аксон";        genKey: "count_axon";     defVal: 1; maxVal: 5 }
                CountRow { label: "Везикулы";     genKey: "count_vesicles"; defVal: 3; maxVal: 10 }
                CountRow { label: "Митохондрии";  genKey: "count_mito";     defVal: 3; maxVal: 10 }
                CountRow { label: "Спам";         genKey: "count_spam";     defVal: 5; maxVal: 20 }
            }

            // ── image size ─────────────────────────────────────────────────────
            SectionHeader { title: "Размер изображения" }

            RowLayout {
                Layout.fillWidth: true
                Layout.margins: 12
                spacing: 8

                Label { text: "Размер (px):"; font.pixelSize: 12; opacity: 0.87 }

                ComboBox {
                    model: [128, 256, 512]
                    currentIndex: 1
                    font.pixelSize: 12
                    implicitWidth: 100
                    onCurrentValueChanged: backend.setGenParam("size", currentValue)
                }
            }

            // ── dataset settings ───────────────────────────────────────────────
            SectionHeader {
                title: "Настройки датасета"
                visible: rbDataset.checked
            }

            ColumnLayout {
                visible: rbDataset.checked
                Layout.fillWidth: true
                Layout.margins: 12
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: "Кол-во кадров:"
                        font.pixelSize: 12; opacity: 0.87
                        Layout.preferredWidth: 140
                    }
                    SpinBox {
                        from: 1; to: 10000; value: 10
                        font.pixelSize: 12
                        Layout.preferredWidth: 130
                        onValueChanged: backend.setGenParam("count", value)
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: "Стартовый индекс:"
                        font.pixelSize: 12; opacity: 0.87
                        Layout.preferredWidth: 140
                    }
                    SpinBox {
                        from: 0; to: 100000; value: 0
                        font.pixelSize: 12
                        Layout.preferredWidth: 130
                        onValueChanged: backend.setGenParam("start_index", value)
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label {
                        text: "Папка сохранения:"
                        font.pixelSize: 12; opacity: 0.87
                        Layout.preferredWidth: 140
                    }
                    TextField {
                        Layout.fillWidth: true
                        text: "dataset/ui_dataset"
                        font.pixelSize: 11
                        padding: 4
                        onTextChanged: backend.setGenParam("dir_save", text)
                    }
                }
            }

            Item { height: 16; Layout.fillWidth: true }
        }
    }

    // ── CountRow: label + [−] value [+] in one line ──────────────────────────
    component CountRow: RowLayout {
        id: crow
        property string label:  ""
        property string genKey: ""
        property int    defVal: 1
        property int    maxVal: 10
        property int    val:    defVal

        width: parent ? parent.width : 300
        spacing: 0

        Label {
            text: crow.label
            font.pixelSize: 12
            opacity: 0.87
            Layout.fillWidth: true
            elide: Text.ElideRight
        }

        // [−] button
        RoundButton {
            implicitWidth: 30; implicitHeight: 30
            radius: 4
            text: "−"
            font.pixelSize: 16
            flat: true
            enabled: crow.val > 0
            onClicked: {
                crow.val = Math.max(0, crow.val - 1)
                backend.setGenParam(crow.genKey, crow.val)
            }
            contentItem: Label {
                text: parent.text
                font: parent.font
                color: parent.enabled
                    ? (Material.theme === Material.Dark ? "#e0e0ff" : "#1a1a2e")
                    : (Material.theme === Material.Dark ? "#44445e" : "#bbb")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            background: Rectangle {
                radius: 4
                color: parent.hovered
                    ? (Material.theme === Material.Dark ? Qt.rgba(1,1,1,0.10) : Qt.rgba(0,0,0,0.06))
                    : "transparent"
                border.color: Material.theme === Material.Dark ? "#3a3a5a" : "#c0c0d8"
                border.width: 1
            }
        }

        // value label
        Rectangle {
            implicitWidth: 44; implicitHeight: 30
            radius: 4
            color: Material.theme === Material.Dark ? "#22223a" : "#f0f0f8"
            border.color: Material.theme === Material.Dark ? "#3a3a5a" : "#c0c0d8"
            border.width: 1

            Label {
                anchors.centerIn: parent
                text: crow.val
                font.pixelSize: 13
                font.weight: Font.Medium
                color: Material.theme === Material.Dark ? "#e0e0ff" : "#1a1a2e"
            }
        }

        // [+] button
        RoundButton {
            implicitWidth: 30; implicitHeight: 30
            radius: 4
            text: "+"
            font.pixelSize: 16
            flat: true
            enabled: crow.val < crow.maxVal
            onClicked: {
                crow.val = Math.min(crow.maxVal, crow.val + 1)
                backend.setGenParam(crow.genKey, crow.val)
            }
            contentItem: Label {
                text: parent.text
                font: parent.font
                color: parent.enabled
                    ? (Material.theme === Material.Dark ? "#e0e0ff" : "#1a1a2e")
                    : (Material.theme === Material.Dark ? "#44445e" : "#bbb")
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            background: Rectangle {
                radius: 4
                color: parent.hovered
                    ? (Material.theme === Material.Dark ? Qt.rgba(1,1,1,0.10) : Qt.rgba(0,0,0,0.06))
                    : "transparent"
                border.color: Material.theme === Material.Dark ? "#3a3a5a" : "#c0c0d8"
                border.width: 1
            }
        }
    }

    // ── section header ────────────────────────────────────────────────────────
    component SectionHeader: Rectangle {
        property string title: ""
        Layout.fillWidth: true
        height: 30
        color: Material.theme === Material.Dark ? "#2d2d4a" : "#e8eaf6"

        Label {
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.leftMargin: 12
            text: parent.title
            font.pixelSize: 11
            font.weight: Font.Medium
            font.letterSpacing: 0.8
            opacity: 0.72
            color: Material.theme === Material.Dark ? "#7b8cde" : "#3949ab"
        }

        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1
            color: Material.theme === Material.Dark ? "#3a3a58" : "#c5cae9"
        }
    }
}
