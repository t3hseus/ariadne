import QtQuick 2.12
import QtQuick.Controls 2.12
import QtGraphicalEffects 1.0

AbstractButton {
    property bool toggleEnabled: false

    property alias progressHelper: progressRect

    property string tooltipText: "IDS_NO_TOOLTIP_INFO"
    property alias controlTextItem: edText

    property int animDuration: toggleEnabled ? 200 : 50

    property int fontSize: 10
    property int textLeftMargin: 4

    property alias back: controlItem
    property color disabledControlColor: "#a6a6a6"
    property color disabledImgColor: "#e9e9e9"
    property color baseControlColor: "#757575"
    property color baseImgColor: "white"
    property color hoveredImgControlColor: baseImgColor
    property color hoveredControlColor: "#9E9E9E"
    property color hoveredToggledControlColor: "#616161"
    property color toggledControlColor: "#424242"
    property color toggledImgColor: "#B3E5FC"
    property color progressColor: "red"

    hoverEnabled: true
    checkable: toggleEnabled
    leftPadding: 0
    rightPadding: 0
    id: root
    implicitHeight: background.height
    implicitWidth: background.width

    Keys.onPressed: {

    }
    contentItem: Item {
        Text {
            id: edText
            text: root.text
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: parent.left
            anchors.leftMargin: textLeftMargin
            color: baseImgColor
            font.pointSize: fontSize
            elide: Text.ElideRight
        }
    }

    background: Rectangle {
        id: controlItem
        color: baseControlColor
        radius: 4
        implicitHeight: 16
        width: textLeftMargin + edText.contentWidth + 4 + controlItem.height + 4
        border.color: toggledImgColor
        border.width: 0
        Rectangle {
            id: progressRect
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            color: progressColor
            width: 0
            height: parent.height
            radius: parent.radius
        }
    }
    states: [
        State {
            name: "disabled"
            PropertyChanges {
                target: controlItem
                color: disabledControlColor
            }
            PropertyChanges {
                target: edText
                color: disabledImgColor
            }
        },
        State {
            name: "base"
            PropertyChanges {
                target: controlItem
                color: baseControlColor
            }
            PropertyChanges {
                target: edText
                color: baseImgColor
            }
        },
        State {
            name: "hovered"
            extend: "base"
            PropertyChanges {
                target: controlItem
                color: hoveredControlColor
            }
        },
        State {
            extend: "toggled"
            name: "hoveredToggled"
            PropertyChanges {
                target: controlItem
                color: hoveredToggledControlColor
            }
        },
        State {
            name: "toggled"
            PropertyChanges {
                target: controlItem
                color: toggledControlColor
            }
            PropertyChanges {
                target: edText
                color: toggledImgColor
            }
        },
        State {
            name: "pressed"
            extend: "toggled"
        }
    ]

    onEnabledChanged: {
        if (enabled) {
            if (hovered)
                if (checked)
                    state = "hoveredToggled"
                else
                    state = "hovered"
            else
                state = toggleEnabled && checked ? "toggled" : "base"
        } else {
            state = "disabled"
        }
    }

    state: !enabled ? "disabled" : (toggleEnabled
                                    && checked ? "toggled" : "base")
    function processChecked() {
        if (!enabled) {
            state = "disabled"
            return
        }
        if (checked)
            if (hovered)
                state = "hoveredToggled"
            else
                state = "toggled"
        else
            state = hovered ? "hovered" : "base"
    }

    onCheckedChanged: {
        processChecked()
    }
    onPressed: {
        if (!enabled) {
            state = "disabled"
            return
        }
        if (pressed)
            state = "pressed"
    }
    onReleased: {
        if (!enabled) {
            state = "disabled"
            return
        }
        if (hovered)
            if (checked)
                state = "hoveredToggled"
            else
                state = "hovered"
        else
            state = toggleEnabled && checked ? "toggled" : "base"
    }
    onCanceled: {
        if (!enabled) {
            state = "disabled"
            return
        }
        state = toggleEnabled && checked ? "toggled" : "base"
    }
    function processHovered() {
        if (!enabled) {
            state = "disabled"
            return
        }
        if (toggleEnabled) {
            if (hovered)
                if (checked)
                    state = "hoveredToggled"
                else
                    state = "hovered"
            else if (checked)
                state = "toggled"
            else if (!pressed)
                state = "base"
        } else {
            if (hovered)
                state = "hovered"
            else if (!pressed)
                state = "base"
        }
    }

    onHoveredChanged: {
        processHovered()
    }
}
