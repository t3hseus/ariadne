import QtQuick 2.12
import QtQuick.Controls 2.12
import QtGraphicalEffects 1.0

AbstractButton {
    property bool toggleEnabled: false
    property bool tooltipEnabled: true
    property bool disableAnimations: false

    property string tooltipText: "IDS_NO_TOOLTIP_INFO"
    property string sourceIcon: ""

    property int iconSize: controlItem.height
    property int animDuration: toggleEnabled ? 200 : 50

    property int startAngle: 0
    property int stopAngle: 0
    property int fontSize: 10
    property int textLeftMargin: 4
    property alias back: controlItem

    property bool smoothImage: false
    property color disabledControlColor: "#a6a6a6"
    property color disabledImgColor: "#e9e9e9"
    property color baseControlColor: "#757575"
    property color baseImgColor: "white"
    property color hoveredImgControlColor: baseImgColor
    property color hoveredControlColor: "#9E9E9E"
    property color hoveredToggledControlColor: "#616161"
    property color toggledControlColor: "#424242"
    property color toggledImgColor: "#B3E5FC"

    ToolTip.text: tooltipText
    focusPolicy: Qt.StrongFocus
    ToolTip.delay: 1500
    ToolTip.visible: tooltipEnabled && hovered

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
            Behavior on color {
                enabled: !disableAnimations
                ColorAnimation {
                    duration: animDuration
                    alwaysRunToEnd: true
                }
            }
        }
        Item {
            anchors.verticalCenter: parent.verticalCenter
            anchors.right: parent.right
            anchors.rightMargin: text === "" ? 0 : 4
            anchors.centerIn: text === "" ? parent : undefined
            width: iconSize
            height: iconSize
            Image {
                id: edImg
                mipmap: smoothImage
                source: sourceIcon
                anchors.fill: parent
            }
            ColorOverlay {
                anchors.fill: edImg
                source: edImg
                color: baseImgColor
                id: imgOver
                Behavior on color {
                    enabled: !disableAnimations
                    ColorAnimation {
                        duration: animDuration
                    }
                }
            }

            transform: Rotation {
                origin.x: edImg.width / 2
                origin.y: edImg.width / 2
                angle: checked ? startAngle : stopAngle
                Behavior on angle {
                    enabled: !disableAnimations
                    SmoothedAnimation {
                        duration: animDuration
                    }
                }
            }
        }
    }

    background: Rectangle {
        id: controlItem
        color: baseControlColor
        radius: 4
        implicitHeight: 16
        width: (text === "" ? -8 : textLeftMargin) + edText.contentWidth + 4 + edImg.width + 4
        Behavior on color {
            enabled: !disableAnimations
            ColorAnimation {
                duration: animDuration
            }
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
                target: imgOver
                color: disabledImgColor
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
                target: imgOver
                color: baseImgColor
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
            PropertyChanges {
                target: imgOver
                color: hoveredImgControlColor
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
                target: imgOver
                color: toggledImgColor
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
