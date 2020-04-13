import QtQuick 2.14
import QtQuick.Controls 2.14
import QtQuick.Layouts 1.14

Rectangle {
    border.color: "black"
    border.width: 1
    property real realHeight: controlRow.height + 12 + (errorArea.visible ? errorArea.height : -4)
    RowLayout {
        id: controlRow
        anchors.margins: 4
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        spacing: 6

        SpecialButton {
            id: runButton
            disabledImgColor: "#629d62"
            baseImgColor: "#3bc43b"
            toggledImgColor: baseImgColor
            smooth: true
            height: 26
            iconSize: 24
            Layout.preferredWidth: implicitWidth
            Layout.preferredHeight: implicitHeight
            sourceIcon: "res/play.png"
            enabled: input.text != ""
            onClicked: {
                input.hasError = !input.hasError
            }
        }
        SpecialTextEdit {
            property bool hasError: false
            Layout.minimumWidth: 100
            Layout.fillWidth: true
            Layout.preferredHeight: 26
            id: input
            placeholderText: "code expression"
            text: ""
            backRectBorder.color: hasError ? "#f44c4c" : (input.activeFocus ? "#29B6F6" : "#9E9E9E")
            onAccepted: {
                hasError = !hasError
            }
        }

        SpecialButton {
            id: delButton

            disabledImgColor: "#9d6262"
            baseImgColor: "#f44c4c"
            toggledImgColor: baseImgColor
            onClicked: {
                input.clear()
            }

            smooth: true
            height: 26
            iconSize: 24
            Layout.preferredWidth: implicitWidth
            Layout.preferredHeight: implicitHeight
            sourceIcon: "res/del.png"
        }
    }
    TextArea {
        id: errorArea
        visible: input.hasError
        anchors.top: controlRow.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 4
        bottomInset: 0
        topInset: 0
        padding: 6
        background: Rectangle {
            color: "transparent"
            border.color: "red"
            border.width: 1
            radius: 6
        }

        readOnly: true
        selectByMouse: true
        text: "Traceback (most recent call last):\n
File \"<stdin>\", line 1, in <module>\n
ZeroDivisionError: division by zero\n
File \"<stdin>\", line 1, in <module>\n
ZeroDivisionError: division by zero\n
File \"<stdin>\", line 1, in <module>\n
ZeroDivisionError: division by zero\n
File \"<stdin>\", line 1, in <module>\n
ZeroDivisionError: division by zero\n
File \"<stdin>\", line 1, in <module>\n
ZeroDivisionError: division by zero\n
File \"<stdin>\", line 1, in <module>\n
ZeroDivisionError: division by zero\n
File \"<stdin>\", line 1, in <module>\n
ZeroDivisionError: division by zero\n
"
    }
}
