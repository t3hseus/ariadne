import QtQuick 2.14
import QtQuick.Controls 2.14

TextField {

    property alias fontSize: valueHolder.font.pointSize
    property alias backRectBorder: backRect.border

    id: valueHolder
    selectByMouse: true

    horizontalAlignment: TextInput.AlignLeft
    verticalAlignment: TextInput.AlignVCenter
    leftPadding: 4
    bottomPadding: 1
    topPadding: 1
    rightPadding: 3
    font.pointSize: 12

    placeholderText: "Empty"
    color: valueHolder.focus ? "#212121" : "#424242"
    property real myRealWidth: myTextRealMetr.width + leftPadding + rightPadding
    TextMetrics {
        id: myTextRealMetr
        text: valueHolder.text.length > 0 ? valueHolder.text : "12345678"
        font: valueHolder.font
    }
    //workaround [QTBUG-71875]
    background: Item {
        Rectangle {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            height: valueHolder.height
            width: valueHolder.width - 15
            color: valueHolder.activeFocus ? "#e8e8e8" : "#E0E0E0"
            border.width: 1
            border.color: valueHolder.activeFocus ? "#29B6F6" : "#9E9E9E"
            id: backRect
            Rectangle {
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                anchors.topMargin: 1
                anchors.bottomMargin: 1
                anchors.leftMargin: -4
                anchors.left: parent.right
                color: backRect.color
                width: 11
                id: hider
                z: 2
            }
            Rectangle {
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                anchors.leftMargin: -3
                anchors.left: backRect.right
                color: backRect.color
                border.width: 1
                border.color: backRect.border.color
                width: 18
                radius: 4
                z: 1
            }
        }
    }
}
