import QtQuick 2.12

Rectangle {
    property string columnName: "<none>"
    width: name.contentWidth + 4
    height: 18

    color: area.containsMouse ? "#D5D5D5":"transparent"
    border.width: 1
    border.color: "#ababab"

    Text {
        id: name
        text: columnName
        anchors.centerIn: parent
        font.pointSize: 10
    }
    MouseArea{
        id:area
        anchors.fill: parent
        hoverEnabled: true

    }
}
