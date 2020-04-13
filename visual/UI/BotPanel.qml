import QtQuick 2.0

Rectangle {
    anchors.fill: parent
    color: "#F5F5F5"
    width: 700
    height: 420
    CodeRunner{
        height: realHeight
        width: parent.width
        anchors.centerIn: parent
    }
}
