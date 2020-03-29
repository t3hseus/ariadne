import QtQuick 2.14
import QtQuick.Window 2.14
import QtQuick.Controls 2.14
import QtQuick.Layouts 1.14

Rectangle {
    anchors.fill: parent
    color: "#F5F5F5"
    property QtObject visualizer: visualizer_o ? visualizer_o : null
    onVisualizerChanged: {
        console.log("AAAAAAAAAA", visualizer)
    }

    objectName: "TOP_PANEL_OBJ"

    Row{
        anchors.margins: 6
        anchors.fill: parent
        spacing: 6
        Button{
            width: implicitWidth
            height: 28
            text: "Open data file"
            onClicked: {
                visualizer.current_data_file = input.text
            }
        }
        TextField{
            width: implicitWidth
            id:input
            placeholderText: "Path to data"
            text: "F:/Documents/Downloads/mpd_data_100_events_numpy.dat"
        }
    }
    Rectangle{
        height: 1
        color: "#ababab"
        width: parent.width
    }
    Rectangle{
        height: 1
        color: "#ababab"
        width: parent.width
        anchors.bottom: parent.bottom
    }
}
