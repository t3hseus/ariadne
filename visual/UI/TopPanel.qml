import QtQuick 2.14
import QtQuick.Window 2.14
import QtQuick.Controls 2.14
import QtQuick.Layouts 1.14
import Qt.labs.settings 1.0


Rectangle {
    anchors.fill: parent
    color: "#F5F5F5"
    property QtObject visualizer: visualizer_o ? visualizer_o : null
    property int minEvent: visualizer.event_range[0]
    property int maxEvent: visualizer.event_range[1]

    onVisualizerChanged: {
        console.log("AAAAAAAAAA", visualizer)
    }

    objectName: "TOP_PANEL_OBJ"
    Settings{
        property alias lastFileText: input.text
    }

    GridLayout{
        anchors.margins: 6        
        anchors.fill: parent
        columnSpacing: 6
        columns: 2
        Button{
            Layout.preferredWidth: implicitWidth
            height: 28
            text: "Open data file"
            onClicked: {
                visualizer.current_data_file = input.text
            }
        }
        TextField{
            Layout.minimumWidth: 100
            Layout.fillWidth: true
            id:input
            placeholderText: "Path to data"
            readOnly: false
            selectByMouse: true
            text: ""
        }

        Button{
            Layout.preferredWidth: implicitWidth
            height: 28
            text: "Load event"
            visible: visualizer.current_data_file
            onClicked: {
                if (textidinput.text == "")
                    return
                visualizer.current_event = Number(textidinput.text)
            }
        }
        TextField{
            Layout.preferredWidth: Math.max(contentWidth + 40, 60)
            visible: visualizer.current_data_file
            id:textidinput
            placeholderText: "#"
            selectByMouse: true
            text: ""
            validator: IntValidator{bottom: minEvent; top: maxEvent}
        }
        Text {
            id:columntext
            text: "Columns:"
            font.pointSize: 12
            visible: visualizer.current_data_file
            Layout.preferredWidth: contentWidth
            height: 20
        }
        DataInfo{
            id:datainfo
            columnsList: datainfo.visible ? visualizer.columns : []
            visible: visualizer.current_data_file
            Layout.fillWidth: true
            Layout.preferredHeight: 20
        }
        Text {
            id:cameratext
            text: "Cameras:"
            font.pointSize: 12
            visible: visualizer.current_data_file
            Layout.preferredWidth: contentWidth
            height: 20
        }
        ComboBox{
            model: visualizer.cameras
            id:cameraselector
            currentIndex: -1
            visible: visualizer.current_data_file
            onActivated: {
                visualizer.set_camera(currentValue)
            }
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
