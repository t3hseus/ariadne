import QtQuick 2.0

Item{
    property var columnsList: []
    Row{
        id:columnsrow
        spacing: 6
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        height: 20
        Repeater{
            model: columnsList
            ColumnInfo{
                columnName: modelData
            }
        }
    }
}
