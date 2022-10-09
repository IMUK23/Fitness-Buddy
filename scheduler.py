# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scheduler.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1180, 980)
        MainWindow.setMinimumSize(QtCore.QSize(1180, 980))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(340, 30, 571, 71))
        self.label.setObjectName("label")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setEnabled(True)
        self.frame.setGeometry(QtCore.QRect(210, 130, 811, 781))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.checkBox = QtWidgets.QCheckBox(self.frame)
        self.checkBox.setGeometry(QtCore.QRect(350, 140, 161, 41))
        self.checkBox.setObjectName("checkBox")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(80, 30, 711, 71))
        self.label_2.setObjectName("label_2")
        self.checkBox_2 = QtWidgets.QCheckBox(self.frame)
        self.checkBox_2.setGeometry(QtCore.QRect(350, 200, 161, 41))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.frame)
        self.checkBox_3.setGeometry(QtCore.QRect(350, 260, 171, 41))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.frame)
        self.checkBox_4.setGeometry(QtCore.QRect(350, 320, 171, 41))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.frame)
        self.checkBox_5.setGeometry(QtCore.QRect(350, 380, 181, 41))
        self.checkBox_5.setObjectName("checkBox_5")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(50, 560, 301, 51))
        self.label_3.setObjectName("label_3")
        self.spinBox = QtWidgets.QSpinBox(self.frame)
        self.spinBox.setGeometry(QtCore.QRect(370, 570, 111, 31))
        self.spinBox.setObjectName("spinBox")
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(330, 700, 181, 71))
        self.pushButton.setStyleSheet("color:red;\n"
"font-size:15px;")
        self.pushButton.setObjectName("pushButton")
        self.checkBox_6 = QtWidgets.QCheckBox(self.frame)
        self.checkBox_6.setGeometry(QtCore.QRect(350, 440, 131, 41))
        self.checkBox_6.setObjectName("checkBox_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1180, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:28pt; font-weight:600; color:#aa0000;\">Yoga Schedule Creator</span></p></body></html>"))
        self.checkBox.setText(_translate("MainWindow", "Tadasana"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Please Tick the Checkbox for every pose you want to add</span></p></body></html>"))
        self.checkBox_2.setText(_translate("MainWindow", "Bhujangasana"))
        self.checkBox_3.setText(_translate("MainWindow", "Trikonasana"))
        self.checkBox_4.setText(_translate("MainWindow", "Padamasana"))
        self.checkBox_5.setText(_translate("MainWindow", "Shavasana"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Enter the number of reps</span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Create Schedule"))
        self.checkBox_6.setText(_translate("MainWindow", "Vrikashasana"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

