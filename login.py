# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\ASUS\Desktop\Project work\major project\UI files\login.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1294, 773)
        MainWindow.setMinimumSize(QtCore.QSize(1294, 0))
        MainWindow.setStyleSheet("background-color:rgb(0, 20, 93);\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setStyleSheet("font: 8pt \"MS Sans Serif\";\n"
"background-color:rgb(255, 255, 255)")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setObjectName("gridLayout")
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setStyleSheet("background-color:rgb(0, 20, 93);\n"
"font: 75 8pt \"MS Shell Dlg 2\";\n"
"color:rgb(255, 255, 255);\n"
"font: 75 italic 20pt \"Georgia\";")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.pushButton = QtWidgets.QPushButton(self.frame)
        self.pushButton.setStyleSheet("background-color: white;\n"
"border-style: outset;\n"
"border-width: 0px;\n"
"border-radius: 0px;\n"
"padding: 4px;\n"
"color:red;\n"
"text-decoration:underline;")
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 5, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setStyleSheet("background-color: orange;\n"
"border-style: outset;\n"
"border-width: 0px;\n"
"border-radius: 15px;\n"
"border-color: black;\n"
"padding: 4px;\n"
"color:white;\n"
"font-size:15px;\n"
"font: 75 18pt \"MS Shell Dlg 2\";")
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 6, 0, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setStyleSheet("background-color: white;\n"
"font: 12pt \"MS Shell Dlg 2\";\n"
"border-style: outset;\n"
"border-width: 0px;\n"
"border-radius: 0px;\n"
"padding: 4px;\n"
"color:red;\n"
"text-decoration:underline;")
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 7, 0, 1, 2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame)
        self.lineEdit_2.setStyleSheet("height:100px;")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)
        self.gridLayout.addWidget(self.lineEdit_2, 4, 0, 1, 2)
        self.lineEdit = QtWidgets.QLineEdit(self.frame)
        self.lineEdit.setMinimumSize(QtCore.QSize(0, 22))
        self.lineEdit.setStyleSheet("height:100px;")
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 2, 0, 1, 2)
        self.gridLayout_2.addWidget(self.frame, 1, 2, 2, 1)
        self.boundary2 = QtWidgets.QLabel(self.centralwidget)
        self.boundary2.setObjectName("boundary2")
        self.gridLayout_2.addWidget(self.boundary2, 0, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 0, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 1, 3, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 2, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 2, 3, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 3, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 3, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setStyleSheet("border-image:url(../images/Login_image.jpg);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 1, 1, 2, 1)
        self.boundary1 = QtWidgets.QLabel(self.centralwidget)
        self.boundary1.setText("")
        self.boundary1.setObjectName("boundary1")
        self.gridLayout_2.addWidget(self.boundary1, 3, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 3, 3, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">Password</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">FITNESS BUDDY</p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "Forgot Password"))
        self.pushButton_2.setText(_translate("MainWindow", "LOGIN"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt;\">Username</span></p></body></html>"))
        self.pushButton_3.setText(_translate("MainWindow", "Not having username? Sign up here"))
        self.boundary2.setText(_translate("MainWindow", ""))
        self.label_3.setText(_translate("MainWindow", ""))
        self.label_7.setText(_translate("MainWindow", ""))
        self.label_9.setText(_translate("MainWindow", ""))
        self.label_4.setText(_translate("MainWindow", ""))
        self.label_8.setText(_translate("MainWindow", ""))
        self.label_11.setText(_translate("MainWindow", ""))
        self.label_12.setText(_translate("MainWindow", ""))
        self.label_13.setText(_translate("MainWindow", ""))
        self.label_10.setText(_translate("MainWindow", ""))
        self.label_14.setText(_translate("MainWindow", ""))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

