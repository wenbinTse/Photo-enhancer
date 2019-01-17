# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1000, 400)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.open_file = QtWidgets.QPushButton(Form)
        self.open_file.setObjectName("open_file")
        self.verticalLayout.addWidget(self.open_file, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.image_origin = QtWidgets.QLabel(Form)
        self.image_origin.setAlignment(QtCore.Qt.AlignCenter)
        self.image_origin.setObjectName("image_origin")
        self.horizontalLayout_2.addWidget(self.image_origin)
        self.image_result = QtWidgets.QLabel(Form)
        self.image_result.setAlignment(QtCore.Qt.AlignCenter)
        self.image_result.setObjectName("image_result")
        self.horizontalLayout_2.addWidget(self.image_result)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "图像增强"))
        self.open_file.setText(_translate("Form", "打开图片"))
        self.image_origin.setText(_translate("Form", "处理前"))
        self.image_result.setText(_translate("Form", "处理后"))

