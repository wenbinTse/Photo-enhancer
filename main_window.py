from my_test import test

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from ui import Ui_Form

import sys

class Window(QWidget, Ui_Form):
    origin_image = None
    result_image = None
    origin_image_name = ''

    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.open_file.clicked.connect(self.choose_file)

    def resizeEvent(self, a0: QResizeEvent):
        if self.origin_image is not None:
            self.resize_image(self.image_origin, self.origin_image)

        if self.result_image is not None:
            self.resize_image(self.image_result, self.result_image)

    @pyqtSlot()
    def choose_file(self):
        imgName, imgType = QFileDialog.getOpenFileName(self,
                                                       "打开图片",
                                                       "",
                                                       "Images (*.jpg *.jpeg *.tif *.bmp *.png)")

        if imgName == '':
            return

        self.origin_image = QPixmap(imgName)
        self.origin_image_name = imgName
        self.resize_image(self.image_origin, self.origin_image)
        self.repaint()

        test(imgName)

        self.result_image = QPixmap('result.png')
        self.resize_image(self.image_result, self.result_image)

    def resize_image(self, qlabel: QLabel, origin_image: QPixmap):
        img_ratio = origin_image.width() / origin_image.height()
        new_width, new_height = origin_image.width(), origin_image.height()
        label_width, label_height = qlabel.width(), qlabel.height()
        if new_width > label_width:
            new_width = label_width
            new_height = new_width / img_ratio
        if new_height > label_height:
            new_height = label_height
            new_width = new_height * img_ratio
        image = origin_image.scaled(new_width, new_height)
        qlabel.setPixmap(image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())