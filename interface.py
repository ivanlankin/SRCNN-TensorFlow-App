from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys

from matplotlib import widgets
from sympy import Q
from demo import *
from test import *


class MainWindow(QMainWindow): # главное окно
    def __init__(self):
        super().__init__()
        self.setupUi()
    def setupUi(self):
        self.setWindowTitle("SRCNN TensorFlow App") # заголовок окна
        self.move(200, 100) # положение окна
        self.setFixedSize(QSize(400, 320))
        self.check = False
        self.model_type = 915
        self.scale = 2
        self.button = QPushButton("Start depixelization", self)
        self.button.setGeometry(QRect(10, 250, 110, 30))
        self.button.clicked.connect(self.the_start_was_clicked)
        self.button.setEnabled(False)

        self.button_before = QPushButton("LR image", self)
        self.button_before.setGeometry(QRect(250, 10, 100, 30))
        self.button_before_scaled = QPushButton("LR image (scaled)", self)
        self.button_before_scaled.setGeometry(QRect(250, 60, 100, 30))
        self.button_bicubic = QPushButton("bicubic image", self)
        self.button_bicubic.setGeometry(QRect(250, 110, 100, 30))
        self.button_after = QPushButton("HR image", self)
        self.button_after.setGeometry(QRect(250, 160, 100, 30))

        self.button_before.clicked.connect(lambda:self.show_image_window(self.button_before.text()))
        self.button_before_scaled.clicked.connect(lambda:self.show_image_window(self.button_before_scaled.text()))
        self.button_bicubic.clicked.connect(lambda:self.show_image_window(self.button_bicubic.text()))
        self.button_after.clicked.connect(lambda:self.show_image_window(self.button_after.text()))

        self.button_before.setEnabled(False)
        self.button_before_scaled.setEnabled(False)
        self.button_bicubic.setEnabled(False)
        self.button_after.setEnabled(False)

        self.dialog_button = QPushButton("Open file", self)
        self.dialog_button.setGeometry(QRect(10, 10, 100, 30))
        self.dialog_button.clicked.connect(self.dialog)

        self.save_button = QPushButton("Save file", self)
        self.save_button.setGeometry(QRect(110, 10, 100, 30))
        self.save_button.clicked.connect(self.saveFileDialog)
        self.save_button.setEnabled(False)

        label_widget1 = QLabel("Choose a neural network to improve", self)
        label_widget1.setGeometry(10, 40, 200, 30)
        label_widget2 = QLabel("Choose dimensions increase", self)
        label_widget2.setGeometry(10, 135, 200, 30)

        self.label_PSNR_HR = QLabel("HR PSNR:", self)
        self.label_PSNR_HR.setGeometry(200, 250, 200, 30)
        self.layout_first()
        self.layout_second()
        self.pixmap_before = None
        self.pixmap_bicubic = None
        self.pixmap_after = None
        self.resize_before = None
        
    def layout_first(self):
        widget = QWidget(self)
        vbox = QVBoxLayout()
        b915 = QRadioButton("SRCNN915", widget)
        b935 = QRadioButton("SRCNN935", widget)
        b955 = QRadioButton("SRCNN955", widget)
        b915.toggled.connect(lambda:self.btn_mode_state(b915))
        b935.toggled.connect(lambda:self.btn_mode_state(b935))
        b955.toggled.connect(lambda:self.btn_mode_state(b955))
        b915.setChecked(True)
        vbox.addWidget(b915)
        vbox.addWidget(b935)
        vbox.addWidget(b955)
        widget.setGeometry(10, 60, 90, 90)
        widget.setLayout(vbox)

    def layout_second(self):
        widget = QWidget(self)
        vbox = QVBoxLayout()
        bx2 = QRadioButton("x2", widget)
        bx3 = QRadioButton("x3", widget)
        bx4 = QRadioButton("x4", widget)
        bx2.toggled.connect(lambda:self.btn_scale_state(bx2))
        bx3.toggled.connect(lambda:self.btn_scale_state(bx3))
        bx4.toggled.connect(lambda:self.btn_scale_state(bx4))
        bx2.setChecked(True)
        vbox.addWidget(bx2)
        vbox.addWidget(bx3)
        vbox.addWidget(bx4)
        widget.setGeometry(10, 150, 90, 90)
        widget.setLayout(vbox)
    


    def btn_mode_state(self, b):
        if b.text() == "SRCNN915":
            if b.isChecked():
                self.model_type = 915
        elif b.text() == "SRCNN935":
            if b.isChecked():
                self.model_type = 935
        elif b.text() == "SRCNN955":
            if b.isChecked():
                self.model_type = 955
    
    def btn_scale_state(self, b):
        if b.text() == "x2":
            if b.isChecked():
                self.scale = 2
        elif b.text() == "x3":
            if b.isChecked():
                self.scale = 3
        elif b.text() == "x4":
            if b.isChecked():
                self.scale = 4

    def the_start_was_clicked(self):
        if self.check:
            begin_sr(str(self.model_type), self.file, "checkpoint/SRCNN{}/SRCNN-{}.h5".format(self.model_type, self.model_type), self.scale)
            self.resize_before = NewImage("lr upscaled image", QPixmap(self.file).scaled(QPixmap("bicubic.png").width(), QPixmap("bicubic.png").height()))
            self.resize_before.move(self.pos().x(), self.pos().y() + self.height() + 50)
            self.pixmap_bicubic = NewImage("bicubic image",QPixmap("bicubic.png"))
            self.pixmap_bicubic.move(self.pos().x() + self.resize_before.width() + 20, self.pos().y() + self.height() + 50)
            self.pixmap_after = NewImage("hr image",QPixmap("sr.png"))
            self.pixmap_after.move(self.pos().x() + self.resize_before.width() + self.pixmap_bicubic.width() + 40, self.pos().y() + self.height() + 50)
            self.label_PSNR_HR.setText("HR PSNR: " + str(test_psnr(self.scale, str(self.model_type))))
            self.button_before_scaled.setEnabled(True)
            self.button_bicubic.setEnabled(True)
            self.button_after.setEnabled(True)
            self.save_button.setEnabled(True)
            

    def dialog(self):
        self.file , self.check = QFileDialog.getOpenFileName(None, "Open File", "", "PNG (*.png);; JPG (*jpg)")
        if self.check:
            self.pixmap_before = NewImage("lr image",QPixmap(self.file))
            self.pixmap_before.move(self.pos().x() + self.width() + 20, self.pos().y() + self.height() - self.pixmap_before.height())
            self.button.setEnabled(True)
            self.button_before.setEnabled(True)
            self.button_before_scaled.setEnabled(False)
            self.button_bicubic.setEnabled(False)
            self.button_after.setEnabled(False)
            if self.pixmap_bicubic:
                self.pixmap_bicubic.close()
            if self.pixmap_after:
                self.pixmap_after.close()
            if self.resize_before:
                self.resize_before.close()


    def show_image_window(self, text):
        if self.check:
            if text == "LR image" and self.pixmap_before:
                self.pixmap_before.show()
                self.pixmap_before.move(self.pos().x() + self.width() + 20, self.pos().y() + self.height() - self.pixmap_before.height())
            if text == "LR image (scaled)" and self.resize_before:
                self.resize_before.show()
                self.resize_before.move(self.pos().x(), self.pos().y() + self.height() + 50)
            if text == "bicubic image" and self.pixmap_bicubic:
                self.pixmap_bicubic.show()
                self.pixmap_bicubic.move(self.pos().x() + self.resize_before.width() + 20, self.pos().y() + self.height() + 50)
            if text == "HR image" and self.pixmap_after:
                self.pixmap_after.show()
                self.pixmap_after.move(self.pos().x() + self.resize_before.width() + self.pixmap_bicubic.width() + 40, self.pos().y() + self.height() + 50)
    
    def saveFileDialog(self):
        filename, checked = QFileDialog.getSaveFileName(self,"Save File ","","PNG (*.png)")
        if checked:
            pixmap = read_image("sr.png")
            write_image(filename, pixmap)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            if self.pixmap_before:
                self.pixmap_before.close()
            if self.pixmap_bicubic:
                self.pixmap_bicubic.close()
            if self.pixmap_after:
                self.pixmap_after.close()
            if self.resize_before:
                self.resize_before.close()
            event.accept()
        else:
            event.ignore()

class NewImage(QWidget):

    def __init__(self, name, file):
        super().__init__()
        label = QLabel(self)
        pixmap = QPixmap(file)
        self.setWindowTitle(name)
        label.setPixmap(pixmap)
        label.setScaledContents(False)
        self.setFixedSize(min(pixmap.width(), 500) + 24, min(pixmap.height(), 500) + 24)
        scroll_area = QScrollArea(self)
        scroll_area.setMinimumWidth(label.width())
        scroll_area.setMinimumHeight(label.height())
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setWidget(label)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll_area)
        scroll_area.show()
        self.show()
        

        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())