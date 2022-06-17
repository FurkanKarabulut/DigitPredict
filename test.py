import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import os
from PIL import Image, ImageOps
import pickle

path = os.path.dirname(__file__)
os.chdir(path)


class MainWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        self.loaded_model = pickle.load(open('modelSvm.pkl', 'rb'))

        self.initUI()

    def initUI(self):
        self.container = QtWidgets.QVBoxLayout()
        self.container.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel()
        window = QtGui.QPixmap(500, 500)
        window.fill(QtGui.QColor("black"))
        self.label.setPixmap(window)
        self.last_x, self.last_y = None, None

        self.prediction = QtWidgets.QLabel('Tahmin Sonucu: ...')
        self.prediction.setFont(QtGui.QFont('Monospace', 18))

        self.button_clear = QtWidgets.QPushButton('TEMİZLE')
        self.button_clear.clicked.connect(self.canvasClear)

        self.button_save = QtWidgets.QPushButton('TAHMİN ET')
        self.button_save.clicked.connect(self.predict)

        self.container.addWidget(self.label)
        self.container.addWidget(self.prediction, alignment=QtCore.Qt.AlignHCenter)
        self.container.addWidget(self.button_clear)
        self.container.addWidget(self.button_save)

        self.setLayout(self.container)

    def canvasClear(self):
        self.label.pixmap().fill(QtGui.QColor('#000000'))
        self.update()

    def predict(self):
        s = self.label.pixmap().toImage().bits().asarray(500 * 500 * 4)
        array = np.frombuffer(s, dtype=np.uint8).reshape((500, 500, 4))
        array = np.array(ImageOps.grayscale(Image.fromarray(array).resize((28, 28), Image.ANTIALIAS)))
        array = (array / 255.0).reshape(1, -1)
        self.prediction.setText('Prediction: ' + str(self.loaded_model.predict(array)[0]))

    def mouseMoveEvent(self, e):
        if self.last_x is None:
            self.last_x = e.x()
            self.last_y = e.y()
            return

        pointer = QtGui.QPainter(self.label.pixmap())

        p = pointer.pen()
        p.setWidth(20)
        self.pen_color = QtGui.QColor('#FFFFFF')
        p.setColor(self.pen_color)
        pointer.setPen(p)

        pointer.drawLine(self.last_x, self.last_y, e.x(), e.y())
        pointer.end()
        self.update()

        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None


class AppWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.UIinit()

    def UIinit(self):
        self.mainWidget = MainWidget()
        self.setCentralWidget(self.mainWidget)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    App = AppWindow()
    App.setWindowTitle('RAKAM TAHMİN')
    App.show()
    sys.exit(app.exec_())
