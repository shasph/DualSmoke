import sys
import os
from PyQt5.QtWidgets import (
  QBoxLayout, QCheckBox, QDockWidget, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QRadioButton, QVBoxLayout, QPushButton, QWidget, QApplication, QMainWindow, QAction,
  QFileDialog, QColorDialog, QInputDialog, QListWidget
)
from PyQt5.QtGui import QPainter, QImage, QPen, qRgb
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QDir
from collections import deque

TMP_FOLDER = 'tmp/'

WIDTH = 1024
HEIGHT = 1024
lines = []

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Canvasクラスを呼び出すよ。おいでー＾＾
        self.canvas = Canvas()
        # 呼び出したら箱に入れてあげようね。そうしないと動いてくれないよ。
        self.setCentralWidget(self.canvas)

        self.initCustomizeDock()
        self.initToolBar()

        self.setGeometry(300, 300, WIDTH, HEIGHT)
        self.setFixedSize(WIDTH, HEIGHT)
        self.setWindowTitle("MainWindow")
        self.show()

    def initCustomizeDock(self):
        self.dock = QDockWidget("Parameter", self)
        self.dock.setFloating(True)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock)

        dockWidget = QWidget(self)
        self.dock.setWidget(dockWidget)
        self.setWindowTitle("Dock demo")
        dockLayout = QVBoxLayout()
        dockWidget.setLayout(dockLayout)

        paintLayout = QGroupBox('Paint')
        cbLayout = QGroupBox('Boundary Condition (check:close)')
        actLayout = QGroupBox('Action')
        opLayout = QGroupBox()
        dockLayout.addWidget(paintLayout, alignment=(Qt.AlignTop))
        dockLayout.addWidget(cbLayout, alignment=Qt.AlignTop)
        dockLayout.addWidget(actLayout, alignment=Qt.AlignBottom)
        dockLayout.addWidget(opLayout, alignment=Qt.AlignBottom)

        paintObs = QRadioButton('Obstacle', self)
        paintSmoke = QRadioButton('Smoke', self)
        paintObs.setChecked(True)

        paintVbox = QVBoxLayout()
        paintVbox.addWidget(paintObs)
        paintVbox.addWidget(paintSmoke)
        paintLayout.setLayout(paintVbox)

        self.cbTop = QCheckBox('Top', self)
        self.cbBottom = QCheckBox('Bottom', self)
        self.cbRight = QCheckBox('Right', self)
        self.cbLeft = QCheckBox('Left', self)

        cbVbox = QVBoxLayout()
        cbVbox.addWidget(self.cbTop)
        cbVbox.addWidget(self.cbBottom)
        cbVbox.addWidget(self.cbRight)
        cbVbox.addWidget(self.cbLeft)
        cbLayout.setLayout(cbVbox)

        actReset = QPushButton('Reset', self)
        actReset.clicked.connect(self.canvas.resetImage)
        actBack = QPushButton('Back', self)
        actBack.setShortcut('Ctrl+Z')
        actBack.clicked.connect(self.canvas.backImage)
        actNext = QPushButton('Next', self)
        actNext.setShortcut('Ctrl+Y')
        actNext.clicked.connect(self.canvas.nextImage)

        actVbox = QVBoxLayout()
        actVbox.addWidget(actReset)
        actVbox.addWidget(actBack)
        actVbox.addWidget(actNext)
        actLayout.setLayout(actVbox)

        opEnter = QPushButton('Enter', self)
        opEnter.clicked.connect(self.nextPhase)
        opQuit = QPushButton('Quit', self)
        opQuit.clicked.connect(self.exitSys)
                
        opVBox = QVBoxLayout()
        opVBox.addWidget(opEnter)
        opVBox.addWidget(opQuit)
        opLayout.setLayout(opVBox)


    def initToolBar(self):
        menubar = self.menuBar()

        actOpen = QAction('&Open', self)
        actOpen.setShortcut('Ctrl+O')
        actOpen.triggered.connect(self.openFile)

        actQuit = QAction('&Quit', self)
        actQuit.setShortcut('Ctrl+Q')
        actQuit.triggered.connect(self.exitSys)

        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(actOpen)
        fileMenu.addAction(actQuit)

        selectColorAct = QAction('&Pen Color', self)
        selectColorAct.triggered.connect(self.selectColor)

        selectWidthAct = QAction('&Pen Width', self)
        selectWidthAct.triggered.connect(self.selectWidth)

        penColorMenu = menubar.addMenu('&Pen Color')
        penColorMenu.addAction(selectColorAct)
        penWidthMenu = menubar.addMenu('&Pen Width')
        penWidthMenu.addAction(selectWidthAct)

        dockMenu = menubar.addMenu('&Dock')
        dockMenu.addAction(self.dock.toggleViewAction())


    def selectColor(self):
        newColor = QColorDialog.getColor(self.canvas.penColor())
        self.canvas.setPenColor(newColor)

    def selectWidth(self):
        newWidth, ok = QInputDialog.getInt(
            self, "select",
            "select pen width: ", self.canvas.penWidth(), 1, 100, 1
        )
        if ok:
           self.canvas.setPenWidth(newWidth)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        if fileName:
           self.canvas.openImage(fileName)

    def saveList(self):
        path = QDir.currentPath()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save as", path)
      

    def nextPhase(self):
        if(os.path.exists(TMP_FOLDER)):
            with open(TMP_FOLDER + 'obs.txt', 'wt') as fout:
                fout.write("%d %d\n" % (WIDTH, HEIGHT))
                for line in lines:
                    for l in line:
                        fout.write("%d %d\n" % (l[0], l[1]))
        self.terminate(1)

    def exitSys(self):
        self.terminate(0)        

    def getBoundaryCond(self):
        s = ''
        if(self.cbTop.isChecked()):
            s += 't'
        if(self.cbBottom.isChecked()):
            s += 'b'
        if(self.cbLeft.isChecked()):
            s += 'l'
        if(self.cbRight.isChecked()):
            s += 'r'
        s += ','
        return s

    def terminate(self, n):
        code = ''
        if (n == 0):
            code = '0,'
        else:
            code = '1,' + self.getBoundaryCond()
        print(code)
        self.close()


class Canvas(QWidget):
    def __init__(self, parent = None):
        super(Canvas, self).__init__(parent)

        self.myPenWidth = 2
        self.myPenColor = Qt.black
        self.image = QImage()
        self.check = False
        self.back = deque(maxlen = 10)
        self.next = deque(maxlen = 10)
        self.line = []
        # initUIはもう必要ないから消しておこうね

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.back.append(self.resizeImage(self.image, self.image.size()))
            self.lastPos = event.pos()
            self.check = True

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.check:
            self.calcLinePoints(self.lastPos, event.pos())
            # self.lastPos = event.pos()
            self.drawLine(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.check:
            self.calcLinePoints(self.lastPos, event.pos())
            self.drawLine(event.pos())
            self.check = False
            lines.append(self.line)
            self.line = []

    def calcLinePoints(self, sP, gP):
        dx = abs(sP.x() - gP.x())
        dy = abs(sP.y() - gP.y())
        if(dx < dy):
            y0 = sP.y()
            if(sP.y() < gP.y()):
                while(y0 < gP.y()):
                    self.line.append([self.lerpY(sP, gP, y0), y0])
                    y0 += 1
            else:
                while(gP.y() < y0):
                    self.line.append([self.lerpY(sP, gP, y0), y0])
                    y0 -= 1
        else:
            x0 = sP.x()
            if(sP.x() < gP.x()):
                while(x0 < gP.x()):
                    self.line.append([x0, self.lerpX(sP, gP, x0)])
                    x0 += 1
            else:
                while(gP.x() < x0):
                    self.line.append([x0, self.lerpX(sP, gP, x0)])
                    x0 -= 1



    def lerpX(self, sP, gP, x):
        return sP.y() + (gP.y() - sP.y()) * (x - sP.x()) / (gP.x() - sP.x())

    def lerpY(self, sP, gP, y):
        return sP.x() + (gP.x() - sP.x()) * (y - sP.y()) / (gP.y() - sP.y())

    def drawLine(self, endPos):
        painter = QPainter(self.image)
        painter.setPen(
            QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        painter.drawLine(self.lastPos, endPos)
        self.update()
        self.lastPos = QPoint(endPos)


    def paintEvent(self, event):
        painter = QPainter(self)
        rect = event.rect()
        painter.drawImage(rect, self.image, rect)

    def resizeEvent(self, event):
        if self.image.width() < self.width() or self.image.height() < self.height():
            changeWidth = max(self.width(), self.image.width())
            changeHeight = max(self.height(), self.image.height())
            self.image = self.resizeImage(self.image, QSize(changeWidth, changeHeight))
            self.update()

    def resizeImage(self, image, newSize):
        changeImage = QImage(newSize, QImage.Format_RGB32)
        changeImage.fill(qRgb(255, 255, 255))
        painter = QPainter(changeImage)
        painter.drawImage(QPoint(0, 0), image)
        return changeImage

    def saveImage(self, filename):
        if self.image.save(filename):
            return True
        else:
            return False

    def openImage(self, filename):
        image = QImage()
        if not image.load(filename):
            return False

        self.image = image
        self.update()
        return True

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def resetImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.update()
        global lines
        lines = []

    def backImage(self):
        if self.back:
            back_ = self.back.pop()
            self.next.append(back_)
            self.image = QImage(back_)
            self.update()
            lines.pop(-1)

    def nextImage(self):
        if self.next:
            next_ = self.next.pop()
            self.back.append(next_)
            self.image = QImage(next_)
            self.update()

if __name__ == '__main__':
  # 動け～
    app = QApplication(sys.argv)
  # ここがCanvasのままだと何も表示されないよ。気を付けようね
    ex = MainWindow()
    sys.exit(app.exec_())