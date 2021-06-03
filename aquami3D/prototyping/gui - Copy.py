# -*- coding: utf-8 -*-
import sys
from threading import Thread
from pathlib import Path
import time

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import lognorm, norm

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import Qt

    # from .visualization import VtkWindow
    # from .inout import *
    # from .measure import *
    # from .qslider import QSliceRange

from visualization import VtkWindow
from inout import *
from measure import *
from qslider import QSliceRange


class MainWindow(Qt.QMainWindow):
    # Threading signals
    calc_skel = pyqtSignal(np.ndarray)
    calc_dist = pyqtSignal(np.ndarray)

    def __init__(self, parent = None):
        Qt.QMainWindow.__init__(self)
        self.initUI()

        # Steps
        self.im = None
        self.skel = None
        self.dist = None
        self.diam = None

        # Connect threaded outputs to finishing function
        self.calc_skel.connect(self.finish_skeletonize)
        self.calc_dist.connect(self.finish_dist)

    def initUI(self):
        self.setGeometry(200, 200, 600, 600)
        self.setWindowTitle("AQUAMI 3D")
        self.statusBar().setStyleSheet("background-color:white")

        # Menu frame
        self.menuFrame = QFrame
        self.menuGrid = QGridLayout()
        self.menuGrid.setSpacing(10)
        self.menuGrid.setRowStretch(10,1)

        load_button = QPushButton("Load", self)
        load_button.setToolTip('Load a 3D dataset')
        load_button.clicked.connect(self.load_click)
        self.menuGrid.addWidget(load_button, 0,0)

        skel_button = QPushButton("Skeletonize", self)
        skel_button.setToolTip('Skeletonize image')
        skel_button.clicked.connect(self.skel_click)
        self.menuGrid.addWidget(skel_button, 1, 0)

        dist_button = QPushButton("Dist Transform", self)
        dist_button.setToolTip('Find the distance transform of the image')
        dist_button.clicked.connect(self.dist_click)
        self.menuGrid.addWidget(dist_button, 2, 0)

        diam_button = QPushButton("Diameter", self)
        diam_button.setToolTip('Calculate the diameter distribution')
        diam_button.clicked.connect(self.diam_click)
        self.menuGrid.addWidget(diam_button, 3, 0)

        self.sliceRange = QSliceRange()

        self.vtk = VtkWindow()

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setRowStretch(0, 10)
        grid.setRowMinimumHeight(0,500)
        grid.setColumnStretch(1, 5)
        grid.addLayout(self.menuGrid, 0, 0)  # widget,row,column
        grid.addWidget(self.vtk, 0,1)
        grid.addWidget(self.sliceRange, 1,1)


        centralWidget = QWidget()
        centralWidget.setLayout(grid)
        self.setCentralWidget(centralWidget)

        self.show()

        self.threadpool = QThreadPool()

    @pyqtSlot()
    def load_click(self):
        path, _ = QFileDialog.getOpenFileName(self,"Select 3D image")
        self.im = read_tiff_stack(path)
        self.vtk.update(self.im, (0.05, 0.5))
        self.sliceRange.set_range_maximums(self.im.shape)
        try:
            self.sliceRange.valueChanged.disconnect()
        except:
            pass
        self.sliceRange.valueChanged.connect(self.load_update)

    @pyqtSlot()
    def skel_click(self):
        if self.im is None:
            QMessageBox.warning(self, "Warning", "Please load an image first",
                                QMessageBox.Ok)
        elif self.skel is None:
            self.statusBar().showMessage("Skeletonizing...")
            Thread(target=self.start_skeletonize).start()
        else:
            self.vtk.update(self.skel, (0, 1))

    def start_skeletonize(self):
        self.calc_skel.emit(skeletonize(self.im))
        
    def finish_skeletonize(self, skel):
        self.skel = skel
        self.statusBar().showMessage("Finding nodes...")
        self.nodes = find_nodes(self.skel)
        self.vtk.display(self.im, self.skel, self.nodes)
        try:
            self.sliceRange.valueChanged.disconnect()
        except:
            pass
        self.sliceRange.valueChanged.connect(self.skel_update)
        self.statusBar().showMessage("")

    @pyqtSlot()
    def dist_click(self):
        if self.im is None:
            QMessageBox.warning(self, "Warning", "Please load an image first",
                                QMessageBox.Ok)
        elif self.dist is None:
            self.statusBar().showMessage("Calculating distance tranform...")
            Thread(target=self.start_dist).start()
        else:
            self.vtk.update(self.dist, (0, 1))

    def start_dist(self):
        self.calc_dist.emit(distance_transform(self.im))


    def finish_dist(self, dist):
        self.dist = dist
        self.vtk.update(self.dist, (0, 0.5))
        self.statusBar().showMessage("")

    @pyqtSlot()
    def diam_click(self):
        if self.im is None:
            QMessageBox.warning(self, "Warning", "Please load an image first",
                                QMessageBox.Ok)
        else:
            data = VolumeData(self.im)
            # diams = calculate_diameter(self.skel, self.dist)
            # n, bins, patches = plt.hist(diams, 20, edgecolor='black', normed=1)
            # gfit = norm.fit(diams.flatten())
            # gauss_plot = norm.pdf(bins, gfit[0], gfit[1])
            # plt.plot(bins, gauss_plot, 'r--', linewidth=2)
            # plt.show()

    def load_update(self, i):
        self.vtk.update(self.im[i[0]:i[1], i[2]:i[3], i[4]:i[5]], (0.05, 0.5))

    def skel_update(self, i):
        self.vtk.display(self.im[i[0]:i[1], i[2]:i[3], i[4]:i[5]],
                         self.skel[i[0]:i[1], i[2]:i[3], i[4]:i[5]],
                         self.nodes[i[0]:i[1], i[2]:i[3], i[4]:i[5]])



if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
