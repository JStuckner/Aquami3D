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
from PyQt5.Qt import QMainWindow, QApplication

from visualization import VtkWindow, Plot, PlotCanvas
from inout import *
from measure import *
from qslider import QSliceRange


class MainWindow(QMainWindow):
    volume_data = None

    def __init__(self, parent = None):
        QMainWindow.__init__(self)
        self.initUI()

    def initUI(self):
        # region MainWindow properties
        self.setGeometry(200, 200, 600, 600)
        self.setWindowTitle("AQUAMI 3D")
        # endregion
        # region Status bar
        self.statusBar().setStyleSheet("background-color:white")
        self.progress = QProgressBar()
        self.statusLabel = QLabel()
        self.statusBar().addPermanentWidget(self.statusLabel, 1)
        self.statusBar().addPermanentWidget(self.progress, 1)
        self.progress.setVisible(False)
        # endregion

        # Menu frame
        self.menuFrame = QFrame
        self.menuGrid = QGridLayout()
        self.menuGrid.setSpacing(10)
        self.menuGrid.setRowStretch(10,1)

        load_button = QPushButton("Load", self)
        load_button.setToolTip('Load a 3D dataset')
        load_button.clicked.connect(self.load_click)
        self.menuGrid.addWidget(load_button, 0,0)

        calc_button = QPushButton("Calculate", self)
        calc_button.setToolTip('Skeletonize image')
        calc_button.clicked.connect(self.calculate)
        self.menuGrid.addWidget(calc_button, 1, 0)

        self.save_button = QPushButton('Save data', self)
        self.save_button.setToolTip('Save all measurements')
        self.save_button.clicked.connect(self.save_clicked)
        self.save_button.setEnabled(False)
        self.menuGrid.addWidget(self.save_button, 2,0)

        # Pixel size
        lblPixelSize = QLabel('Pixel Size:')
        lblXPixelSize = QLabel('X:')
        lblYPixelSize = QLabel('Y:')
        lblZPixelSize = QLabel('Z:')
        self.txtXPixelSize = QLineEdit(self)
        self.txtYPixelSize = QLineEdit(self)
        self.txtZPixelSize = QLineEdit(self)
        self.validator = QDoubleValidator()
        layoutXPix = QHBoxLayout()
        layoutYPix = QHBoxLayout()
        layoutZPix = QHBoxLayout()
        self.menuGrid.addWidget(lblPixelSize)
        for i, lbl, txt, layout in zip(
                (4,5,6),
                (lblXPixelSize, lblYPixelSize, lblZPixelSize),
                (self.txtXPixelSize, self.txtYPixelSize, self.txtZPixelSize),
                (layoutXPix, layoutYPix, layoutZPix)):
            txt.setValidator(self.validator)
            layout.addWidget(lbl)
            layout.addWidget(txt)
            self.menuGrid.addLayout(layout, i,0)


        self.sliceRange = QSliceRange()
        self.vtk = VtkWindow()

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.setRowStretch(0, 10)
        grid.setRowMinimumHeight(0,500)
        grid.setColumnStretch(1, 5)
        grid.setColumnMinimumWidth(1, 500)
        grid.addLayout(self.menuGrid, 0, 0)  # widget,row,column
        grid.addWidget(self.vtk, 0,1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(self.sliceRange, 1,1)

        # region matplotlib
        self.plot = Plot(self)
        grid.addWidget(self.plot, 0,2)
        self.plot.setVisible(False)
        # endregion

        centralWidget = QWidget()
        centralWidget.setLayout(grid)
        self.setCentralWidget(centralWidget)

        self.show()

    @pyqtSlot()
    def load_click(self):
        path, _ = QFileDialog.getOpenFileName(self,"Select 3D image")
        if path == '':
            return
        try: #tiff stack
            im = read_tiff_stack(path)
        except OSError: #xyz file
            im = read_xyz(path)
        im = im[:150, :150, :150]
        #im = im[:100, :100, :100]

        try:
            xsize = float(self.txtXPixelSize.text())
            ysize = float(self.txtYPixelSize.text())
            zsize = float(self.txtZPixelSize.text())
            im, pixel_size = resize_and_get_pixel_size(im, xsize,ysize,zsize)
        except:
            print('Size values not entered correctly.\n'
                  'No scaling and pixel size is 1.')
            pixel_size = 1
        self.sliceRange.set_range_maximums(im.shape)

        self.volume_data = VolumeData(im, pixel_size, self.statusLabel, self.progress,
                                      self.display, self.plot, invert_im=False)
        self.vtk.update2(self.volume_data.im, (0.05, 0.5))
        try:
            self.sliceRange.valueChanged.disconnect()
        except:
            pass
        self.sliceRange.valueChanged.connect(self.load_update)

    @pyqtSlot()
    def calculate(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "Warning", "Please load an image first",
                                QMessageBox.Ok)
        else:
            self.volume_data.calculate()
            self.sliceRange.valueChanged.disconnect()
            self.sliceRange.valueChanged.connect(self.display)
            self.save_button.setEnabled(True)

    @pyqtSlot()
    def save_clicked(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as...",
                                              filter='*.txt')
        self.volume_data.export(path)


    @pyqtSlot()
    def dist_click(self):
        if self.im is None:
            QMessageBox.warning(self, "Warning", "Please load an image first",
                                QMessageBox.Ok)
        elif self.dist is None:
            self.statusBar().showMessage("Calculating distance tranform...")
            Thread(target=self.start_dist).start()
        else:
            self.vtk.update2(self.dist, (0, 1))

    def start_dist(self):
        self.calc_dist.emit(distance_transform(self.im))


    def finish_dist(self, dist):
        self.dist = dist
        self.vtk.update2(self.dist, (0, 0.5))
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
        self.vtk.update2(
            self.volume_data.im[i[0]:i[1], i[2]:i[3], i[4]:i[5]], (0.05, 0.5))

    def skel_update(self, i):
        self.vtk.display(self.im[i[0]:i[1], i[2]:i[3], i[4]:i[5]],
                         self.skel[i[0]:i[1], i[2]:i[3], i[4]:i[5]],
                         self.nodes[i[0]:i[1], i[2]:i[3], i[4]:i[5]])

    def display(self, i):
        try:
            im = self.volume_data.im[i[0]:i[1], i[2]:i[3], i[4]:i[5]]
        except TypeError:
            im = None
        try:
            skel = self.volume_data.skel[i[0]:i[1], i[2]:i[3], i[4]:i[5]]
        except TypeError:
            skel = None
        try:
            nodes = self.volume_data.nodes[i[0]:i[1], i[2]:i[3], i[4]:i[5]]
        except TypeError:
            nodes = None
        try:
            term = self.volume_data.terminal[i[0]:i[1], i[2]:i[3], i[4]:i[5]]
        except TypeError:
            term = None
        try:
            node_mask = self.volume_data.node_mask[i[0]:i[1], i[2]:i[3],
                   i[4]:i[5]]
        except TypeError:
            node_mask = None
        self.vtk.display(im, skel, nodes, term, node_mask)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
