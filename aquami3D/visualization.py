# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import vtk
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.stats import lognorm, norm
from scipy import ndimage
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import Qt

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from inout import *

class MainWindow(Qt.QMainWindow):

    def __init__(self, parent = None):
        Qt.QMainWindow.__init__(self)
        self.vtkwidget = VtkWindow(self)
        self.setCentralWidget(self.vtkwidget)

def get_edges(im):
    # Takes a 3D volume and returns just the edge pixels that boarder 0 values.
    struct = ndimage.generate_binary_structure(3, 3)
    mask = im > 0
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    out = np.zeros(im.shape)
    out[edges] = 255
    return out

class VtkWindow(QtWidgets.QWidget):

    def __init__(self, parent = None):
        super(VtkWindow, self).__init__(parent)
        self.parent = parent
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.layout.addWidget(self.vtkWidget)
        self.volume = None
        self.volumeSkel = None
        self.ren = vtk.vtkRenderer()
        self.renderWin = self.vtkWidget.GetRenderWindow()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()


    def display(self, im=None, skel=None, nodes=None, term=None,
                node_mask=None):
        if self.volume is not None:
            self.ren.RemoveVolume(self.volume)

        if im is not None:
            matrix = np.zeros(im.shape)
        elif skel is not None:
            matrix = np.zeros(skel.shape)
        elif nodes is not None:
            matrix = np.zeros(nodes.shape)
        elif term is not None:
            matrix = np.zeros(term.shape)
        elif node_mask is not None:
            matrix = np.zeros(node_mask.shape)
        else:
            print('ERROR, NO IMAGES PASSED')
            return

        if im is not None:
            matrix = get_edges(im)
        if node_mask is not None:
            node_mask = get_edges(node_mask)
            matrix[node_mask != 0] = 251
        if skel is not None:
            matrix[skel != 0] = 253
        if nodes is not None:
            matrix[nodes != 0] = 254
        if term is not None:
            matrix[term != 0] = 252

        # Change the volume numpy matrix to a VTK-image
        matrix = matrix.astype(np.uint16)
        dataImporter = vtk.vtkImageImport()
        data_string = matrix.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedShort()  # uint16
        dataImporter.SetNumberOfScalarComponents(1)  # grayscale

        # Describe how the data is stored and the dimensions of the array.
        w, h, d = matrix.shape
        dataImporter.SetDataExtent(0, d - 1, 0, h - 1, 0, w - 1)
        dataImporter.SetWholeExtent(0, d - 1, 0, h - 1, 0, w - 1)

        # Create color data from a few color points.
        colorFunc = vtk.vtkColorTransferFunction()
        colorFunc.AddRGBPoint(255, 1.0, 1.0, 1.0)
        colorFunc.AddRGBPoint(254, 0.0, 1.0, 1.0)
        colorFunc.AddRGBPoint(253, 1.0, 0.0, 1.0)
        colorFunc.AddRGBPoint(252, 1.0, 0.0, 0.0)
        colorFunc.AddRGBPoint(251, 0.0, 1.0, 0.0)
        colorFunc.AddRGBPoint(0,   0.0, 0.0, 0.0)

        # Transparency function. 0 is clear, 1 is opaque
        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        alphaChannelFunc.AddPoint(255, 0.05)
        alphaChannelFunc.AddPoint(254, 1)
        alphaChannelFunc.AddPoint(253, 1)
        alphaChannelFunc.AddPoint(252, 1)
        alphaChannelFunc.AddPoint(251, 0.05)
        alphaChannelFunc.AddPoint(0,   0)

        # Apply color and transparency volume properties.
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorFunc)
        volumeProperty.SetScalarOpacity(alphaChannelFunc)

        # This class describes how the volume is rendered (ray tracing).
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

        # Create Volume.Specify data, as well as how the data will be rendered.
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetMaximumImageSampleDistance(0.01)
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        # The class vtkVolume is used to pair the previously declared volume
        # as well as the properties to be used when rendering that volume.
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(volumeMapper)
        self.volume.SetProperty(volumeProperty)

        # Add the volume to the renderer.
        self.ren.AddVolume(self.volume)

        # Set background color to black.
        self.ren.SetBackground(0, 0, 0)

        # Set window size.
        self.renderWin.SetSize(500, 500)
        self.renderWin.SetMultiSamples(4)

        # A simple function to be called when the user quits the application.
        def exitCheck(obj, event):
            if obj.GetEventPending() != 0:
                obj.SetAbortRender(1)

                # Tell the application to use the function as an exit check.
                self.renderWin.AddObserver("AbortCheckEvent", exitCheck)

        self.iren.Initialize()
        self.renderWin.Render()
        self.iren.Start()


    def update2(self, matrix, alpha=(0,1)):
        """
        :param matrix: numpy array (uint8) containing data to be visualized.
        :param alpha:
        :return:
        """

        if self.volume is not None:
            self.ren.RemoveVolume(self.volume)

        # Change the volume numpy matrix to a VTK-image
        matrix = matrix.astype(np.uint16)
        dataImporter = vtk.vtkImageImport()
        data_string = matrix.tostring()
        dataImporter.CopyImportVoidPointer(data_string, len(data_string))
        dataImporter.SetDataScalarTypeToUnsignedShort() #uint16
        dataImporter.SetNumberOfScalarComponents(1) # grayscale

        # Describe how the data is stored and the dimensions of the array.
        w, h, d = matrix.shape
        dataImporter.SetDataExtent(0, d - 1, 0, h - 1, 0, w - 1)
        dataImporter.SetWholeExtent(0, d - 1, 0, h - 1, 0, w - 1)

        # Create color data from a few color points.
        colorFunc = vtk.vtkColorTransferFunction()
        colorFunc.AddRGBPoint(matrix.max(), 1.0, 1.0, 1.0)
        colorFunc.AddRGBPoint(matrix.min(), 0.0, 0.0, 0.0)

        # Transparency function. 0 is clear, 1 is opaque
        alphaChannelFunc = vtk.vtkPiecewiseFunction()
        alphaChannelFunc.AddPoint(matrix.max(), alpha[1])
        alphaChannelFunc.AddPoint(matrix.min(), alpha[0])

        # Apply color and transparency volume properties.
        volumeProperty = vtk.vtkVolumeProperty()
        volumeProperty.SetColor(colorFunc)
        volumeProperty.SetScalarOpacity(alphaChannelFunc)

        # This class describes how the volume is rendered (ray tracing).
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()

        # Create Volume.Specify data, as well as how the data will be rendered.
        volumeMapper = vtk.vtkVolumeRayCastMapper()
        volumeMapper.SetMaximumImageSampleDistance(0.01)
        volumeMapper.SetVolumeRayCastFunction(compositeFunction)
        volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

        # The class vtkVolume is used to pair the previously declared volume
        # as well as the properties to be used when rendering that volume.
        self.volume = vtk.vtkVolume()
        self.volume.SetMapper(volumeMapper)
        self.volume.SetProperty(volumeProperty)


        # Add the volume to the renderer.
        self.ren.AddVolume(self.volume)

        # Set background color to black.
        self.ren.SetBackground(0, 0, 0)

        # Set window size.
        self.renderWin.SetSize(500, 500)
        self.renderWin.SetMultiSamples(4)

        # A simple function to be called when the user quits the application.
        def exitCheck(obj, event):
            if obj.GetEventPending() != 0:
                obj.SetAbortRender(1)

        # Tell the application to use the function as an exit check.
                self.renderWin.AddObserver("AbortCheckEvent", exitCheck)

        self.iren.Initialize()
        self.renderWin.Render()
        self.iren.Start()

    def show_skel(self, skel, im=None):
        """
        :param matrix: numpy array (uint8) containing data to be visualized.
        :param alpha:
        :return:
        """

        if self.volume is not None:
            self.ren.RemoveVolume(self.volume)
        if self.volumeSkel is not None:
            self.ren.RemoveVolume(self.volumeSkel)

        # Setup image volume
        if im is not None:
            # Change the volume numpy matrix to a VTK-image
            im = im.astype(np.uint16)
            dataImporter = vtk.vtkImageImport()
            data_string = im.tostring()
            dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            dataImporter.SetDataScalarTypeToUnsignedShort() #uint16
            dataImporter.SetNumberOfScalarComponents(1) # grayscale
            # Describe how the data is stored and the dimensions of the array.
            w, h, d = im.shape
            dataImporter.SetDataExtent(0, d - 1, 0, h - 1, 0, w - 1)
            dataImporter.SetWholeExtent(0, d - 1, 0, h - 1, 0, w - 1)
            # Create color data from a few color points.
            colorFunc = vtk.vtkColorTransferFunction()
            colorFunc.AddRGBPoint(im.max(), 0.75, 0.75, 0.75)
            colorFunc.AddRGBPoint(im.min(), 0.0, 0.0, 0.0)
            # Transparency function. 0 is clear, 1 is opaque
            alphaChannelFunc = vtk.vtkPiecewiseFunction()
            alphaChannelFunc.AddPoint(im.max(), 0.10)
            alphaChannelFunc.AddPoint(im.min(), 0)
            # Gradient function. Decreases opacity at non interfaces
            gradientFunc = vtk.vtkPiecewiseFunction()
            gradientFunc.AddPoint(0,   0.0)
            gradientFunc.AddPoint(100, 1.0)
            # Apply color and transparency volume properties.
            volumeProperty = vtk.vtkVolumeProperty()
            volumeProperty.SetColor(colorFunc)
            volumeProperty.SetScalarOpacity(alphaChannelFunc)
            volumeProperty.SetGradientOpacity(gradientFunc)
            # This class describes how the volume is rendered (ray tracing).
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
            # Create Volume.Specify data, as well as how the data will be rendered.
            volumeMapper = vtk.vtkVolumeRayCastMapper()
            volumeMapper.SetMaximumImageSampleDistance(0.01)
            volumeMapper.SetVolumeRayCastFunction(compositeFunction)
            volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
            # The class vtkVolume is used to pair the previously declared volume
            # as well as the properties to be used when rendering that volume.
            self.volume = vtk.vtkVolume()
            self.volume.SetMapper(volumeMapper)
            self.volume.SetProperty(volumeProperty)
            # Add the volume to the renderer.
            self.ren.AddVolume(self.volume)

        # Setup skeletal
        if skel is not None:
            # Change the volume numpy matrix to a VTK-image
            skel = skel.astype(np.uint16)
            dataImporter = vtk.vtkImageImport()
            data_string = skel.tostring()
            dataImporter.CopyImportVoidPointer(data_string, len(data_string))
            dataImporter.SetDataScalarTypeToUnsignedShort() #uint16
            dataImporter.SetNumberOfScalarComponents(1) # grayscale
            # Describe how the data is stored and the dimensions of the array.
            w, h, d = skel.shape
            dataImporter.SetDataExtent(0, d - 1, 0, h - 1, 0, w - 1)
            dataImporter.SetWholeExtent(0, d - 1, 0, h - 1, 0, w - 1)
            # Create color data from a few color points.
            colorFunc = vtk.vtkColorTransferFunction()
            colorFunc.AddRGBPoint(skel.max(), 1.0, 0.0, 0.0)
            colorFunc.AddRGBPoint(skel.min(), 0.0, 0.0, 0.0)
            # Transparency function. 0 is clear, 1 is opaque
            alphaChannelFunc = vtk.vtkPiecewiseFunction()
            alphaChannelFunc.AddPoint(skel.max(), 1)
            alphaChannelFunc.AddPoint(skel.min(), 0)
            # Apply color and transparency volume properties.
            volumeProperty = vtk.vtkVolumeProperty()
            volumeProperty.SetColor(colorFunc)
            volumeProperty.SetScalarOpacity(alphaChannelFunc)
            # This class describes how the volume is rendered (ray tracing).
            compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
            # Create Volume.Specify data, as well as how the data will be rendered.
            volumeMapper = vtk.vtkVolumeRayCastMapper()
            volumeMapper.SetMaximumImageSampleDistance(0.01)
            volumeMapper.SetVolumeRayCastFunction(compositeFunction)
            volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
            # The class vtkVolume is used to pair the previously declared volume
            # as well as the properties to be used when rendering that volume.
            self.volumeSkel = vtk.vtkVolume()
            self.volumeSkel.SetMapper(volumeMapper)
            self.volumeSkel.SetProperty(volumeProperty)
            # Add the volume to the renderer.
            self.ren.AddVolume(self.volumeSkel)

        
        
        

        # Set background color to black.
        self.ren.SetBackground(0, 0, 0)

        # Set window size.
        self.renderWin.SetSize(500, 500)
        self.renderWin.SetMultiSamples(4)

        # A simple function to be called when the user quits the application.
        def exitCheck(obj, event):
            if obj.GetEventPending() != 0:
                obj.SetAbortRender(1)

        # Tell the application to use the function as an exit check.
                self.renderWin.AddObserver("AbortCheckEvent", exitCheck)

        self.iren.Initialize()
        self.renderWin.Render()
        self.iren.Start()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(None)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data, xaxis):
        self.axes.clear()
        # find number bins
        u = len(np.unique(data))
        numBins = int(2 * u ** (1 / 2))
        if numBins < 4:
            numBins = len(np.unique(data))
            if numBins < 1:
                numBins = 1

        n, bins, patches = self.axes.hist(data, bins=numBins, density=1, edgecolor='black')
        gfit = norm.fit(data.flatten())
        gauss_plot = norm.pdf(bins, gfit[0], gfit[1])
        self.axes.plot(bins, gauss_plot, 'r--', linewidth=1, label='gaussian')
        self.axes.set_xlabel(xaxis)
        self.draw()


class Plot(QtWidgets.QFrame):

    def __init__(self, parent=None):
        self.parent = parent
        QtWidgets.QFrame.__init__(self)
        layout = QtWidgets.QVBoxLayout()
        self.plot_canvas = PlotCanvas()
        layout.addWidget(self.plot_canvas)
        self.setLayout(layout)

    def plot(self, values, xaxis):
        self.parent.setGeometry(200, 200, 1200, 600)
        self.setVisible(True)
        self.plot_canvas.plot(values, xaxis)


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
