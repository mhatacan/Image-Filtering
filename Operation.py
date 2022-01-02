###########################################
# Filename    :  Operation.py
# Author      :  Muhammet Harun ATACAN
# Date        :  13.06.2021
# Description :  Operation Code
###########################################

from PyQt5 import QtWidgets, QtCore
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap

import matplotlib.pyplot as plt

from skimage import img_as_float, filters
from skimage.segmentation import chan_vese, morphological_chan_vese, checkerboard_level_set
from skimage.color import rgb2hsv, rgb2gray
from skimage.util import compare_images
from skimage import io

import numpy as np

class Operation(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(Operation,self).__init__()
        loadUi("Interface.ui",self)
        
        self.makebutonconnection()
        self.fdir = 0
        self.fdir_2 = 0
        self.ctr = np.zeros(2)
    
    def makebutonconnection(self):
        ## File menu
        self.File_Open_Source.triggered.connect(self.OpenFile)
        self.File_Save_Output.triggered.connect(self.SaveOutput)
        self.File_Save_As_Output.triggered.connect(self.SaveAsOutput)
        self.File_ExportAs_Source.triggered.connect(self.ExportAsSource)
        self.File_ExportAs_Output.triggered.connect(self.ExportAsOutput)
        self.File_Exit.triggered.connect(self.ExitFunction)
        ## Edit menu
        self.Edit_Clear_Source.triggered.connect(self.Clear_Source)
        self.Edit_Clear_Output.triggered.connect(self.Clear_Output)
        self.Edit_Undo_Output.triggered.connect(self.UndoOutput)
        self.Edit_Redo_Output.triggered.connect(self.RedoOutput)
        ## Conversion menu
        self.Conversion_RGB_to_Grayscale.triggered.connect(self.RGB_to_Gray)
        self.Conversion_RGB_to_HSV.triggered.connect(self.RGB_to_HSV_filter)
        ## Segmentation menu
        self.Segmentation_Multi_Otsu_Thresholding.triggered.connect(self.Multi_Otsu_Thresholding)
        self.Segmentation_Chan_Vese_Segmentation.triggered.connect(self.Chan_Vese_Segmentation)
        self.Segmentation_Morphological_Snakes.triggered.connect(self.Morphological_Snakes)
        ## EdgeDetection menu
        self.EdgeDetection_Roberts.triggered.connect(self.Roberts)
        self.EdgeDetection_Sobel.triggered.connect(self.Sobel)
        self.EdgeDetection_Scharr.triggered.connect(self.Scharr)
        self.EdgeDetection_Prewitt.triggered.connect(self.Prewitt)
        ## Source box
        self.File_OpenSource_button.clicked.connect(self.OpenFile)
        self.File_ExportAs_Source_button.clicked.connect(self.ExportAsSource)
        self.Edit_Clear_Source_button.clicked.connect(self.Clear_Source)
        ## Output box
        self.File_Save_Output_button.clicked.connect(self.SaveOutput)
        self.File_SaveAs_Output_button.clicked.connect(self.SaveAsOutput)
        self.File_ExportAs_Output_button.clicked.connect(self.ExportAsOutput)
        self.Edit_Clear_Output_button.clicked.connect(self.Clear_Output)
        self.Edit_Undo_Output_button.clicked.connect(self.UndoOutput)
        self.Edit_Redo_Output_button.clicked.connect(self.RedoOutput)
        ## Conversion box
        self.Conversion_RGB_to_Grayscale_button.clicked.connect(self.RGB_to_Gray)
        self.Conversion_RGB_to_HSV_button.clicked.connect(self.RGB_to_HSV_filter)
        ## Segmentation box
        self.Segmentation_Multi_Otsu_Thresholding_button.clicked.connect(self.Multi_Otsu_Thresholding)
        self.Segmentation_Chan_Vese_Segmentation_button.clicked.connect(self.Chan_Vese_Segmentation)
        self.Segmentation_Morphological_Snakes_button.clicked.connect(self.Morphological_Snakes)
        ## EdgeDetection box
        self.EdgeDetection_Roberts_button.clicked.connect(self.Roberts)
        self.EdgeDetection_Sobel_button.clicked.connect(self.Sobel)
        self.EdgeDetection_Scharr_button.clicked.connect(self.Scharr)
        self.EdgeDetection_Prewitt_button.clicked.connect(self.Prewitt)
        
    def Controller(self):
        if(self.ctr[0]!=0):
            self.File_ExportAs_Source.setEnabled(True)
            self.File_ExportAs_Source_button.setEnabled(True)
            
            self.Edit_Clear_Source.setEnabled(True)
            self.Edit_Clear_Source_button.setEnabled(True)
            
            self.Conversion_RGB_to_Grayscale.setEnabled(True)
            self.Conversion_RGB_to_HSV.setEnabled(True)
            
            self.Segmentation_Multi_Otsu_Thresholding.setEnabled(True)
            self.Segmentation_Chan_Vese_Segmentation.setEnabled(True)
            self.Segmentation_Morphological_Snakes.setEnabled(True)
            
            self.EdgeDetection_Roberts.setEnabled(True)
            self.EdgeDetection_Sobel.setEnabled(True)
            self.EdgeDetection_Scharr.setEnabled(True)
            self.EdgeDetection_Prewitt.setEnabled(True)
            
            self.Conversion_RGB_to_Grayscale_button.setEnabled(True)
            self.Conversion_RGB_to_HSV_button.setEnabled(True)
            
            self.Segmentation_Multi_Otsu_Thresholding_button.setEnabled(True)
            self.Segmentation_Chan_Vese_Segmentation_button.setEnabled(True)
            self.Segmentation_Morphological_Snakes_button.setEnabled(True)
            
            self.EdgeDetection_Roberts_button.setEnabled(True)
            self.EdgeDetection_Sobel_button.setEnabled(True)
            self.EdgeDetection_Scharr_button.setEnabled(True)
            self.EdgeDetection_Prewitt_button.setEnabled(True)
        if(self.ctr[1]!=0):
            
            self.File_Save_Output.setEnabled(True)
            self.File_Save_As_Output.setEnabled(True)
            self.File_ExportAs_Output.setEnabled(True)
            self.File_Exit.setEnabled(True)
            
            
            self.Edit_Clear_Output.setEnabled(True)
            self.Edit_Undo_Output.setEnabled(True)
            self.Edit_Redo_Output.setEnabled(True)
            
            self.File_OpenSource_button.setEnabled(True)
            self.File_ExportAs_Source_button.setEnabled(True)
            self.Edit_Clear_Source_button.setEnabled(True)
            
            self.File_Save_Output_button.setEnabled(True)
            self.File_SaveAs_Output_button.setEnabled(True)
            self.File_ExportAs_Output_button.setEnabled(True)
            self.Edit_Clear_Output_button.setEnabled(True)
            self.Edit_Undo_Output_button.setEnabled(True)
            self.Edit_Redo_Output_button.setEnabled(True)
            
    def OpenFile(self):
        self.fdir = QtWidgets.QFileDialog.getOpenFileName(self,'Open File',r'C:\Users\pc\Desktop\151220152073',"Image files (*.jpg *.png)")
        
        self.Source_PixMap.setPixmap(QPixmap(self.fdir[0]))
        self.ctr[0] += 1
        self.Controller()
        
    def SaveOutput(self):
        self.fdir_2 = QtWidgets.QFileDialog.getSaveFileName(self,'Save File',r'C:\Users\pc\Desktop\151220152073',"Image files (*.jpg *.png)")
        
        plt.imsave((self.fdir_2[0]),io.imread(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
    def SaveAsOutput(self):
        self.fdir_2 = QtWidgets.QFileDialog.getSaveFileName(self,'Save File',r'C:\Users\pc\Desktop\151220152073',"Image files (*.jpg *.png)")
        
        plt.imsave((self.fdir_2[0]),io.imread(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
    
    def ExportAsSource(self):
        if self.fdir[0].endswith('jpg'):
            self.fdir_2 = QtWidgets.QFileDialog.getSaveFileName(self,'Save File',r'C:\Users\pc\Desktop\151220152073',"Image files (*.png)")
        elif self.fdir[0].endswith('png'):
            self.fdir_2 = QtWidgets.QFileDialog.getSaveFileName(self,'Save File',r'C:\Users\pc\Desktop\151220152073',"Image files (*.jpg)")
        
        plt.imsave((self.fdir_2[0]),io.imread(self.fdir[0]))
        
    def ExportAsOutput(self):
        if self.fdir[0].endswith('jpg'):
            self.fdir_2 = QtWidgets.QFileDialog.getSaveFileName(self,'Save File',r'C:\Users\pc\Desktop\151220152073',"Image files (*.png)")
        elif self.fdir[0].endswith('png'):
            self.fdir_2 = QtWidgets.QFileDialog.getSaveFileName(self,'Save File',r'C:\Users\pc\Desktop\151220152073',"Image files (*.jpg)")
        
        plt.imsave((self.fdir_2[0]),io.imread(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
    
    def ExitFunction(self):
        QtCore.QCoreApplication.quit()
    
    def Clear_Source(self):
        self.Source_PixMap.clear()
        
    def Clear_Output(self):
        self.Output_PixMap.clear()
    
    def UndoOutput(self):
        print("İşlem Başarısız..")
    
    def RedoOutput(self):
        print("İşlem Başarısız..")
        
    def RGB_to_HSV_filter(self):
        rgb_img = io.imread(self.fdir[0])
        hsv_img = rgb2hsv(rgb_img)
        hue_img = hsv_img[:, :, 0]
        value_img = hsv_img[:, :, 2]
        hue_threshold = 0.04
        binary_img = hue_img > hue_threshold
        value_threshold = 0.10
        binary_img = (hue_img > hue_threshold) | (value_img < value_threshold)
        
        plt.imsave('asdf.png', binary_img)
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()
        
    def RGB_to_Gray(self):
        original = io.imread(self.fdir[0])
        grayscale = rgb2gray(original)
        
        plt.imsave('asdf.png', grayscale, cmap=plt.cm.gray)
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()
    
    def Multi_Otsu_Thresholding(self):
        image = io.imread(self.fdir[0])
        thresholds = filters.threshold_multiotsu(image)
        regions = np.digitize(image, bins=thresholds)
        
        plt.imsave('asdf.png', regions, cmap='jet')
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()
        
    def Chan_Vese_Segmentation(self):
        xxyy = io.imread(self.fdir[0])
        image = img_as_float(xxyy)
        cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200, dt=0.5, init_level_set="checkerboard", extended_output=True)
        
        plt.imsave('asdf.png', cv[1], cmap="gray")
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()
        
    def Morphological_Snakes(self):
        xxyy = io.imread(self.fdir[0])
        def store_evolution_in(lst):
            def _store(x):
                lst.append(np.copy(x))
            return _store
        image = img_as_float(xxyy)
        init_ls = checkerboard_level_set(image.shape, 6)
        evolution = []
        callback = store_evolution_in(evolution)
        ls = morphological_chan_vese(image, 35, init_level_set=init_ls, smoothing=3, iter_callback=callback)
        
        plt.imsave('asdf.png', ls, cmap="gray")
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()
    
    def Roberts(self):
        image = io.imread(self.fdir[0])
        edge_roberts = filters.roberts(image)
        
        plt.imsave('asdf.png',edge_roberts, cmap=plt.cm.gray)
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()
        
    def Sobel(self):
        image = io.imread(self.fdir[0])
        edge_sobel = filters.sobel(image)
        
        plt.imsave('asdf.png',edge_sobel, cmap=plt.cm.gray)
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()
        
    def Scharr(self):
        x, y = np.ogrid[:300, :300]
        image_rot = np.exp(1j * np.hypot(x, y) ** 1.3 / 20.).real
        edge_scharr = filters.scharr(image_rot)
        
        plt.imsave('asdf.png',edge_scharr, cmap=plt.cm.gray)
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller() 
        
    def Prewitt(self):
        x, y = np.ogrid[:300, :300]
        image_rot = np.exp(1j * np.hypot(x, y) ** 1.3 / 20.).real
        edge_sobel = filters.sobel(image_rot)
        edge_scharr = filters.scharr(image_rot)
        edge_prewitt = filters.prewitt(image_rot)
        diff_scharr_prewitt = compare_images(edge_scharr, edge_prewitt)
        diff_scharr_sobel = compare_images(edge_scharr, edge_sobel)
        max_diff = np.max(np.maximum(diff_scharr_prewitt, diff_scharr_sobel))
        
        plt.imsave('asdf.png',diff_scharr_prewitt, cmap=plt.cm.gray, vmax=max_diff)
        self.Output_PixMap.setPixmap(QPixmap(r'C:\Users\pc\Desktop\151220152073\asdf.png'))
        
        self.ctr[1] = 99
        self.Controller()