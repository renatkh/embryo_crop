# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'embryoCropUI.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui
import cropAPI, os
import myFunc
import numpy as np

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(533, 388)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.mainGridLayout = QtGui.QGridLayout()
        self.mainGridLayout.setObjectName(_fromUtf8("mainGridLayout"))
        self.z_Spin = QtGui.QSpinBox(self.centralwidget)
        self.z_Spin.setMinimum(1)
        self.z_Spin.setObjectName(_fromUtf8("z_Spin"))
        self.mainGridLayout.addWidget(self.z_Spin, 2, 0, 1, 2)
        self.time_Spin = QtGui.QSpinBox(self.centralwidget)
        self.time_Spin.setMinimum(1)
        self.time_Spin.setObjectName(_fromUtf8("time_Spin"))
        self.mainGridLayout.addWidget(self.time_Spin, 2, 2, 1, 1)
        self.openFile_Button = QtGui.QPushButton(self.centralwidget)
        self.openFile_Button.setObjectName(_fromUtf8("openFile_Button"))
        self.mainGridLayout.addWidget(self.openFile_Button, 0, 0, 1, 1)
        self.DIC_Spin = QtGui.QSpinBox(self.centralwidget)
        self.DIC_Spin.setMinimum(1)
        self.DIC_Spin.setObjectName(_fromUtf8("DIC_Spin"))
        self.mainGridLayout.addWidget(self.DIC_Spin, 2, 4, 1, 1)
        self.Z_Label = QtGui.QLabel(self.centralwidget)
        self.Z_Label.setObjectName(_fromUtf8("Z_Label"))
        self.mainGridLayout.addWidget(self.Z_Label, 1, 0, 1, 2)
        self.correctAtt_Check = QtGui.QCheckBox(self.centralwidget)
        self.correctAtt_Check.setObjectName(_fromUtf8("correctAtt_Check"))
        self.mainGridLayout.addWidget(self.correctAtt_Check, 3, 4, 1, 1)
        self.Ch_label = QtGui.QLabel(self.centralwidget)
        self.Ch_label.setObjectName(_fromUtf8("Ch_label"))
        self.mainGridLayout.addWidget(self.Ch_label, 1, 3, 1, 1)
        self.Bkgd_horizontalLayout = QtGui.QHBoxLayout()
        self.Bkgd_horizontalLayout.setObjectName(_fromUtf8("Bkgd_horizontalLayout"))
        self.removeBG_Check = QtGui.QCheckBox(self.centralwidget)
        self.removeBG_Check.setObjectName(_fromUtf8("removeBG_Check"))
        self.Bkgd_horizontalLayout.addWidget(self.removeBG_Check)
        self.customize_Check = QtGui.QCheckBox(self.centralwidget)
        self.customize_Check.setObjectName(_fromUtf8("customize_Check"))
        self.Bkgd_horizontalLayout.addWidget(self.customize_Check)
        self.mainGridLayout.addLayout(self.Bkgd_horizontalLayout, 3, 2, 1, 2)
        self.comboBox = QtGui.QComboBox(self.centralwidget)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.comboBox.addItem(_fromUtf8(""))
        self.mainGridLayout.addWidget(self.comboBox, 4, 0, 3, 2)
        self.fileName_Line = QtGui.QLineEdit(self.centralwidget)
        self.fileName_Line.setObjectName(_fromUtf8("fileName_Line"))
        self.mainGridLayout.addWidget(self.fileName_Line, 0, 1, 1, 4)
        self.correctDrift_Check = QtGui.QCheckBox(self.centralwidget)
        self.correctDrift_Check.setObjectName(_fromUtf8("correctDrift_Check"))
        self.mainGridLayout.addWidget(self.correctDrift_Check, 3, 0, 1, 2)
        self.T_Label = QtGui.QLabel(self.centralwidget)
        self.T_Label.setObjectName(_fromUtf8("T_Label"))
        self.mainGridLayout.addWidget(self.T_Label, 1, 2, 1, 1)
        self.DIC_label = QtGui.QLabel(self.centralwidget)
        self.DIC_label.setObjectName(_fromUtf8("DIC_label"))
        self.mainGridLayout.addWidget(self.DIC_label, 1, 4, 1, 1)
        self.channel_Spin = QtGui.QSpinBox(self.centralwidget)
        self.channel_Spin.setMinimum(1)
        self.channel_Spin.setMaximum(5)
        self.channel_Spin.setObjectName(_fromUtf8("channel_Spin"))
        self.mainGridLayout.addWidget(self.channel_Spin, 2, 3, 1, 1)
        self.run_Button = QtGui.QPushButton(self.centralwidget)
        self.run_Button.setObjectName(_fromUtf8("run_Button"))
        self.mainGridLayout.addWidget(self.run_Button, 9, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.resolution_Spin = QtGui.QDoubleSpinBox(self.centralwidget)
        self.resolution_Spin.setSingleStep(0.01)
        self.resolution_Spin.setProperty("value", 0.21)
        self.resolution_Spin.setObjectName(_fromUtf8("resolution_Spin"))
        self.horizontalLayout.addWidget(self.resolution_Spin)
        self.resolution_Label = QtGui.QLabel(self.centralwidget)
        self.resolution_Label.setObjectName(_fromUtf8("resolution_Label"))
        self.horizontalLayout.addWidget(self.resolution_Label)
        self.mainGridLayout.addLayout(self.horizontalLayout, 7, 0, 2, 2)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.correctAtt_Spin = QtGui.QDoubleSpinBox(self.centralwidget)
        self.correctAtt_Spin.setMaximum(1.0)
        self.correctAtt_Spin.setMinimum(0.01)
        self.correctAtt_Spin.setValue(0.1)
        self.correctAtt_Spin.setSingleStep(0.01)
        self.correctAtt_Spin.setObjectName(_fromUtf8("correctAtt_Spin"))
        self.verticalLayout.addWidget(self.correctAtt_Spin)
        spacerItem = QtGui.QSpacerItem(88, 37, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.mainGridLayout.addLayout(self.verticalLayout, 4, 4, 5, 1)
        self.featureSize_verticalLayout = QtGui.QVBoxLayout()
        self.featureSize_verticalLayout.setObjectName(_fromUtf8("featureSize_verticalLayout"))
        self.horizontalLayout_1 = QtGui.QHBoxLayout()
        self.horizontalLayout_1.setObjectName(_fromUtf8("horizontalLayout_1"))
        self.featureSize1_Label = QtGui.QLabel(self.centralwidget)
        self.featureSize1_Label.setEnabled(True)
        self.featureSize1_Label.setObjectName(_fromUtf8("featureSize1_Label"))
        self.horizontalLayout_1.addWidget(self.featureSize1_Label)
        self.featureSize1_Spin = QtGui.QSpinBox(self.centralwidget)
        self.featureSize1_Spin.setEnabled(True)
        self.featureSize1_Spin.setMaximum(300)
        self.featureSize1_Spin.setSingleStep(2)
        self.featureSize1_Spin.setObjectName(_fromUtf8("featureSize1_Spin"))
        self.horizontalLayout_1.addWidget(self.featureSize1_Spin)
        self.featureSize_verticalLayout.addLayout(self.horizontalLayout_1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.featureSize2_Label = QtGui.QLabel(self.centralwidget)
        self.featureSize2_Label.setEnabled(True)
        self.featureSize2_Label.setObjectName(_fromUtf8("featureSize2_Label"))
        self.horizontalLayout_2.addWidget(self.featureSize2_Label)
        self.featureSize2_Spin = QtGui.QSpinBox(self.centralwidget)
        self.featureSize2_Spin.setEnabled(True)
        self.featureSize2_Spin.setMaximum(300)
        self.featureSize2_Spin.setSingleStep(2)
        self.featureSize2_Spin.setObjectName(_fromUtf8("featureSize2_Spin"))
        self.horizontalLayout_2.addWidget(self.featureSize2_Spin)
        self.featureSize_verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.featureSize3_Label = QtGui.QLabel(self.centralwidget)
        self.featureSize3_Label.setEnabled(True)
        self.featureSize3_Label.setObjectName(_fromUtf8("featureSize3_Label"))
        self.horizontalLayout_3.addWidget(self.featureSize3_Label)
        self.featureSize3_Spin = QtGui.QSpinBox(self.centralwidget)
        self.featureSize3_Spin.setEnabled(True)
        self.featureSize3_Spin.setMaximum(300)
        self.featureSize3_Spin.setSingleStep(2)
        self.featureSize3_Spin.setObjectName(_fromUtf8("featureSize3_Spin"))
        self.horizontalLayout_3.addWidget(self.featureSize3_Spin)
        self.featureSize_verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.featureSize4_Label = QtGui.QLabel(self.centralwidget)
        self.featureSize4_Label.setEnabled(True)
        self.featureSize4_Label.setObjectName(_fromUtf8("featureSize4_Label"))
        self.horizontalLayout_5.addWidget(self.featureSize4_Label)
        self.featureSize4_Spin = QtGui.QSpinBox(self.centralwidget)
        self.featureSize4_Spin.setEnabled(True)
        self.featureSize4_Spin.setMaximum(300)
        self.featureSize4_Spin.setSingleStep(2)
        self.featureSize4_Spin.setObjectName(_fromUtf8("featureSize4_Spin"))
        self.horizontalLayout_5.addWidget(self.featureSize4_Spin)
        self.featureSize_verticalLayout.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.featureSize5_Label = QtGui.QLabel(self.centralwidget)
        self.featureSize5_Label.setEnabled(True)
        self.featureSize5_Label.setObjectName(_fromUtf8("featureSize5_Label"))
        self.horizontalLayout_8.addWidget(self.featureSize5_Label)
        self.featureSize5_Spin = QtGui.QSpinBox(self.centralwidget)
        self.featureSize5_Spin.setEnabled(True)
        self.featureSize5_Spin.setMaximum(300)
        self.featureSize5_Spin.setSingleStep(2)
        self.featureSize5_Spin.setObjectName(_fromUtf8("featureSize5_Spin"))
        self.horizontalLayout_8.addWidget(self.featureSize5_Spin)
        self.featureSize_verticalLayout.addLayout(self.horizontalLayout_8)
        spacerItem1 = QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.featureSize_verticalLayout.addItem(spacerItem1)
        self.mainGridLayout.addLayout(self.featureSize_verticalLayout, 4, 2, 5, 2)
        self.gridLayout.addLayout(self.mainGridLayout, 0, 0, 1, 1)
        self.resolution_Label.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusBar = QtGui.QStatusBar(MainWindow)
        self.statusBar.setObjectName(_fromUtf8("statusBar"))
        MainWindow.setStatusBar(self.statusBar)
        
        self.featureSpins = [self.featureSize1_Spin, self.featureSize2_Spin, self.featureSize3_Spin, self.featureSize4_Spin, self.featureSize5_Spin]
        self.featureLabels = [self.featureSize1_Label, self.featureSize2_Label, self.featureSize3_Label, self.featureSize4_Label, self.featureSize5_Label]
        self.defaultFeatureValue = 201
        self.initialSetup()

        self.retranslateUi(MainWindow)
        self.connectUI()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.openFile_Button.setText(_translate("MainWindow", "Open", None))
        self.Z_Label.setText(_translate("MainWindow", "Z", None))
        self.correctAtt_Check.setText(_translate("MainWindow", "Correct Attenuation", None))
        self.Ch_label.setText(_translate("MainWindow", "C", None))
        self.removeBG_Check.setText(_translate("MainWindow", "Remove Bkgd", None))
        self.customize_Check.setText(_translate("MainWindow", "Customize", None))
        self.comboBox.setItemText(0, _translate("MainWindow", "czt", None))
        self.comboBox.setItemText(1, _translate("MainWindow", "zct", None))
        self.correctDrift_Check.setText(_translate("MainWindow", "Correct Drift", None))
        self.T_Label.setText(_translate("MainWindow", "T", None))
        self.DIC_label.setText(_translate("MainWindow", "DIC", None))
        self.run_Button.setText(_translate("MainWindow", "Run", None))
        self.resolution_Label.setText(_translate("MainWindow", "um/pix", None))
        self.featureSize1_Label.setText(_translate("MainWindow", "Feature Size Ch X", None))
        self.featureSize2_Label.setText(_translate("MainWindow", "Feature Size Ch 2", None))
        self.featureSize3_Label.setText(_translate("MainWindow", "Feature Size Ch 3", None))
        self.featureSize4_Label.setText(_translate("MainWindow", "Feature Size Ch 4", None))
        self.featureSize5_Label.setText(_translate("MainWindow", "Feature Size Ch 5", None))
    
    def initialSetup(self):
        for feature in self.featureSpins:
            feature.setValue(self.defaultFeatureValue)
        self.noBckgdChecked()
        self.correctAttClicked(False)
        self.run_Button.setEnabled(False)
        
#         self.z_Spin.setValue(15)
#         self.time_Spin.setValue(3)
#         self.channel_Spin.setValue(3)
#         self.DIC_Spin.setValue(3)
#         self.fileName_Line.setText('/home/renat/Documents/work/development/test/Well005.tif')
        
# #         self.z_Spin.setValue(1)
# #         self.time_Spin.setValue(17)
# #         self.channel_Spin.setValue(3)
# #         self.DIC_Spin.setValue(1)
# #         self.fileName_Line.setText('/home/renat/Documents/work/development/test/TESTME_BGLI141_1B_t2.tif')
#         
# #         self.z_Spin.setValue(5)
# #         self.time_Spin.setValue(7)
# #         self.channel_Spin.setValue(3)
# #         self.DIC_Spin.setValue(3)
# #         self.fileName_Line.setText('/home/renat/Documents/work/development/test/OD3465-0002.tif_Files/OD3465-0002_z0t0c0.tif')
#         self.run_Button.setEnabled(True)
# 
# #         self.z_Spin.setValue(5)
# #         self.time_Spin.setValue(7)
# #         self.channel_Spin.setValue(3)
# #         self.DIC_Spin.setValue(3)
# #         self.fileName_Line.setText('/home/renat/Documents/work/development/test/OD3465-0003_2x2.tif_Files/OD3465-0003_2x2_z0t0c0.tif')
# #         self.resolution_Spin.setValue(0.42)
# #         self.run_Button.setEnabled(True)0
    
    def connectUI(self):
        self.openFile_Button.clicked.connect(self.openFile)
        self.run_Button.clicked.connect(self.run)
        self.correctAtt_Check.toggled.connect(self.correctAttClicked)
        
        self.channel_Spin.valueChanged.connect(self.channelNumChanged)
        self.DIC_Spin.valueChanged.connect(self.DIC_changed)
        self.removeBG_Check.toggled.connect(self.removeBGClicked)
        self.customize_Check.toggled.connect(self.customBGClicked)
        for i in range(len(self.featureSpins)):
            self.featureSpins[i].valueChanged.connect(self.checkOdd)
        
    def openFile(self):
        fileFilter = "TIF (*.tif)"
        fileName = QtGui.QFileDialog.getOpenFileName(self.centralwidget, 'Open File', '/home/renat/Documents/work/development/Well005/', fileFilter)
        self.fileName_Line.setText(_translate("MainWindow", fileName, None))
        self.run_Button.setEnabled(True)
        
    def correctAttClicked(self, state):
        self.correctAtt_Spin.setEnabled(state)
        self.correctAtt_Spin.setVisible(state)
                
    def removeBGClicked(self, state):
        if state:
            self.customize_Check.setVisible(True)
            self.customize_Check.setEnabled(True)
            if self.channel_Spin.value()>1: self.showFeatureSize(0, True)
        else: self.noBckgdChecked()
    
    def customBGClicked(self, state):
        nCh = self.channel_Spin.value()
        if state:
            self.showFeatureSize(0, False)
            self.featureSize1_Label.setText("Feature Size Ch 1")
            for i in range(nCh):
                if i!= self.DIC_Spin.value()-1: self.showFeatureSize(i, True)
        else:
            self.hideAllFeatures()
            self.featureSize1_Label.setText("Feature Size Ch X")
            if self.channel_Spin.value()>1: self.showFeatureSize(0, True)
            
    def channelNumChanged(self, nCh):
        self.DIC_Spin.setMaximum(nCh)
        if self.customize_Check.isChecked():
            self.customBGClicked(False)
            self.customBGClicked(True)
    
    def DIC_changed(self):
        if self.customize_Check.isChecked():
            self.customBGClicked(False)
            self.customBGClicked(True)
       
    def hideAllFeatures(self):
        for i in range(len(self.featureLabels)):
            self.showFeatureSize(i, False)
    
    def noBckgdChecked(self):
        self.hideAllFeatures()
        self.customize_Check.setChecked(False)
        self.customize_Check.setEnabled(False)
        self.customize_Check.setVisible(False)
            
    def showFeatureSize(self, ch, bool):
        self.featureSpins[ch].setEnabled(bool)
        self.featureSpins[ch].setVisible(bool)
        self.featureLabels[ch].setVisible(bool)
            
    def checkOdd(self, val):
        if val%2==0:
            self.centralwidget.sender().setValue(val+1)
        
    def run(self):
        self.statusBar.showMessage('Running...')
        QtGui.QApplication.processEvents() 
        if self.customize_Check.isChecked():
            featureList = []
            for f in self.featureSpins:
                featureList.append(f.value())
        else:
            featureList = [self.featureSize1_Spin.value() for i in range(self.channel_Spin.value())]
        featureList[self.DIC_Spin.value()-1]=None
        imgs = self.openImage()
        try:
            allEmbs = cropAPI.cropEmbs(imgs, self.DIC_Spin.value()-1, self.correctDrift_Check.isChecked(),\
                         self.correctAtt_Check.isChecked(),self.correctAtt_Spin.value(), self.removeBG_Check.isChecked(), featureList, self.resolution_Spin.value())
            self.save(allEmbs)
        except Exception as inst:
            self.statusBar.showMessage('Error:'+str(inst))
        
    def openImage(self):
        path, nZ, nT, nCh, order = str(self.fileName_Line.text()), self.z_Spin.value(), self.time_Spin.value(),\
                                   self.channel_Spin.value(), str(self.comboBox.currentText())
        if os.path.isfile(path):
            ims = myFunc.loadImTif(path)
            if len(ims.shape)==2:
                path = '/'.join(path.split('/')[:-1])+'/'
                ims = myFunc.loadImFolder(path)
        elif os.path.isdir(path): ims = myFunc.loadImFolder(path)
        else: raise('Error: wrong path {0}'.format(path))
        if len(ims.shape)>3: ims = np.reshape(ims,(-1,ims.shape[-2],ims.shape[-1]))
        if np.array(ims).size==nZ*nT*nCh*ims[0].size:
            if order=='czt': ims = np.reshape(ims, (nT, nZ, nCh, ims[0].shape[0], ims[0].shape[1])).astype(np.float)
            else:
                ims = np.reshape(ims, (nT, nCh, nZ, ims[0].shape[0], ims[0].shape[1])).astype(np.float)
                ims = np.swapaxes(ims, 1, 2)
        else:
            self.statusBar.showMessage('Error: number of images (or sizes) does not correspond to z={0}, t={1}, ch={2}'.format(nZ, nT, nCh))
            raise Exception('Error: number of images (or sizes) does not correspond to z={0}, t={1}, ch={2}'.format(nZ, nT, nCh))
        return ims
    
    def save(self, allEmbs):
        path = '/'.join(str(self.fileName_Line.text()).split('/')[:-1])+'/'
        if not os.path.exists(path+'crop/'): myFunc.mkdir_p(path+'crop/')
        for i in range(len(allEmbs)):
            self.statusBar.showMessage('saving Embryo {}'.format(i+1))
            myFunc.saveImagesMulti(allEmbs[i].astype(np.uint16), path+'crop/Emb{0:02}.tif'.format(i))
        self.statusBar.showMessage('embryos saved')

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

