'''
Created on Oct 24, 2014

@author: Admin
'''

import matplotlib.pyplot as plt
import mahotas
from myMath import *
from findEmbryo import showIm
import cv2
import scipy.signal as signal
import glob, os
import numpy as np

def correctAttenuation(stack, firstIm=0, lastImageAt=0.1):
    '''corrects attenuation of an image stack with first image not attenuated and last attenuated by lastImageAt:= Intensity last/ Intensity first'''
    res = stack[:firstIm+1] #+1 because first image is not modified
    l = len(stack)-firstIm
    for i in range(firstIm+1,len(stack)):
        im = (stack[i]*(1.+(1./lastImageAt-1.)*(i-firstIm)/(l-1)))
        im[np.where(im>65535)]=65535
        res.append(im.astype(np.uint16))
    return res

def getMaxIntensity(stack):
    return np.array([getIntensity(im) for im in stack])

def getIntensity(im):
#     im[245:270,480:505]=0
#     noise = mahotas.thresholding.otsu(im)
#     imBi = np.zeros_like(im)
#     imBi[np.where(im>noise)]=1
#     plt.imshow(im)
#     plt.show()
    tmp = np.sort(im.ravel())
    return np.mean(tmp[-50:])

def correctAttAll(stack, z):
    imOut = []
    for i in range(len(stack)/z):
        imOut += correctAttenuation(stack[i*z:(i+1)*z])
    return imOut

def loadCropped():
    z=18
    folder = 'C:\\Users\\Admin\\Desktop\\EMBD_Files\\cropped\\'
    for path, subdirs, files in os.walk(folder):
        pSplit = path.split('\\')
        if pSplit[-1][:3] =='Emb' or pSplit[-1][:3] =='xEm':
            print(pSplit[-1])
            aPath = '\\'.join([t if t!='cropped' else 'attenuated' for t in pSplit])
            if not os.path.exists(aPath): os.makedirs(aPath)
            im = []
            fileNames = glob.glob(path+'\\*C1.tif')
            fileNames.sort()
            for fileName in fileNames:
                im.append(cv2.imread(fileName, -1))
            im = correctAttAll(im,z)
            for i in range(len(fileNames)):
                name = fileNames[i].split('\\')
                name = [t if t!='cropped' else 'attenuated' for t in name ]
                res = '\\'.join(name)
                cv2.imwrite(res, im[i])

            im = []
            fileNames = glob.glob(path+'\\*C2.tif')
            fileNames.sort()
            for fileName in fileNames:
                im.append(cv2.imread(fileName, -1))
            im = correctAttAll(im,z)
            for i in range(len(fileNames)):
                name = fileNames[i].split('\\')
                name = [t if t!='cropped' else 'attenuated' for t in name ]
                res = '\\'.join(name)
                cv2.imwrite(res, im[i])
             
            im = []
            fileNames = glob.glob(path+'\\*C3.tif')
            fileNames.sort()
            for fileName in fileNames:
                im.append(cv2.imread(fileName, -1))
            for i in range(len(fileNames)):
                name = fileNames[i].split('\\')
                name = [t if t!='cropped' else 'attenuated' for t in name ]
                res = '\\'.join(name)
                cv2.imwrite(res, im[i])

if __name__ == '__main__':
    loadCropped()
    
    
#     z = 18 #number of z planes
#     folderIn = 'C:\\Users\\Admin\\Desktop\\EMBD_Files\\tmp\\20140319T140206'
#     folder = 'C:\\Users\\Admin\\Desktop\\EMBD_Files\\attenuated\\EMBD0000\\GLS\\20140319T140206\\Emb1\\'
#     fileName = 'EMBD0000_Emb1_20140319T140206_W01F1T03Z12C1.tif'
#     im1, im2, im3 = loadImages(folderIn, 1, 1)
#     im1 = [removeBG(im) for im in im1]
#     im2 = [removeBG(im) for im in im2]
#     att = []
#     slopes = []
#     inters = []
#     
# #     for im in im1:
# #         im = cv2.imread(folder+fileName)
# #         removeBG(im)
#     for i in range(len(im1)/z):
#         skip = 11
#         maxs = getMaxIntensity(im1[i*z:(i+1)*z])
#         slope, inter, se, ie = lineFit(np.arange(maxs.size)[skip:15],maxs[skip:15])
#         slopes.append(slope)
#         inters.append(inter)
#         att.append((inter+slope*17)/(inter+slope*0))
#         plt.plot(np.arange(maxs.size),maxs)
#     x = np.arange(maxs.size)
#     slope, inter = np.mean(slopes),np.mean(inters)
#     print(np.mean(att), (inter+slope*17)/(inter+slope*0))
# #     plt.plot(x, np.mean(slopes)*x+np.mean(inters))
#     plt.show()
         