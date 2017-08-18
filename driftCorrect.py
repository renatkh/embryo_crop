'''
Created on Apr 17, 2014

@author: renat
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label, morphology
from myFunc import correctDrift, a16a8, showIm, findDrift
from PIL import Image

def getMask(image, DIC = True):
    minEmbArea = 5000
#     kernel = numpy.ones((11,11),np.uint8)
    kernel = np.ones((21,21),np.uint8)
    if DIC:
        im = a16a8(image)
        edge = cv2.Canny(im,60,255)
        edgeExt = np.zeros(np.array(edge.shape)+60)
        edgeExt[30:-30,30:-30] = edge
        imt = cv2.morphologyEx(edgeExt, cv2.MORPH_CLOSE, kernel)
        imt = morphology.binary_fill_holes(imt       )
        imt = imt[30:-30, 30:-30]
        labeled,nr_objects = label(imt)
        area = np.array([np.sum(labeled==k) for k in range(1,nr_objects+1)])
        mask = np.uint8(np.zeros_like(labeled))
        for ind in np.where(area>=minEmbArea)[0]:
            mask = mask + 255*np.uint8(labeled==ind+1)
    return mask

def showMovie(imgs):
    for im in imgs:
        cv2.imshow('img', a16a8(im))
        cv2.waitKey(300)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    import time
    timeIni = time.clock()
    folder = 'D:/EMBD_Files/tmp/CV1000/20140904T131050/'
    fileName = 'C3-Well008.tif'
    z=18
    
#     folder = '/home/renat/Documents/work/imaging/shaohe/'
#     fileName = 'test2.tif'
#     z=1
    
    im = Image.open(folder+fileName)
    imArray = []
    i=0
    try:
        while True:
            im.seek(i*z+int(z/2))
            imArray.append(np.asarray(im))
            i+=1
    except EOFError:
        pass # end of sequence
    drift = findDrift(imArray)
    corrected = correctDrift(imArray, drift)
    print('time=', time.clock()-timeIni)
#     plt.imshow(corrected[0])
#     plt.figure()
#     plt.imshow(corrected[-1])
#     plt.show()
    showMovie(corrected)
#     showMovie(imArray)
        