'''
Created on Jun 15, 2014

@author: renat

Collection of various commonly used functions
'''

import os, numpy, cv2, re, glob
import numpy as np
import shutil
from tifffile import imsave, imread
import scipy.ndimage.filters as filters

def sort_nicely(l):

    """ Sort the given list in the way that humans expect.
    """
    
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    
    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
            "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('([0-9]+)', s) ]

    l.sort(key=alphanum_key)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def clearFolder(folder, subFolder=False):
    if os.path.exists(folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif subFolder and os.path.isdir(file_path): shutil.rmtree(file_path)
            except:
                pass
    else: os.mkdir(folder)

def a16a8(im, imMin=None, imMax=None):
    im = im.astype(np.float)
    if np.max(im)>0:
        if imMax is None: imMax = np.max(im[np.where(im>0)])
        if imMin is None: imMin = np.min(im[np.where(im>0)])
        if imMax>0 and imMin==imMax: imMin=0
        if imMax>0 and imMax!=imMin: im[np.where(im>0)] = (im[np.where(im>0)]-imMin)/(imMax-imMin)
    im = np.uint8(255*im)
    return im

def maxIntensProject(imList):
    ''' Maximum intensity projection.
    INPUT:
    imList: list of images in numpy array form
    OUTPUT:
    single image of the size of input images
    '''
    
    return numpy.max(imList, axis=0)

def showIm(img, title='image'):
    #show image
    if np.max(img)>255: cv2.imshow(title, a16a8(img))
    else: cv2.imshow(title, img)
    code = cv2.waitKey()
    cv2.destroyAllWindows()
    return code

def blurImList(imList, rad):
    import scipy.signal as signal
    kernelSize = rad
    karnel = numpy.ones([kernelSize,kernelSize])
    karnel = 1.0*karnel/sum(karnel)
#     imList = [a16a8(signal.convolve2d(im,karnel,'same')/kernelSize) for im in imList]
    imList = [a16a8(filters.gaussian_filter(im,kernelSize)) for im in imList]
    return imList

def correctDrift(imArray, shift):
    '''
    corrects drift
    
    Parameters:
    imArray : list of N images as numpy arrays
    shift: List of shifts of the image (tuple)
    
    Return:
    res : list of N images as numpy arrays
    '''

    res = []
    for i in range(len(imArray)):
        im = numpy.float32(imArray[i])
        x,y = shift[i]
        mapy, mapx = numpy.mgrid[0:im.shape[0],0:im.shape[1]].astype(np.float32)
        mapx = mapx-1.*x
        mapy = mapy-1.*y
        im = cv2.remap(im, mapx, mapy, interpolation=cv2.INTER_LINEAR)
        res.append(im.astype(imArray[0].dtype))
    return res

def findDrift(imArray):
    Gsize = 1
    res = [(0,0)]
    x, y = 0, 0
    karnel = np.ones([Gsize,Gsize])
    karnel = 1.0*karnel/sum(karnel)
    im1 = np.float32(cv2.GaussianBlur(a16a8(imArray[0]), (Gsize,Gsize), 0))
    for i in range(1,len(imArray)):
        im2 = np.float32(cv2.GaussianBlur(a16a8(imArray[i]), (Gsize,Gsize), 0))
        xt, yt= cv2.phaseCorrelate(im1,im2)
        if np.sqrt(xt**2+yt**2)>50: xt, yt =0., 0.
        x-=xt
        y-=yt
        res.append((x,y))
        im1=im2
    return res

def correctAttenuation(stack, firstIm=0, lastImageAt=0.1):
    '''corrects attenuation of an image stack with first image not attenuated and last attenuated by lastImageAt:= Intensity last/ Intensity first'''
    res = stack[:firstIm+1] #+1 because first image is not modified
    l = len(stack)-firstIm
    for i in range(firstIm+1,len(stack)):
        im = (stack[i]*(1.+(1./lastImageAt-1.)*(i-firstIm)/(l-1)))
        im[numpy.where(im>65535)]=65535
        res.append(im.astype(numpy.uint16))
    return res

def loadImFolder(folder):
    imNames = glob.glob('{0}*.tif'.format(folder))
    if len(imNames)>0:
        sort_nicely(imNames)
        im = cv2.imread(imNames[0],-1)
        if len(im.shape)==2: images = np.array([im])
        else: images = im
        for name in imNames[1:]:
#             print('loading {}'.format(name))
            im = cv2.imread(name,-1)
            if im is not None and len(im.shape)==2: images=np.concatenate((images,np.array([im])))
            else: images=np.concatenate((images,im))
        return images
    else:
        print('Msg: Can not find images in {0}'.format(folder))
        return None

def loadImTif(fileName):
    return imread(fileName)

def saveImages(imgs, filePrefix, folder):
    '''
    saves images
    Input:
    imgs: list of images as numpy arrays
    filePrefix: string prefix to use in front of the file name
    folder: folder to save images into
    '''
    for i in range(len(imgs)):
        cv2.imwrite(folder+filePrefix+'{0:0>3}.tif'.format(i), imgs[i])
        
def saveImagesMulti(imgs, fileName):
    imsave(fileName, np.array(imgs))

def removeBG(im, Gsize):
    im = np.uint16(im)
    karnel = np.ones([Gsize,Gsize])
    karnel = 1.0*karnel/sum(karnel)
    imBG = cv2.GaussianBlur(im, (Gsize,Gsize), 0)
    imTmp = np.float32(im)-np.float32(imBG)
    imTmp[np.where(imTmp<0)]=0
    if np.max(imTmp)==0:
        print('zero image after removing Background')
        imTmp= np.uint16(imTmp)
    elif np.max(imTmp)>65535:
        print('large value image after removing Background')
    imTmp= np.uint16(imTmp)
    return imTmp

if __name__ == '__main__':
    pass
