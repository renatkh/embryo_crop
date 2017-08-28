'''
Created on Aug 23, 2017

@author: renat
API for cropping program
'''

import os
import myFunc
import numpy as np
import tkMessageBox
from findEmbryo import showIm, getMaskStak, findEmbsonIm, cropRotate, getAP, getAP2
from AttenuationCorrection import correctAttAll, corrAttMultiCh
import findEmbryo


def cropEmbs(ims, dicCh, corrDrift, corrAtt, attVal, removeBkgd, featureList, resolution, EmbdScreen=False):
    findEmbryo.RES_SCALE=0.21/resolution
    z, ch = ims.shape[1:3]
    if corrDrift: ims = correctDrift4AllC(ims, dicCh)
    allEmbsIms, rs = cropAllC(ims,0,dicCh,corrDrift, corrAtt, attVal, removeBkgd, featureList, EmbdScreen)
    if corrAtt:
        result = []
        for embImgs in allEmbsIms:
            result.append(corrAttMultiCh(embImgs, z, ch, dicCh, 0, attVal))
        return result
    else:  return allEmbsIms
    
def cropAllC(imgs, tCrop, dicCh, corrDrift, corrAtt, attVal, removeBkgd, featureList, EmbdScreen):
    '''
    Crops all images based on the embryos found in the first central plane of dic channel
    
    Parameters:
    imgs : multidimensional array of images
    tCrop: timepoint for cropping
    
    Return:
    list of images for each embryo
    '''
    z = imgs.shape[1]
    if removeBkgd:
        imtmp=np.zeros_like(imgs)
        for ch in range(imgs.shape[2]):
            if featureList[ch] is None: imtmp[:,:,ch]=imgs[:,:,ch]
            else:
                for i in range(imgs.shape[0]):
                    for j in range(imgs.shape[1]):
                        imtmp[i,j,ch]=myFunc.removeBG(imgs[i,j,ch], featureList[ch])
        imgs = imtmp
    im3 = np.reshape(imgs[:,:,dicCh],(-1,imgs.shape[-2],imgs.shape[-1]))
    ims8b = np.uint8(255.*(im3[tCrop*z:(tCrop+1)*z] - np.min(im3[tCrop*z:(tCrop+1)*z]))/np.max(im3[tCrop*z:(tCrop+1)*z]))

    mask = getMaskStak(ims8b)
    eParams = findEmbsonIm(mask)
    j=0
    allEmbIms, aspRatio = [], []
    for params in eParams:
        print('cropping Embryo={0}'.format(j+1))
        if EmbdScreen:
            im1 = np.reshape(imgs[:,:,0],(-1,imgs.shape[-2],imgs.shape[-1]))
            im2 = np.reshape(imgs[:,:,1],(-1,imgs.shape[-2],imgs.shape[-1]))
            flip = checkAPRotation(im1,im2,im3, EmbdScreen, corrAtt, attVal, params,z)
            del im1, im2
        else: flip=False
        imAll = []
        for im in np.reshape(imgs,(-1,imgs.shape[-2],imgs.shape[-1])):
            imAll.append(cropRotate((im, params, flip)))
        imAll = np.array(imAll)
        allEmbIms.append(imAll)
        (a,b), center, angle = params
        aspRatio.append(1.*a/b)
        j+=1
        del imAll
    del im3, ims8b
    return allEmbIms, aspRatio

def correctDrift4AllC(ims, dicCh):
    '''
    Correct drift for all images based on central plane of the third image sequence
    
    Parameters:
    imgs : tuple of three lists of images
    
    Return:
    tuple of three lists of corrected images
    '''
    
    centralPlane = ims[:,ims.shape[1]/2,dicCh]
    drift = myFunc.findDrift(centralPlane)
    drift4All = []
    for d in drift:
        for i in range(ims.shape[1]):
            drift4All.append(d)
    imsCorr = np.zeros_like(ims)
    for ch in range(ims.shape[2]):
        imsChXCorr = myFunc.correctDrift(np.reshape(ims[:,:,ch],(-1,ims[0,0,0].shape[0],ims[0,0,0].shape[1])),drift4All)
        imsCorr[:,:,ch] = np.reshape(imsChXCorr, ims[:,:,0].shape)
    print('drift corrected')
    return imsCorr

def checkAPRotation(im1,im2,im3, strain, corrAtt, attVal, params,z):
    apR = []
    for k in [6,8,10]:
        if corrAtt: imtmp = cropRotate([myFunc.maxIntensProject(correctAttAll(im2[k*z:(k+1)*z],z,attVal)),params, False])
        else: imtmp = cropRotate([myFunc.maxIntensProject(im2[k*z:(k+1)*z]),params, False])
        apR.append(getAP(imtmp))
    if np.mean(apR)<0.8:flip = True
    elif np.mean(apR)>1.25: flip = False
    elif strain=='GLS':
        apR = []
        for k in [8,10,12]:
            if corrAtt:
                imtmp1 = cropRotate([myFunc.maxIntensProject(correctAttAll(im1[k*z:(k+1)*z],z,attVal)),params, False])
                imtmp2 = cropRotate([myFunc.maxIntensProject(correctAttAll(im2[k*z:(k+1)*z],z,attVal)),params, False])
            else:
                imtmp1 = cropRotate([myFunc.maxIntensProject(im1[k*z:(k+1)*z]),params, False])
                imtmp2 = cropRotate([myFunc.maxIntensProject(im2[k*z:(k+1)*z]),params, False])
            apR.append(getAP2(imtmp1, imtmp2))
    if np.mean(apR)<1: flip = True
    else: flip = False
    return flip

def checkEmbDebris(im):
    ''' lets user debug supplied image, and returns 1 for yes (save) and 0 for not and 2 for special case'''
    code = showIm(im)
    if code == ord('d') or code == ord('D'):
        result = tkMessageBox.askquestion("Delete", "Are You Sure?", icon='warning')
        if result == 'yes':
            return 0
        else:
            return checkEmbDebris(im)
    elif code == ord('x') or code == ord('X'): return 2
    else: return 1

def saveEmb(imgs, point, i, well, check, r):
    '''
    Saves embryo images according to a certain pattern
    
    Parameters:
    imgs: tuple of 3 lists of images for each channel
    f: point visit number
    i: embryo number
    r: aspect ratio
    '''
    
    im1, im2, im3 = imgs
    strain = strains[well]
    ri = RNAi[well]
    if ri!='EMBD0000': folderOut = loadFolder + 'cropped/{0}/{1}/'.format(ri,strain) #outputFolder
    else: folderOut = loadFolder + 'cropped/EMBD0000/{0}/{1}/'.format(strain,date) #outputFolder
    if check == 2 :folderOut = folderOut+'x'
    else: folderOut = folderOut
    j = 1
    folder = folderOut+ 'Emb{0}/'.format(j)
    while os.path.exists(folder):
        fileName = glob.glob(folder+'*_T01_Z01_C1.tif')
        if len(fileName)>0:
            fileName = fileName[0].split('/')[-1]
            if fileName.split('_')[2]==date:
                if i==1: break
                else: i -= 1
            j += 1
            folder = folderOut+ 'Emb{0}/'.format(j)
        else:
            j += 1
            folder = folderOut+ 'Emb{0}/'.format(j)
    fileName = '{0}_Emb{1}_{2}_W{3:0>2}F{4}_'.format(ri,j,date,well+1,point+1)
    
    ''' correct attenuation and save local '''
    if not os.path.exists(folder): os.makedirs(folder)
    else:
        print('file exist, clearing folder', folder)
        clearFolder(folder)
        if os.path.exists(folder+'batch-output'.format(splitSym)):
            shutil.rmtree(folder+'batch-output'.format(splitSym))
    if attCorrect:
        im1CA = correctAttAll(im1,z)
        im2CA = correctAttAll(im2,z)
    else:
        im1CA = im1
        im2CA = im2
    saveAllIms(im1CA, im2CA, im3, folder, fileName)
    saveImgs(im1CA, folder, fileName,1)
    saveImgs(im2CA, folder, fileName,2)
    saveImgs(im3, folder, fileName,3)
        
    ''' populate aspect ratio file '''
    print('saveEmb aspect j=',j)
    addAspect(r,date, ri, j)

if __name__ == '__main__':
    pass