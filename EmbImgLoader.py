'''
Created on Jun 16, 2014

@author: renat 

Loads embryo images generated by CV1000 and crops/orients and saves separate embryos based on DIC (C3) central image
'''
 
date = '20161130T120748' 

loadFolder = 'Z:/'
folderIn = loadFolder + 'CV1000/' + date #Input folder
trackingFile = 'Z:/Experiment_tracking_sheets/EMBD_fileNames_Tracking_Sheet.csv'
z = 18 #number of z planes
corrDrift = True
removeBG = True
attCorrect = True
apRotate = True
nWells = 14#number of wells (14)
pointVisits = 4# number of point visits (4)


import cv2, glob, Image, time, csv, shutil
import numpy as np
import Tkinter
import tkMessageBox
from driftCorrect import * 
from findEmbryo import *
from myFunc import *
import multiprocessing as mp
from itertools import repeat
from AttenuationCorrection import correctAttAll
import numpy as np

debug=False
splitSym = '/'
RNAi, strains = [], [] #RNA condition, strain condition

def removeBG(im, well):
    im = np.float32(im) - 3000
    im[np.where(im<0)]=0
    im = np.uint16(im)
    if strains[well]!='MS':
        Gsize = 41
    else:Gsize = 201
    karnel = np.ones([Gsize,Gsize])
    karnel = 1.0*karnel/sum(karnel)
    imBG = cv2.GaussianBlur(im, (Gsize,Gsize), 0)
    imTmp = np.float32(im)-np.float32(imBG)
    imTmp[np.where(imTmp<0)]=0
    if np.max(imTmp)==0:
        print('zero image')
        imTmp= np.uint16(imTmp)
    elif np.max(imTmp)>65535:
        print('large value image')
        imTmp= np.uint16(imTmp)
        showIm(im, 'original')
        showIm(imTmp, 'subtracted')
    imTmp= np.uint16(imTmp)
    return imTmp

def getConditions(date, fileName):
    ''' loads RNAi strains for a specified date from a csv file '''
    global RNAi, strains
#     csvFile = csv.reader(open(fileName, 'rb'), delimiter=',')
    csvFile = csv.reader(open(fileName, 'rU'), delimiter=',') #universal
    fileData=[]
    for row in csvFile:
        fileData.append(row[1:-1])
    myDate = [s for s in fileData if s[0]==date]
    myDate = sorted(myDate, key=lambda well: well[2])
    RNAi = [s[3] for s in myDate]
    strains =  [s[4] for s in myDate]
    return

def loadImages(folder, well, j):
    '''
    loads images from a folder and splits them in separate point visits
    
    Parameters:
    folder : folder to read images from
    
    Return:
    allImgs: list of 4 different point visits with 3 channels in each. images are numpy arrays.
    '''
    imc1, imc2, imc3 = [], [], []
    folderNames = glob.glob(folder+'/Well{0:0>3}/*F{1:0>3}*C1.tif'.format(well,j))
    folderNames.sort()
    for fileName in folderNames:
        imc1.append(cv2.imread(fileName, -1))
        
    folderNames = glob.glob(folder+'/Well{0:0>3}/*F{1:0>3}*C2.tif'.format(well,j))
    folderNames.sort()
    for fileName in folderNames:
        imc2.append(cv2.imread(fileName, -1))
        
    folderNames = glob.glob(folder+'/Well{0:0>3}/*F{1:0>3}*C3.tif'.format(well,j))
    folderNames.sort()
    for fileName in folderNames:
        imc3.append(cv2.imread(fileName, -1))
        
    allImgs=(imc1, imc2, imc3)
    return allImgs

# def reassignWell(date):
#     if int(date[:7]) > 20160202:
#         pass     

def getPlane(imgs, z, j):
    '''
    outputs image list of only specified plane
    '''
    res = [imgs[i*z+j] for i in range(len(imgs)/z)]
    return res

def correctDrift4AllC(imgs):
    '''
    Correct drift for all images based on central plane of the third image sequence
    
    Parameters:
    imgs : tuple of three lists of images
    
    Return:
    tuple of three lists of corrected images
    '''
    im1, im2, im3 = imgs
    centralPlane = getPlane(im3, z, z/2)
    drift = findDrift(centralPlane)
    drift4All = []
    for d in drift:
        for i in range(z):
            drift4All.append(d)
    im1c = correctDrift(im1,drift4All)
    im2c = correctDrift(im2,drift4All)
    im3c = correctDrift(im3,drift4All)
    corrected = [im1c, im2c, im3c]
    print('drift corrected')
    return corrected

def cropAllC(imgs, well):
    '''
    Crops all images based on the embryos found in the first central plane of C3
    
    Parameters:
    imgs : tuple of three lists of images of different channels
    
    Return:
    list of tuples of three channels for each embryo
    '''
    tCrop = 3
    if corrDrift: imgs = correctDrift4AllC(imgs)
    im1, im2, im3 = imgs
    if removeBG:
        imtmp=[]
        for im in im1:
            imtmp.append(removeBG(im, well))
        im1=imtmp
        imtmp=[]
        for im in im2:
            imtmp.append(removeBG(im, well))
        im2=imtmp
        del imgs, imtmp
    ims8b = np.uint8(255.*(im3[tCrop*z:(tCrop+1)*z] - np.min(im3[tCrop*z:(tCrop+1)*z]))/np.max(im3[tCrop*z:(tCrop+1)*z]))
#       
#     r = np.max(1.*(im2[tCrop*z:(tCrop+1)*z] - np.min(im2[tCrop*z:(tCrop+1)*z]))/np.max(im2[tCrop*z:(tCrop+1)*z]),axis=0)
#     g = np.max(1.*(im1[tCrop*z:(tCrop+1)*z] - np.min(im1[tCrop*z:(tCrop+1)*z]))/np.max(im1[tCrop*z:(tCrop+1)*z]),axis=0)
#     b = np.zeros_like(r)
#     fig = myFigure()
#     fig.imshow(np.dstack([r,g,b]), colorbar=False)
#     fig.noAxis()
#     fig.noClip()
#     fig.show()
    
    mask = getMaskStak(ims8b)
    if np.sum(mask)/255>0.5*mask.size:
        mask = getMask(a16a8(im3[tCrop*z+z/2]), True)
        print('cropAllC, too much dirt')
    eParams = findEmbsonIm(mask)
    j=0
    allEmb, aspRatio = [], []
    flip = False
    pool = mp.Pool(processes=10)
    for params in eParams:
        print('well {0}, cropping Embryo={1}'.format(well+1, j+1))
        if apRotate:
            apR = []
            for k in [6,8,10]:
                if attCorrect: imtmp = cropRotate([maxIntensProject(correctAttAll(im2[k*z:(k+1)*z],z)),params, False])
                else: imtmp = cropRotate([maxIntensProject(im2[k*z:(k+1)*z]),params, False])
                apR.append(getAP(imtmp))
            if np.mean(apR)<0.8:flip = True
            elif np.mean(apR)>1.25: flip = False
            elif strains[well]=='GLS':
                apR = []
                for k in [8,10,12]:
                    if attCorrect:
                        imtmp1 = cropRotate([maxIntensProject(correctAttAll(im1[k*z:(k+1)*z],z)),params, False])
                        imtmp2 = cropRotate([maxIntensProject(correctAttAll(im2[k*z:(k+1)*z],z)),params, False])
                    else:
                        imtmp1 = cropRotate([maxIntensProject(im1[k*z:(k+1)*z]),params, False])
                        imtmp2 = cropRotate([maxIntensProject(im2[k*z:(k+1)*z]),params, False])
                    apR.append(getAP2(imtmp1, imtmp2))
            if np.mean(apR)<1: flip = True
            else: flip = False
        imAll = pool.map(cropRotate, zip(im1+im2+im3, repeat(params), repeat(flip)))
        length = len(im1)
        emb = (imAll[0:length],imAll[length:2*length],imAll[2*length:3*length])
        allEmb.append(emb)
        (a,b), center, angle = params
        aspRatio.append(1.*a/b)
        j+=1
        del emb, imAll
    del im1, im2, im3
    pool.close()
    pool.join()
    return allEmb, aspRatio

def getAllEmb(folder):
    '''
    Finds and saves all embryos
    
    ParametersL
    folder: folder to load embryos
    
    Return:
    None
    '''
    global RNAi, strains
    
    print('STARTED!!!')
    getConditions(date, trackingFile)
    totalEmb = 0
    embs=[]
    
    for well in range(nWells):
        for j in range(pointVisits):
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------------------------------------------#
            if debug: well, j = 8, 3
            imgs = loadImages(folder, well+1, j+1)
            if len(imgs[0])>0:
                print('loaded well {0} point {1}'.format(well+1, j+1))
                es, rs = cropAllC(imgs, well)
                embs.append([well, j, es, rs])
                totalEmb+=len(embs[-1])-2
            else: print('no images well {0} point {1}'.format(well+1, j+1))
            del imgs
        
    print('Done analyzing, ready to save!')
    checks = np.array([checkEmbDebris(e[2][3*z+z/2]) for well, j, es, rs in embs for e in es])
    totalEmb = checks.size
    uniqeRNAi = np.array(list(set(RNAi)))
    embN = np.ones([len(uniqeRNAi),2])
    xembN = np.ones([len(uniqeRNAi),2])
    if not debug:
        for well,j, es, rs in embs:
            if strains[well]!='MS':k=0
            else: k=1
            for l in range(len(es)):
                e=es[l]
                r=rs[l]
                print('{0} embryos left, saving...'.format(totalEmb))
                i = checks.size-totalEmb
                if checks[i]==1:
                    saveEmb(e,j,int(embN[np.where(uniqeRNAi==RNAi[well])[0],k]), well, checks[i],r)
                    embN[np.where(uniqeRNAi==RNAi[well])[0],k]+=1
                elif checks[i]==2:
                    saveEmb(e,j,int(xembN[np.where(uniqeRNAi==RNAi[well])[0],k]), well, checks[i],r)
                    xembN[np.where(uniqeRNAi==RNAi[well])[0],k]+=1
                totalEmb-=1



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
    
def saveAllIms(im1, im2, im3, folder, fileName):
    imAll = np.column_stack((im1, im2, im3))
    imAll = np.reshape(imAll, (-1, im1[0].shape[0], im1[0].shape[1]))
 

def saveImgs(imgs,folder, fileName, c):
    for i in range(len(imgs)):
        cv2.imwrite(folder+fileName+'T{0:0>2}_Z{1:0>2}_C{2}.tif'.format( i/z+1, i%z+1, c), imgs[i])

def addAspect(r, date, ri, j):
    '''
    Adds aspect ratio of an embryo into the csv file
    
    Parameters:
    r: aspect ratio
    date: date
    ri: RNAi conditions
    j: embryo number
    '''
    
    import operator
    fileName = loadFolder+'cropped/'+'aspects.csv'
    newData = []
    oldData = loadAspects(fileName)
    i=0
    while i<len(oldData):
        if oldData[i][0]!=ri: newData.append(oldData[i])
        elif oldData[i][1]!=date: newData.append(oldData[i])
        elif int(oldData[i][2])<j: newData.append(oldData[i])
        i+=1
    newData.append([ri, date, '{0:0>3}'.format(j), str(r)])
    newData = sorted(newData, key=operator.itemgetter(1, 2, 3))
    saveAspects(fileName, newData)

def loadAspects(fileName):
    '''
    reads aspect ratios from file. The aspect ratios are sorted by rnai condition, date, embryo number.
    fileName: name of the file to read from
    '''
    fileData = []
    try:
        csvFile = csv.reader(open(fileName, 'rU'), delimiter=',') #universal
        for row in csvFile:
            fileData.append(row)
    except: pass
    return fileData

def saveAspects(fileName, data):
    with open(fileName, 'wb') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(data)

if __name__ == '__main__':
    getAllEmb(folderIn)
    print('ALL DONE!!!!! :)')
