'''
Created on Apr 1, 2014

@author: renat

fits embryos on a DIC image with an ellipse 
'''

import Image, numpy, cv2, glob,os
import matplotlib.pyplot as plt
from fitEllipse import *
import fitEllipse2
from scipy.ndimage import label, morphology
from driftCorrect import *
from myFigure import myFigure

debug=False
minEmbArea=5000

def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

class ColorMap:
    startcolor = ()
    endcolor = ()
    startmap = 0
    endmap = 0
    colordistance = 0
    valuerange = 0
    ratios = []    

    def __init__(self, startcolor, endcolor, startmap, endmap):
        self.startcolor = numpy.array(startcolor)
        self.endcolor = numpy.array(endcolor)
        self.startmap = float(startmap)
        self.endmap = float(endmap)
        self.valuerange = float(endmap - startmap)
        self.ratios = (self.endcolor - self.startcolor) / self.valuerange

    def __getitem__(self, value):
        color = tuple(self.startcolor + (self.ratios * (value - self.startmap)))
        return (int(color[0]), int(color[1]), int(color[2]))

def getContourPart(contour, indexStart, indexEnd):
    ''' returns part of the contour from indexStart to indexEnd '''
    newCont = []
    step = 1
    if indexEnd<indexStart: step = -1 #make sure that start is always smaller
    for i in range(indexStart, indexEnd, step):
        while i<0: i+=len(contour)
        while i>=len(contour): i-=len(contour) #if i went out of bounds, loop back
        newCont.append(contour[i,0])
    return numpy.array(newCont)

def getEllipse(contour, start, end):
    #Note: horizontal ellipse
    ''' returns parameters of an ellipse fitted to contour between start and end'''
    X = getContourPart(contour, start, end)
    hull = cv2.convexHull(X)
    hull = numpy.array([p[0] for p in hull])
    try: cPos, a, d, ang = fitEllipse2.fitellipse(hull,'linear')
    except RuntimeError: cPos, a, d, ang = (0,0), 0, 0, 0
    if a<d:
        a,d = d,a
        ang+= numpy.pi/2
    while ang>np.pi/2: ang-=np.pi
    while ang<-np.pi/2: ang+=np.pi
    return ((a,d),cPos, ang)

def showIm(img, title='image'):
    #show image
#     cv2.imshow(title, img)
#     code = cv2.waitKey()
#     cv2.destroyAllWindows()
#     return code
    fig = myFigure()
    fig.imshow(img, colorbar=False)
    fig.noAxis()
    fig.noClip()
    fig.title(title)
    fig.show()
    
def saveIm(img):
    i=0
    folder = '/home/renat/Documents/work/imaging/development/tmp/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    while os.path.exists(folder+'%03d.tif' % i):
        i+=1
    cv2.imwrite(folder+'%03d.tif' % i, img)

def showEllipse(contour, start, end, imArray):
    ''' displays an ellipse fitted to contour between start and end'''
    #draw ellipse fitted to the contour
#     cv2.drawContours(imArray,[X],-1, (155,0,0))
#     cPos, a, d, ang = fitEllipse2.fitellipse(X,'linear')
#     eParams = ((a,d),cPos, ang)
#     ellipse = create_ellipse(*eParams)
#     cv2.drawContours(imArray,[ellipse.astype(int)],-1, (200,0,0))
#     print(eParams)
    
    
    eParams = getEllipse(contour, start, end)
#     ellipse = create_ellipse(*eParams)
    
    # draw ellipse fitted to the convex contour
#     hull = cv2.convexHull(getContourPart(contour, start, end))
#     cv2.drawContours(imArray,[hull],-1, (155,0,0))
#     cv2.drawContours(imArray,[ellipse.astype(int)],-1, (100,0,0))
    
    ellipse = create_cassini_oval(*eParams)
    cv2.drawContours(imArray,[ellipse.astype(int)],-1, (155,0,0))
    
    #draw a circle at start point
    while start<0: start+=len(contour)
    while start>=len(contour): start-=len(contour)
    cv2.circle(imArray,tuple(contour[start][0]),5,[50,0,0],-1)
    #draw a circle at end point
    while end>=len(contour): end-=len(contour)
    cv2.circle(imArray,tuple(contour[end][0]),5,[100,0,0],-1)
    print('showEllipse, contour length = ', end-start, start, end)
    showIm(imArray)

def contToArray(contour):
    return numpy.array([point[0] for point in contour])

def findPointIndex(contour, point):
    if cv2.pointPolygonTest(contour,point,False)==0:
        for i in range(contour.size):
            if (contour[i]==point).all():return i

def findDefects(contour):
    distThreash = 5000
    hull = cv2.convexHull(contour,returnPoints = False)
    defects = cv2.convexityDefects(contour,hull)
    result = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        if d>distThreash:
            result.append(far)
    return result

def findArc(contour, startIni):
    start, end = growArcEnd(contour, startIni)
    start, end = growArcEnd(contour, end, start)
    return start, end

def growArcEnd(contour, start, end=None, defect=False):
#     print('growArcEnd, start, end', start, end)
    distInside = 15 #minimal distance for convex hull deviation from the contour (15)
    distSE = 10 #minimal distance between start and end of the deviation (50)
    stepSize = 5
    direction = 1
    if end is None: end = start+100
    if start > end: direction = -1 #the arc grow can be in any direction (in case end is smaller than start)
    stepSize = direction*stepSize
    endPrev = end
    f=None
    while abs(end-start)<400:
        subCont = getContourPart(contour, start, end)
        s,e,f,d = findDefect(subCont, direction)
        if (s is not None and d>distInside and np.linalg.norm(subCont[e]-subCont[s])>distSE) or abs(end-start)>=len(contour):
            break
        else:
            endPrev = end
            end+= stepSize 

    if f is not None: end = start+direction*f[0]
#     else: start, end = 0, end                                   
    if start > end: start, end = end, start
     
    
#     print('growArcEnd, after start, end', start, end)
#     imArray = np.zeros([512,512])
#     cv2.drawContours(imArray,[contour],-1, (200,0,0))
#     startt = start
#     while startt>len(contour): startt-=len(contour)
#     cv2.circle(imArray,tuple(contour[startt][0]),3,[50,0,0],-1)
#     endt = end
#     while endt>=len(contour): endt-=len(contour) 
#     cv2.circle(imArray,tuple(contour[endt][0]),7,[50,0,0],-1)
#     cv2.imshow('img', imArray)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
    
    return start, end

def findDefect(cont,direction):
    ''' direction of the contour, clockwise = 1, counterclockwise = -1'''
    hull = cv2.convexHull(cont,returnPoints = False)
    dist = []
    indHull = np.argmax(hull) #find position of the last contour point in the hull
    indHullPrev = indHull+direction #determine position of the second point based on derection of the contour (clockwise or counterclockwise)
    if indHullPrev==len(hull): indHullPrev-=len(hull) #make sure position loops back
    
    ''' find the deepest defect point '''
    for i in range(hull[indHullPrev],hull[indHull]):
        d = cv2.pointPolygonTest(np.array([cont[hull[indHullPrev]],cont[hull[indHull]]]),tuple(cont[i]),True)
        dist.append(abs(d))
    if len(dist)==0 or max(dist)==0: return None, None, None, None
    j = numpy.argmax(dist)

    ''' draw the first, last and deepest defect points '''
#     imArray = mask.copy()
#     imArray = np.zeros_like(mask.copy())
#     cv2.drawContours(imArray,np.array([cont]),-1, (255,0,0))
#     cv2.drawContours(imArray,np.array([cv2.convexHull(cont,returnPoints = True)]),-1, (155,0,0))
#     cv2.circle(imArray,tuple(cont[hull[indHullPrev]][0]),3,[200,0,0],-1)
#     cv2.circle(imArray,tuple(cont[hull[indHull]][0]),3,[100,0,0],-1)
#     cv2.drawContours(imArray,np.array([[cont[i]] for i in range(hull[indHullPrev],hull[indHullPrev])]),-1, (100,0,0))
#     cv2.circle(imArray,tuple(cont[hull[indHullPrev]+j][0]),3,[255,0,0],-1)
#     cv2.imshow('img', imArray)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
    
    return hull[indHullPrev], hull[indHull], hull[indHullPrev]+j, max(dist)

def getStart(cont, shape, side = 0):
    delta = 3
    contX = [point[0,0] for point in cont]
    contY = [point[0,1] for point in cont]
    left, right = np.argmin(contX), np.argmax(contX)
    top, bott = np.argmin(contY), np.argmax(contY)
    if contY[top]>delta and (side==1 or side==0):
        return top
    elif side==2 or (contX[right]<shape[0]-delta and side==0):
        return right
    elif side==3 or (contY[bott]<shape[1]-delta and side==0):
        return bott
    elif side==4 or (contX[left]>delta and side==0):
        return left
    else: return 0
    
def findEmbryo(im,side=0):
    embDimA, embDimB = 115.,65.
    imTmp = im.copy()
    contours, hierarchy = cv2.findContours(imTmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if side==0:
        quality = []
        for side in range(1,5): #[top, right, bottom, left]
            start = getStart(contours[0], imTmp.shape, side)
            start, end = findArc(contours[0],start)
            eParams = getEllipse(contours[0], start,end)
            a,b = eParams[0]
            if a>70 and b>40: quality.append(abs(embDimA-a)/embDimA+abs(embDimB-b)/embDimB)
            else: quality.append(100)
            if debug: print('findEmbryo', side, a, b, quality[-1])
        side = np.argmin(quality)+1
        if debug: print('findEmbryo, side=',side)
    
    start = getStart(contours[0], imTmp.shape, side)
    start, end = findArc(contours[0],start)
    eParams = getEllipse(contours[0], start,end)
    if debug: showEllipse(contours[0], start,end, im)
    if np.min(quality)<100: return True, eParams
    else: return False, eParams

def removeFromMask(im,eParams):
    ''' returns image with zeroed values inside the ellipse list (defined by eParams) and black image with white ellipse'''

    imTmp = im.copy() #copy image
    imCut = np.zeros_like(im) #create mask to remove from image
    for params in eParams:
        (a,d),cPos, ang = params #get ellipse parameters
        params = (a+1,d+1),cPos, ang #add extra pix To make sure that all of the embryo is cut out.
        ellipse = create_ellipse(*params) #creates ellipse points
        ellipse = numpy.array([[[int(point[0]),int(point[1])]] for point in ellipse]) #converts points into numpy array
        bbox = np.array(cv2.boundingRect(ellipse)) #determine ellipse bounding rectangle to reduce area for checking
        ''' fix box boundaries to be within image '''
        if bbox[0]<0:
            bbox[2]+=bbox[0]
            bbox[0]=0
        if bbox[1]<0:
            bbox[3]+=bbox[1]
            bbox[1]=0
        if bbox[0]+bbox[2]>=im.shape[0]:
            bbox[2] = im.shape[0]-bbox[0]-1
        if bbox[1]+bbox[3]>=im.shape[1]:
            bbox[3] = im.shape[1]-bbox[1]-1
            
        for i in range(bbox[0],bbox[0]+bbox[2]): #make all points on imTmp inside ellipse to be 0 and on imCut 1
            for j in range(bbox[1],min(bbox[1]+bbox[3],im.shape[0])):
                if cv2.pointPolygonTest(ellipse,(i,j),False)>=0:
                    imTmp[j,i]=0
                    imCut[j,i] = 1
    
    #perform dilation
    kernel = numpy.ones((2,2),np.uint8)
    imTmp = cv2.morphologyEx(imTmp, cv2.MORPH_OPEN, kernel)
    kernel = numpy.ones((9,9),np.uint8)
    imTmp = cv2.morphologyEx(imTmp, cv2.MORPH_CLOSE, kernel)
    
    labeled,nr_objects = label(imTmp) #find objects left
    area = np.array([np.sum(labeled==k) for k in range(1,nr_objects+1)]) #get their area
    mask = numpy.uint8(np.zeros_like(labeled))
    for ind in np.where(area>=minEmbArea)[0]: #zero those objects that are larger than minEmbArea
        mask = mask + 255*numpy.uint8(labeled==ind+1)
    return mask, imCut

def getMask(image, DIC = False):
#     kernel = numpy.ones((11,11),np.uint8)
    kernel = numpy.ones((21,21),np.uint8)
    if DIC:
        edge = cv2.Canny(image,60,255)
        if debug: showIm(edge)
        edgeExt = numpy.zeros(numpy.array(edge.shape)+60)
        edgeExt[30:-30,30:-30] = edge
        imt = cv2.morphologyEx(edgeExt, cv2.MORPH_CLOSE, kernel)
        imt = imt[30:-30, 30:-30]
        if debug: showIm(imt)
        labeled,nr_objects = label(imt)
        area = np.array([np.sum(labeled==k) for k in range(1,nr_objects+1)])
        mask = numpy.uint8(np.zeros_like(labeled))
        for ind in np.where(area>=minEmbArea)[0]:
            mask = mask + 255*numpy.uint8(labeled==ind+1)
    else:
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        labeled,nr_objects = label(image)
        area = np.array([numpy.sum(labeled==k) for k in range(1,nr_objects+1)])
        if len(area)==0 or max(area)<minEmbArea: return np.zeros_like(image)
        mask = numpy.uint8(np.zeros_like(labeled))
        for ind in np.where(area>=minEmbArea)[0]:
            mask = mask + 255*numpy.uint8(labeled==ind+1)
    return mask

def getMaskStak(images):
#     kernel = numpy.ones((11,11),np.uint8)
    kernel = numpy.ones((5,5),np.uint8)
    if debug: showIm(images[0])
    allEdges = numpy.zeros_like(images[0])
    for image in images[7:10]:
#         edge = cv2.Canny(image,60,200) 67
        edge = cv2.Canny(image,60,min(255,255*numpy.mean(images[0])/67))
        allEdges[numpy.where(edge==255)]=255
    if debug: showIm(allEdges)
    edgeExt = numpy.zeros(numpy.array(allEdges.shape)+60)
    edgeExt[30:-30,30:-30] = allEdges
    imt = cv2.morphologyEx(edgeExt, cv2.MORPH_CLOSE, kernel)
    imt = imt[30:-30, 30:-30]
    if debug: showIm(imt)
    labeled,nr_objects = label(imt)
    area = np.array([np.sum(labeled==k) for k in range(1,nr_objects+1)])
    mask = numpy.uint8(np.zeros_like(labeled))
    for ind in np.where(area>=minEmbArea)[0]:
        mask = mask + 255*numpy.uint8(labeled==ind+1)
    return mask

def getMaskFromEdge(im):
    kernel = numpy.ones((7,7),np.uint8)
    image = cv2.medianBlur(im, 31)
    edge = cv2.Canny(image,0,20)
    imt = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    imt = cv2.dilate(imt,numpy.ones((3,3),np.uint8))
    imt = np.uint8(morphology.binary_fill_holes(imt))
    labeled,nr_objects = label(imt)
    area = np.array([np.sum(labeled==k) for k in range(1,nr_objects+1)])
    mask = numpy.uint8(np.zeros_like(labeled))
    for ind in np.where(area>=minEmbArea)[0]:
        mask = mask + 255*numpy.uint8(labeled==ind+1)

    imt = cv2.erode(imt,numpy.ones((3,3),np.uint8))
#     showIm(edge)
    return mask

def cropEllipse(im, eParams):
    ''' crops ellipse out of the image, the outside of the ellipse is black.
    NOTE: uses Cassini oval as the ellipse shape.
    INPUT:
    im: image
    eParams: ellipse parameters in form of (a,b), center, angle
    a,b: ellipse size
    center: numpy array with center coordinates in the image
    angle: angular orientation of the ellipse
    
    OUTPUT:
    image of the same size as im with zeros outside of the ellipse
    '''
    imTmp = np.zeros_like(im)
#     ellipse = create_ellipse(*eParams)
    ellipse = create_cassini_oval(*eParams)
    ellipse = numpy.array([[[int(point[0]),int(point[1])]] for point in ellipse])
    bbox = np.array(cv2.boundingRect(ellipse))
    ''' fix box boundaries to be within image '''
    if bbox[0]<0:
        bbox[2]+=bbox[0]
        bbox[0]=0
    if bbox[1]<0:
        bbox[3]+=bbox[1]
        bbox[1]=0
    if bbox[0]+bbox[2]>=im.shape[0]:
        bbox[2] = im.shape[0]-bbox[0]-1
    if bbox[1]+bbox[3]>=im.shape[1]:
        bbox[3] = im.shape[1]-bbox[1]-1

    imTmp[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = im[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    for i in range(bbox[0],bbox[0]+bbox[2]):
        for j in range(bbox[1],bbox[1]+bbox[3]):
            if cv2.pointPolygonTest(ellipse,(i,j),False)<0: imTmp[j,i]=0
    return imTmp

def cropRotate(tmp):
    ''' crops ellipse and rotates the image.
    INPUT:
    tmp: tuple of im, eParams and flip.
    im: image
    flip: flip image 180 degrees for ap orientation
    eParams: ellipse parameters in form of (a,b), center, angle
    a,b: ellipse size
    center: numpy array with center coordinates in the image
    angle: angular orientation of the ellipse
    
    OUTPUT:
    cropped rectangular image of the ellipse size with zeros outside of the ellipse
    '''
    
    im, eParams, flip = tmp
    (a,b), center, angle = eParams
    eParams = (a+5,b+3), center, angle #add extra pix To make sure that all of the embryo is included
    im = cropEllipse(im, eParams)
    im32 = np.float32(im)
    x,y = im.shape[1]/2 - center[0], im.shape[0]/2-center[1]
    mapy, mapx = np.mgrid[0:im.shape[0],0:im.shape[1]].astype(np.float32)
    mapx = mapx-x
    mapy = mapy-y
    im = cv2.remap(im32, mapx, mapy, interpolation=cv2.INTER_LINEAR).astype(im.dtype)
    center=numpy.transpose(im.shape)/2
    matrix = cv2.getRotationMatrix2D(tuple(center), angle*180/np.pi, 1.0)
    rotatedIm = cv2.warpAffine(im, matrix, im.shape)
    width, height = (int(2*a),int(2* b))
    top, bot = max(0,center[1]-height/2), min(rotatedIm.shape[0],center[1]+height/2)
    left, right = max(0,center[0]-width/2), min(rotatedIm.shape[1],center[0]+width/2)
    res = rotatedIm[top:bot, left:right]
    if flip: res=np.rot90(res,k=2)
    return res


def findEmbsonIm(mask):
    ''' finds all embryos on the mask and outputs a list of parameters for each embryo '''
    maskOut = mask.copy()
    eParams = []
    i=0
    while np.max(maskOut)>0:
        success, emb = findEmbryo(maskOut)
#         print('{0} search for embs, success={1}'.format(i,success), np.sum(maskOut))
        if success: eParams.append(emb)
        maskOut , maskIn = removeFromMask(maskOut, [emb])
        maskOut = getMask(maskOut)
        if np.sum(maskIn)==0: break
        i+=1
    return eParams

def getAP(im):
    '''
    calculates necessity of flipping the image using intensity value.
    The intensity on the left side should be higher. 
    INPUT:
    im: image of cropped embryo oriented along the long axis (numpy array type)
    OUTPUT:
    flip: necessity to flip the image 180 degree (boolean)
    '''
    return 1.*numpy.sum(im[:,:im.shape[1]/2])/numpy.sum( im[:,im.shape[1]/2:])
#     if numpy.sum(im[:,:im.shape[1]/2]) < numpy.sum(im[:,im.shape[1]/2:]): return True
#     else: return False

def getAP2(im1, im2):
    '''
    calculates necessity of flipping the image using intensity value.
    im1 intensity is subtracted from im2. The positive intensity should be on the left.
    INPUT:
    im1: Green channel maximum projection image of cropped embryo oriented along the long axis (numpy array type)
    im2: Red channel maximum projection image of cropped embryo oriented along the long axis (numpy array type)
    OUTPUT:
    flip: necessity to flip the image 180 degree (boolean)
    '''
    imt=np.float32(im2)-np.float32(im1)
    imt[np.where(imt<0)]=0.
#     imt=65025*imt/np.max(imt)
#     showIm(im1)
#     showIm(im2)
#     showIm(np.uint16(imt))
    return 1.*np.sum(imt[:,:imt.shape[1]/2])/np.sum(imt[:,imt.shape[1]/2:])

if __name__ == '__main__':
    folder = '/home/renat/Documents/work/imaging/development/test16/'
#     folder = '/home/renat/Documents/work/imaging/RNAi/'
    files = glob.glob(folder+'*.tif')
    files.sort()
    for fileName in  files:
        print('file', fileName)
        z = 1
        slide = 0
        im = Image.open(fileName)
        im.seek(int(z/2)+slide*z)
        imA = np.asarray(im)
        showIm(imA)
        
#         mask = getMaskFromEdge(np.asarray(im))
        im8b = np.uint8(255*(imA - np.min(imA))/np.max(imA))
        mask = getMask(im8b, DIC=True)
        eParams = findEmbsonIm(mask)
        
        for params in eParams:
            cropRotate(imA, params)
    