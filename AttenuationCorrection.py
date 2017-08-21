'''
Created on Oct 24, 2014

@author: Admin
'''

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

def correctAttAll(stack, z):
    imOut = []
    for i in range(len(stack)/z):
        imOut += correctAttenuation(stack[i*z:(i+1)*z])
    return imOut

if __name__ == '__main__':
    pass