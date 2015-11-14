# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:01:12 2015
    load images from certain path
@author: jamin
"""
import os
import numpy as np
import glob
from PIL import Image


def loadImages(dataDir,cropImSz,cropPt):
    # cropImSz: [crop_width, crop_height]
    # cropPt:   [crop_startX,crop_startY]
    if not os.path.exists(dataDir):
        raise RuntimeError('could not find path "%s"' % dataDir)          
        
    images = np.zeros([cropImSz[1],cropImSz[0],3, len(glob.glob(dataDir+'/*.TIF'))],dtype=np.uint8)
    i = 0
    for tifile in glob.glob( dataDir+'/*.TIF' ):
        im = Image.open(tifile)      
          # crop orginal im     
#        im = im.crop([cropPt[0],cropPt[1],cropPt[0]+cropImSz[0],cropPt[1]+cropImSz[1]]);
#        im.show()
        imarray = np.array(im)
        imarray = imarray[cropPt[1]:cropPt[1]+cropImSz[1],cropPt[0]:cropPt[0]+cropImSz[0],:]
        # im = Image.fromarray(imarray)
        # im.show()
        images[:,:,:,i] = imarray
        i = i + 1
        print('loading.. "%d"',i)        
        
    # save to data images.npy
    np.save('data/images.npy',images)
    
    return images


def sampleData(im,pt,inrad,outrad,maxNum):
    # return im or coordinate
    
    
    
    
    
    
    return 1