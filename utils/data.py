# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:01:12 2015
    load images from certain path
@author: jamin
"""
import os,sys
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

# return im or coordinate
def sampleData(im,pt,inrad,outrad,maxNum):
    # return coor:   4*num    
    inrad =  int(inrad)
    outrad = int(outrad)
    x = pt[0]
    y = pt[1]
    w = pt[2]
    h = pt[3]
    [row,col,c] = im.shape
    rowsz = row - h -1
    colsz = col - w -1
    inradsq  = np.square(inrad)
    outradsq = np.square(outrad)
    
    minrow = max(1,y-inrad+1)
    maxrow = min(rowsz-1,y+inrad)
    mincol = max(1,x-inrad+1)
    maxcol = min(colsz-1,x+inrad)
    
    prob = maxNum/float((maxrow -minrow +1)*(maxcol-mincol+1))
    
    [ r, c] = np.meshgrid(range(minrow,maxrow+1),range(mincol,maxcol+1))
    dist = np.square(y-r) + np.square(x-c)
    rd = np.random.rand( r.shape[0],r.shape[1] )
    
    ind = (rd<prob)&(dist<inradsq)&(dist>=outradsq)
    
    c = c[ind == 1]
    r = r[ind == 1]
    wn = np.ones(len(c))*w
    hn = np.ones(len(r))*h
    coordinate = np.vstack((c,r,wn.T,hn.T))
    coordinate = coordinate.T
    
    return coordinate


# from coordinate to imageSample data    
def getSampleData(im,coordinate):
    n = coordinate.shape[0]
    #   num * channel * w* h
    sampleIM= np.ndarray( [n, 3 , coordinate[0,2] , coordinate[0,3]],dtype = np.float32, order = 'C' )
    for i in range(0,n):
        x,y,w,h = coordinate[i,:]
        sampleIM[i,:,:,:] = im[x:x+w,y:y+h,:].transpose(2,0,1)        
    return sampleIM
    
 
# save npy to tifs   
def npy2tif(images,dst,trkIdx):
    w,h,c,n = images.shape
    # here 181,135    
    startIdx = trkIdx[0]
    endIdx   = trkIdx[1]
    for i in range(endIdx,(startIdx+1)):
        im = images[:,:,:,i]
        im = Image.fromarray(im)
        fname = os.path.join(dst, '%d.tif' %(i) );
        im.save(fname)
        print('saved %s' %(fname) ) 
    return
    
    
