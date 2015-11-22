# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:01:12 2015
    load images from certain path
@author: jamin
"""
import os,sys,re
import numpy as np
import glob
import cv2
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
    #  !!!  data different frome image's dim    IM:h*w   CV2:w*H
    c = int(1)
    sampleIM= np.ndarray( [n, c , coordinate[0,3] , coordinate[0,2]],dtype = np.float32, order = 'C' )
#    sampleIM= np.ndarray( [n, 3 , coordinate[0,3] , coordinate[0,2]],dtype = np.float32 )
    for i in range(0,n):
        x,y,w,h =coordinate[i,:]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        DEBUG = 0
        if DEBUG:
            ps = np.ascontiguousarray(im)
            cv2.rectangle(ps,(x,y),(x+w,y+h),(255,255,255),1)
            cv2.imshow('ps',ps)
            cv2.waitKey(0)
        ps = im[y:y+w,x:x+h,1]
        ps = np.ascontiguousarray(ps) 
        fname = '/media/jamin/Data/Cell/classification/0/%d.jpg' %(i)
        cv2.imwrite(fname,ps)        
        if c == 3:
            sampleIM[i,c,:,:] = ps.transpose(2,1,0)              
        else:
            sampleIM[i,0,:,:] = ps.transpose(1,0)              
        if DEBUG:
            vmax = np.max(ps)
            vmin = np.min(ps)
            print('x:%d ,y:%d,w:%d,h:%d max:%f min:%f' %(x,y,w,h,vmax,vmin))       
            cv2.imshow('ps',ps)
            cv2.waitKey(0)
            cv2.destroyWindow('ps')
            for nn in range(0,10):            
                cv2.waitKey(1)

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




def infer_data_dimensions(netFn):
    """Determine the size of the Caffe input data tensor.

    There may be a cleaner way to do this through the pycaffe API (e.g. via the
    network parameters protobuf object).
    """
    with open(netFn, 'r') as f:
        contents = "".join(f.readlines())

    dimNames = ['batch_size', 'channels', 'height', 'width']
    dimensions = np.zeros((4,), dtype=np.int32)

    for ii, dn in enumerate(dimNames):
        pat = r'%s:\s*(\d+)' % dn
        mo = re.search(pat, contents)
        if mo is None:
            raise RuntimeError('Unable to extract "%s" from network file "%s"' % (dn, netFn))
        dimensions[ii] = int(mo.groups()[0])
        
    return dimensions    


    
def trainSubDataGenerator(x,y,batchSize):

    trainIdx = range(0,x.shape[0])
    np.random.shuffle(trainIdx)
    
    for ii in range(0,len(trainIdx),batchSize):
        nRet = min(batchSize,len(trainIdx) - ii)
        yield trainIdx[ii:(ii+nRet)], (1.0*ii)/len(trainIdx)

def testDataGenerator(x,batchSize):
    
    for ii in range(0,x.shape[0],batchSize):
        nRet = min(batchSize,x.shape[0]-ii)        
        yield range(ii,(ii+nRet))
    
    
    