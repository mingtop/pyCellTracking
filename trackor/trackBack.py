# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:27:08 2015
    
    low level tracking 

@author: jamin
"""
import numpy as np
import os
import utils.data as util
import utils.show as show
import cv2
import cnn

# single cell tracking 
def singleTracking(images,trkIdx,segResult,solverDir):    
    startIdx = trkIdx[0]
    endIdx   = trkIdx[1]
    inrad    = 5
    outrad   = 0  
    trkThreshold = 0.75
        
    seqIdx   = range(endIdx,startIdx+1);
    seqIdx.reverse();
    print("Video has total %d frames." %(len(seqIdx)) )
    

# tracking every cell 
    for cellId in range(len(segResult)):
        
        px,py,pw,ph = cv2.boundingRect(segResult[cellId])
        pt = [px,py,pw,ph]
        im = images[:,:,:,startIdx]
        # enssemable data and labels
        posCoor = util.sampleData(im,pt,inrad,outrad,100) 
        posData = util.getSampleData(im,posCoor)
        show.showSampleImage(im,posCoor,'pos')
        negCoor = util.sampleData(im,pt,20, 4+inrad,100)        
        negData = util.getSampleData(im,negCoor)
        show.showSampleImage(im,negCoor,'neg')
        x = np.vstack((posData,negData))
        y = np.zeros(int(posCoor.shape[0]+negCoor.shape[0]))
        y[0:int(posCoor.shape[0])] = 1
        for i in range(0,20):
            ps = posData[i,:,:,:]
            ng = negData[i,:,:,:]
#            cv2.imshow('test',posData[i,:,:,:].transpose(1,2,0))
#            cv2.waitKey(0)
                
        
        # train a classifer saved to net/final.caffemodel 
        cnn.trainCNN(x,y,solverDir,cellId)       
        
        seqIdx.remove(startIdx)     # remove the first frm
       
        # from 2 to end frame trakcing
        for idx in seqIdx:
            im = images[:,:,:,idx];
#           preData = util.sampleData(im,pt,0,7,200)
#           prePt,preVal = cnn.testCNN(preData)
#           while max(preVal) < trkThreshold:
#                # rember old-pt
#                # sample neg pos
#                # train 
#                # prePt,preVal = cnn.test(preData)
#           print('Doing tracking')
#           # rember prePt, preVal
#            
#           # show every frame
                        
        
        return 1  # tracking results ....
    

# mutli cell tracking
def multiTracking(images,trkIdx,segResult):
# to do here
    
    
    
    
    
    return 1