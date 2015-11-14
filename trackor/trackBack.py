# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:27:08 2015
    
    low level tracking 

@author: jamin
"""
import numpy as np
import utils.data as util
import cv2
# caffe


# single cell tracking 
def singleTracking(images,trkIdx,segResult):    
    startIdx = trkIdx[0]
    endIdx   = trkIdx[1]
    seqIdx   = range(endIdx,startIdx+1);
    seqIdx.reverse();
    print("Video has total %d frames." %(len(seqIdx)) )
    

# tracking every cell 
    for cellId in range(len(segResult)):

        x,y,w,h = cv2.boundingRect(segResult[cellId])
        pt = [x,y,w,h]
        im = images[:,:,:,startIdx]
        util.sampleData(im,pt,inrad,outrad,200) 

        # train a classifer    
        
        
        
        
        
        seqIdx.remove(startIdx)     # remove the first frm
        
        # from 2 to end frame trakcing
        for idx in seqIdx:
            im = images[:,:,:,idx];
                    
            
        
            
        
        return 1  # tracking results ....
    
    


# mutli cell tracking
def multiTracking(images,trkIdx,segResult):
# to do here    
    
    return 1