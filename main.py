# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:27:08 2015    
@author: jamin SCU
"""
import sys
import os
import numpy as np
import utils.data as ld
import trackor.trackBack as tr
import segment.segment as seg

# params config
curDir = os.getcwd()
sys.path.append(curDir+'data')
dataDir = '/media/jamin/Data/Cell/c1'
cropImSz = [817,610]    # [crop_width,crop_height]
cropPt   = [3268,1220]  # [crop_startX,crop_startY]
trkIdx   = [181,136]      # [startIdx ,endIdx ] 


# 1. load data from certain Path
print('step1: loading data')
if os.path.isfile('data/images.npy'):
   images = np.load('data/images.npy')
else:
    images = ld.loadImages(dataDir,cropImSz,cropPt)


# 2. cell Detection
print('step2: cell Detection')

segMethod = 1   # 1: opencv2  2:caffe  3. End2End
im = images[:,:,:,trkIdx[0]];

if  segMethod == 1:# opencv2
    segResult, segStatus = seg.cvCellNuclei(im)   
elif segMethod == 2:# CNN caffe
    segResult, segStatus = seg.cnnCellNuclei(im)


# 3. cell tracking
print('step3: cell tracking')
trkMethod = 1  
# 1: single-cell tracking by CNN
# 2: multiCell   tracking by CNN 
if trkMethod ==1:
    trkResult = tr.singleTracking(images,trkIdx,segStatus) 
elif trkMethod == 2:
    trkResutl = tr.multiTracking( images,trkIdx,segStatus)



# 4. show lineage   
print('step4: show lineage')







# 5. evaluation by Different rule or beachmark





