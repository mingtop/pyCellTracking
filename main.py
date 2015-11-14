# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import utils.data as ld
import segment.segment as seg

curDir = os.getcwd()
sys.path.append(curDir+'data')
dataDir = '/media/jamin/Data/Cell/c1'

cropImSz = [817,610]    # [crop_width,crop_height]
cropPt   = [3268,1220]  # [crop_startX,crop_startY]
trkIdx   = [181,136]      # [startIdx ,endIdx ] 
tracker  = 'CNN'


# 1. load data from certain Path
# load from different source later
print('step1: loading data')
if os.path.isfile('data/images.npy'):
   images = np.load('data/images.npy')
else:
    images = ld.loadImages(dataDir,cropImSz,cropPt)
    

# 2. cell Detection
print('step2: cell Detection')

segMethod = '1'

im = images[:,:,:,trkIdx[0]];

segmentResult = seg.cvCellNuclei(im)

# 3. cell tracking















# 4. show lineage   








# 5. evaluation by Different rule or beachmark





