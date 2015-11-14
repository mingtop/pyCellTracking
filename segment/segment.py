# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 21:23:41 2015

    segment cells/cellNucleis in images 
    
    
    return
            im2:        mask of binary 
            cellStatus: contours of cell 
            
@author: jamin
"""

import cv2,cv
import numpy as np

# using opencv2 to segmente cell neucleis 
def cvCellNuclei(im):

    h,w,c = im.shape
    im2 = cv.CreateMat(h,w,cv2.CV_32FC1)
#    org = cv.fromarray(im)
    im2 = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)   # gray
    gray = im2  
#    cv2.imshow('org',gray)      
#    cv2.waitKey(0)
    ret,im2 = cv2.threshold(im2,93,255,cv2.THRESH_BINARY_INV)
#    cv2.imshow('bw',im2)
#    cv2.waitKey(0)
    kernel = np.ones((2,2),np.uint8)
    im2 = cv2.erode(im2,kernel,iterations = 1)
#    cv2.imshow('open',im2)
#    cv2.waitKey(0)
    cnt,hierarchy = cv2.findContours(im2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    
    
# deliminate small spain         
    cellStatus  = []
    for i in range(len(cnt)):
        if cv2.contourArea(cnt[i]) > 10:     
           cellStatus.append(cnt[i])
#    cv2.drawContours(gray, cellStatus, -1, (0,255,255), 3)
    
    for i in range(len(cellStatus)):
        x,y,w,h = cv2.boundingRect(cellStatus[i])
        cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0),2) 
        # gray = cv2.rectangle(gray, (x,y), (x+w,y+h), (0,255,0),2)  
        # gray will destroyed automatic
       
    cv2.imshow('test',gray)                         
    cv2.waitKey(0)
   
    cv2.destroyAllWindows()
        
    return im2,cellStatus
    
    
# using CNN detected the cell nucleis
def cnnCellNuclei(im):
 # to do    
    
    
    
    
    
    
    
    
    
    return 1
