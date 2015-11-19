# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:11:43 2015

@author: jamin
"""
import cv2
import numpy as np

# show parts of sampleData's position on the image
def showSampleImage(im,coor,dataType):
    # b,g,r
    if dataType.find("pos")>-1 :
        colr = (0,255,0)
    elif dataType.find("neg")>-1:
        colr = (0,0,255)
    elif dataType.find("prd")>-1:
        colr = (255,0,0)
    else:
        colr = (255,255,255)
#    org = cv.CreateMatHeader(im.shape[0],im.shape[1],cv.CV_32FC3)
#    cv.SetData(org,im,imshape[1]*   # aim to construct a cv::mat
#    org = cv.fromarray(im)
    im = np.ascontiguousarray(im)
#    org = cv2.cv.fromarray(im)
    n = coor.shape[0]   
#    coor = int(coor)
    for i in range(0,n):
#        cv.Rectangle(org,(int(coor[0,i]),int(coor[1,i])),(int(coor[0,i]+coor[2,i]),int(coor[1,i]+coor[3,i])), colr,2)
        if np.random.rand(1)<0.2 :   # only show parts
            cv2.rectangle(im, (int(coor[i,0]),int(coor[i,1])),(int(coor[i,0]+coor[i,2]),
                               int(coor[i,1]+coor[i,3])), colr,1)

#    cv.ShowImage('test',org);
    x,y,c = im.shape
    cv2.putText(im,dataType,(y-40,15),cv2.FONT_HERSHEY_PLAIN,1,colr,2)
    cv2.imshow(dataType,im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in range(1,10):
        cv2.waitKey(1) 
         
    return
    
# show CNN weights
def showCNNWeights(w):
    
    print( 'No implament' )
    
    
    
    return 


   
# show cell's lineage
def showLineage(trkRes):
# Todo: same cell has same clr
    # GraphicZ   2-X trees

    
    
    
    
    return