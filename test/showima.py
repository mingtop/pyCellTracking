# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 21:17:53 2015

@author: jamin
"""

import cv2
import numpy as np

ps = np.ascontiguousarray(im)
x,y,w,h = np.int8(coordinate[1,:])
cv2.rectangle(ps,(x,y),(x+w,y+h),(255,0,0),1)
cv2.imshow('ps',ps)
cv2.waitKey(0)

cv2.destroyWindow('ps')
cv2.waitKey(1)


#cv2.rectangle(ps,(245,278),(245+14,278+16),(0,255,255),1)