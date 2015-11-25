# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:34:42 2015

@author: jamin
"""

import utils.show as show
import numpy as np

def main():
    print('test utils.show.showLineage() ')
    ##################################################
    #                 cellIdx=1    |   cellIdx = 2
    # frame1         [ x,y,w,h ]   |    [x,y,w,h]
    # frame2         [ x,y,w,h ]   |    [x,y,w,h]
    # frame3         [ x,y,w,h ]   |    [x,y,w,h]
    # frame4         [ x,y,w,h ]   |    [x,y,w,h]
    #################################################
    data = np.array([ [ ] ,
                      [ ] ,
                      [ ] ,
                      [ ] ,
                      [ ] ] )  
    
    
#    show.showLineage(data)



if __name__ == 'main':
    main()
    