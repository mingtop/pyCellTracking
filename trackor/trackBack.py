# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:27:08 2015
    
    low level tracking 

@author: jamin
"""
import numpy as np
import utils.data as util
import utils.show as show
import utils.cnn as cnn
#import cnn

# single cell tracking 
def singleTracking(images,trkIdx,segResult,solverDir):    
    startIdx = trkIdx[0]
    endIdx   = trkIdx[1]
    inrad    = 5
    outrad   = 0  
    hisFrmNum = 1
    trkThreshold = 0.75
    
    netType = 0             #  0: softmax  1 : crossentropy loss
    
        
    seqIdx   = range(endIdx,startIdx+1);
    seqIdx.reverse();
    print("Video has total %d frames." %(len(seqIdx)) )
    

# tracking every cell 
    for cellId in range(len(segResult)):
        
        px,py,pw,ph = segResult[cellId,:]
        pt = [px,py,pw,ph]
        im = images[:,:,:,startIdx]
        # enssemable data and labels
        posCoor = util.sampleData(im,pt,inrad,outrad,100000) 
        posData = util.getSampleData(im,posCoor)
        show.showSampleImage(im,posCoor,'pos')
        negCoor = util.sampleData(im,pt,30, 4+inrad,100)        
        negData = util.getSampleData(im,negCoor)
        show.showSampleImage(im,negCoor,'neg')
        x = np.vstack((posData,negData))
        
        if netType == 0: # Softmax Loss:
            y = np.zeros(int(posCoor.shape[0]+negCoor.shape[0]))
            y[0:int(posCoor.shape[0])] = 1
        elif netType == 1:  # CossEntropy Loss
            y = np.zeros((int(posCoor.shape[0]+negCoor.shape[0]),2,1,1),dtype=np.float32)
            y[0:int(posCoor.shape[0]),0,0,0] = 1
            y[int(posCoor.shape[0]):int(posCoor.shape[0]+negCoor.shape[0]),1,0,0] = 1

          
        print("Y dim :%d" % (y.ndim) )
        # saved to net/CELLID_final.caffemodel 
        cnn.trainCNN(x,y,solverDir,cellId)  

#        seqIdx.remove(startIdx)     # remove the first frm
#        seqCoor = np.zeros([hisFrmNum,4],dtype=np.int8)        
#        seqCoor[0,:] = np.reshape(np.array(pt),[-1,4])
        seqCoor = []  # story history coor for upadte sampeldata
        seqCoor.append(pt)

# from 2 to end frame trakcing
        for idx in seqIdx:
            im = images[:,:,:,idx]            
            prdCoor = util.sampleData(im,pt,5,0,1000)
            prdData = util.getSampleData(im,prdCoor)
#            show.showSampleImage(im,prdCoor,'prd')
            
            
            # do predication 
            preIdx,preVal = cnn.testCNN(prdData,solverDir,cellId,np.mean(x))
            prePos = prdCoor[preIdx,:]
            prePos = prePos.astype(int)
            # update 
            if preVal < trkThreshold:                                
                # sample neg pos
                x,y = util.detectionSample(images[:,:,:,idx-len(seqCoor):idx],seqCoor)
                cnn.updateTrain(x,y,solverDir,cellId)
#               # prePt,preVal = cnn.test(preData)                                            
            
            
            if len(seqCoor) < hisFrmNum:
                seqCoor.append(prePos.tolist())
            else:
                seqCoor.pop(0)
                seqCoor.append(prePos.tolist())
            
            show.showSampleImage(im,prePos,preVal)
           # rember prePt, preVal
            pt = prePos
            pt = pt.tolist()
        print('CellID: %d tracking done!' % (cellId))
        

    return -1  # tracking results ....
    
















# mutli cell tracking
def multiTracking(images,trkIdx,segResult):
# to do here
    
    
    
    
    
    return 1