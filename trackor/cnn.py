# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 22:44:28 2015

@author: jamin
"""
import os,sys,time
import utils.data as utils
import numpy as np
import scipy.misc as symc
# caffe
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


#
def trainCNN(x,y,solverDir,cellIdx):
    # parse Caffe's param prototxt    
    netDir,solverFn = os.path.split(solverDir)           
    projDir = os.getcwd()
    if len(netDir)>0:
        os.chdir(netDir)
    
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFn).read(),solverParam)
    
    netFn = solverParam.net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(),netParam)
           
    batchDim = utils.infer_data_dimensions(netFn)
    print('[train]: batch shape: %s' %str(batchDim))
    
    preDir = str(solverParam.snapshot_prefix)   # unicode ---> str
    outDir = os.path.join(projDir,preDir)
    if os.path.isdir(outDir):
        os.mkdir(outDir)
        
    print('[train]: training data shape: %s'  %(str(x.shape)) )
    
    # Create the caffe solover
    solver = caffe.SGDSolver(solverFn)
    for name, blobs in solver.net.params.iteritems():
        for bIdx, b in enumerate(blobs):
            print("%s[%d] : %s" %( name,bIdx,b.data.shape))
    
    GPU = solverParam.solver_mode  # cpu 0 , gpu 1
    
    if GPU :
        isModeCPU = 0
        gpuId = 0
    else:
        isModeCPU = (solverParam.solver_mode == solverParam.SolverMode.Value('CPU'))
        gpuId = 0
    # different edition has different caffe API
    try:
        if not isModeCPU:
            caffe.set_mode_gpu()
            caffe.set_device(gpuId)
        else:
            caffe.set_mode_cpu()
#            caffe.set_phase_train()        # Needn't set phase as Train
    except AttributeError:
        if not isModeCPU:
            caffe.set_mode_gpu()
#            caffe.set_device(gpuId)
            solver.net.set_device(gpuId)        
        else:
            solver.net.set_mode_cpu()
            solver.net.set_phase_train()
    
    losses, acc = train_loop(solver, x, y,  solverParam, batchDim, outDir)
    
    solver.net.save('%d_final.caffemodel'%(cellIdx))
#    np.save(os.path.join(outDir, '%s_losses' % outDir), losses)
#    np.save(os.path.join(outDir, '%s_acc' % outDir), acc)
    
    print('[train]: ID %d done pretrain!' %(cellIdx))
    
    # change to projDir
    if len(projDir) >0 :
        os.chdir(projDir)
    
    
    return 
    
def train_loop(solver, X, Y, solverParam, batchDim, outDir):
    assert(batchDim[2] == batchDim[3])
    
    xmean = np.mean(X)        
    X = X -xmean
    Xi = np.zeros(batchDim, dtype=np.float32)
    # Softmax Loss
#    Yi = np.zeros((batchDim[0],), dtype=np.float32)
    # Cross-entroy Loss
    Yi = np.zeros((batchDim[0],2,1,1),dtype=np.float32)
    losses = np.zeros((solverParam.max_iter,)) 
    acc = np.zeros((solverParam.max_iter,))
    currIter = 0
    currEpoch = 0    
    
    alpha = solverParam.base_lr            # alpha := learning rate
    mu = solverParam.momentum              # mu := momentum
    gamma = solverParam.gamma              # gamma := step factor
    isModeStep = (solverParam.lr_policy == u'step')
    isTypeSGD = (solverParam.solver_type == solverParam.SolverType.Value('SGD'))
    Vall = {}                              # stores previous SGD steps (for all layers)

    if not (isModeStep and isTypeSGD):
        raise RuntimeError('Sorry - only support SGD "step" mode at the present')
    
    cnnTime = 0.0                          # time spent doing core CNN operations
    tic = time.time()   
    while currIter < solverParam.max_iter:
        currEpoch += 1
        it = utils.trainSubDataGenerator(X, Y, batchDim[0])     
        for Idx, epochPct in it:
            Yi = Y[Idx]                   # softmax loss
            Yi = np.float32(Yi)           # softmax loss
#            Yi[0:len(Idx),0,0,0] = Yi[0:len(Idx),0,0,0] & (Y[Idx]==1) # cross loss
#            Yi[0:len(Idx),1,0,0] = Yi[0:len(Idx),1,0,0] & (Y[Idx]==0) # cross loss
#            Yi = np.float32(Yi)                                       # cross loss
            
            for ii, jj in enumerate(Idx):                
#                temp = symc.imresize(X[jj,:,:,:],[28,28])       # 3-chanels
                temp = symc.imresize(X[jj,0,:,:],[28,28])
                temp = np.float32(temp)
                Xi[ii,0,:,:] = temp
#                Xi[ii,:,:,:] = temp.transpose(2,0,1)            # 3-chanals                    
            if len(Idx) < batchDim[0]:
                tYi = np.zeros((batchDim[0],), dtype=np.float32) # softmax loss
                tYi[0:len(Idx)] = Yi                             # softmax loss
                for i in range(len(Idx),batchDim[0]):
                    tYi[i] = Yi[i%len(Idx)]                      # softmax loss
#                    Yi[i,:,0,0] = Yi[i%len(Idx),:,0,0]            # cross loss
                    Xi[i,:,:,:] = Xi[i%len(Idx),:,:,:]
                Yi = np.float32(tYi)                             # softmax loss
#                Yi = np.float32(Yi)                              # cross loss
                 
            
                
# label-preserving data transformation (synthetic data generation)
#            Xi = _xform_minibatch(Xi)
            
            _tmp = time.time()
            solver.net.set_input_arrays(Xi, Yi)
                                                
            
            # XXX: could call preprocess() here?
            rv = solver.net.forward()
            solver.net.backward()

            for lIdx, layer in enumerate(solver.net.layers):
                for bIdx, blob in enumerate(layer.blobs):
                    key = (lIdx, bIdx)
                    V = Vall.get(key, 0.0)
                    Vnext = mu*V - alpha * blob.diff
                    blob.data[...] += Vnext
                    Vall[key] = Vnext
            cnnTime += time.time() - _tmp
                    
            # update running list of losses with the loss from this mini batch
            losses[currIter] = np.squeeze(rv['loss'])
            acc[currIter] = np.squeeze(rv['accuracy'])
            currIter += 1

            #----------------------------------------
            # Some events occur on mini-batch intervals.
            # Deal with those now.
            #----------------------------------------
            if (currIter % solverParam.snapshot) == 0:
                fn = 'iter_%05d.caffemodel' % (currIter)
                solver.net.save(str(fn))

            if isModeStep and ((currIter % solverParam.stepsize) == 0):
                alpha *= gamma

            if (currIter % solverParam.display) == 1:
                elapsed = (time.time() - tic)/60.
                print "[train]: completed iteration %d (of %d; %0.2f min elapsed; %0.2f CNN min)" % (currIter, solverParam.max_iter, elapsed, cnnTime/60.)
                print "[train]:    epoch: %d (%0.2f), loss: %0.3f, acc: %0.3f, learn rate: %0.3e" % (currEpoch, 100*epochPct, np.mean(losses[max(0,currIter-10):currIter]), np.mean(acc[max(0,currIter-10):currIter]), alpha)
                sys.stdout.flush()
 
            if currIter >= solverParam.max_iter:
                break  # in case we hit max_iter on a non-epoch boundary
    return losses, acc
    
    
def testCNN(x, solverDir,cellId):
        
    curDir = os.getcwd()
    netDir, solverFn = os.path.split(solverDir)
    # change to prototext Dirs    
    if len(netDir)>0:
        os.chdir(netDir)
    
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFn).read(),solverParam)
    
    netFn = str(solverParam.net)
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFn).read(),netParam)
    
    modelDir = str(cellId)+'_final.caffemodel'

    print( modelDir )
    net = caffe.Net(netFn,modelDir, caffe.TEST)
    for name, blobs in net.params.iteritems():
        print("%s:%s" %(name,blobs[0].data.shape))
    
    GPU = solverParam.solver_mode  # cpu 0 , gpu 1    
    if GPU :
        isModeCPU = 0
        gpuId = 0
    else:
        isModeCPU = (solverParam.solver_mode == solverParam.SolverMode.Value('CPU'))
        gpuId = 0
    # different edition has different caffe API
    try:
        if not isModeCPU:
            caffe.set_mode_gpu()
            caffe.set_device(gpuId)
        else:
            caffe.set_mode_cpu()
#            caffe.set_phase_train()
    except AttributeError:
        if not isModeCPU:
            net.set_mode_gpu()
            net.set_device(gpuId)        
        else:
            net.set_mode_cpu()
            net.set_phase_train()
    
    print("[test:] GPU mode: %s" % GPU )
    
    batchDim = utils.infer_data_dimensions(netFn)
    print("[test:] batch shape:%s" %(batchDim))    
    
    Xi = np.zeros((batchDim[0],1,28,28),dtype=np.float32)
    Yi = np.zeros((batchDim[0],),dtype=np.float32)
    xmean = np.mean(x)            
    x = x - xmean
    maxValue = -1
    
    for idx in utils.testDataGenerator(x,batchDim[0]):         
        # resize 
        for ii in range( 0, min(batchDim[0]-1, np.max(idx)+1) ):
            temp = symc.imresize(Xi[ii,0,:,:],[28,28])
            Xi[ii,0,:,:] = np.float32(temp)
        
        if np.max(idx)<(batchDim[0]-1):
            Xi[idx::,:,:,:] = x[idx::,:,:,:]   # work ? or not
        
        net.set_input_arrays(Xi,Yi)
        rv = net.forward()        
        out = rv['prob']
        out = np.squeeze(out)
        ymax = np.max(out[:,0])  # 0 ???????????????? is positive???
        yidx = np.argmax(out[:,0])
        if ymax>maxValue:
            maxValue = ymax
            yidx = yidx
        # yidx in appended data
        if yidx > batchDim[0]-1:
            yidx = batchDim[0]-1
    
    if len(curDir)>0:
        os.chdir(curDir)
    
    
    return yidx,maxValue