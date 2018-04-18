import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import sys
import src.module.nolbo as nolbo

# def trainNolbo(nolboConfig, maxBatchSize, training_epoch, restore=False):
#     print 'start training...'
#     nb = nolbo(config=nolboConfig)
#     dataset = nolboDataset.nolboDataset(nolboConfig=nolboConfig, datasetPath='/media/yonsei/4TB_HDD/dataset/suncg_data/training_data/')
#     if restore == True:
#         vae.restore_model('./weight_m40')
#     loss = np.zeros(5)
#     epoch = 0
#     iteration = 0
#     run_time = 0.0
#     while epoch<training_epoch:
#         start = time.time()
#         batchData = dataset.getNextBatch(maximumBatchSize=maxBatchSize)
#         epochCurr = dataset._epoch
#         dataStart = dataset._dataStart
#         dataLength = dataset._dataLength
#         if epochCurr != epoch:
#             iteration = 0
#             loss = loss * 0.0
#             run_time = 0.0
#         epoch = epochCurr
#         lossTemp = np.array(nb.fit(batchDict = batchData))
#         end = time.time()
#         loss = (loss*inner_iter + loss_temp)/(iteration+1.0)
#         run_time = (run_time*inner_iter + (end-start))/(iteration + 1.0)
#
#         print 'Epoch:{:03d} iter:{:03d} runtime:{:.3f}'.format((epoch+1), (iteration+1), run_time),\
#         'curr/total:{:05d}/{:05d}'.format(dataStart,dataLength), \
#         'loss= {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\r'.format(loss[0],loss[1],loss[2],loss[3],loss[4])
#         if dataStart+maxBatchSize >= dataLength:
#             print ''
#             if iteration%2000 == 0:
#                 print 'save model No.'+str(int(iteration%2000)+1)
#                 vae.save_model('./weight_m40')
#         iteration = iteration + 1

nolboConfig = {
    'inputImgDim':[448,448,1],
    'maxPoolNum':5,
    'predictorNumPerGrid':11,
    'bboxDim':5,
    'class':True, 'zClassDim':64, 'classDim':24,
    'inst':True, 'zInstDim':64, 'instDim':1000,
    'rot':True, 'zRotDim':3, 'rotDim':3,
    'trainable':True,
    'decoderStructure':{
        'outputImgDim':[64,64,64,1],
        'trainable':True,    
        'filterNumList':[512,256,128,64,1],
        'kernelSizeList':[4,4,4,4,4],
        'stridesList':[1,2,2,2,2],
        'activation':tf.nn.leaky_relu,
        'lastLayerActivation':tf.nn.sigmoid
    }
}

def main():
    nb = nolbo.nolbo(config=nolboConfig)
if __name__ == "__main__":
    sys.exit(main())