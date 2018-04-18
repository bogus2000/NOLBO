import numpy as np
import tensorflow as tf
import cv2
import time, sys
import os, pickle
import src.module.nolbo as nolboModule
import dataset_utils.dataset_loader.nolbo as nolboDataset

nolboConfig = {
    'inputImgDim':[416,416,1],
    'maxPoolNum':5,
    'predictorNumPerGrid':2,
    'bboxDim':5,
    'class':True, 'zClassDim':64, 'classDim':30,
    'inst':True, 'zInstDim':64, 'instDim':1300,
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


def trainNolboSingleObject(
        nolboConfig,
        orgShapeTrain=False,
        batchSize=32, training_epoch=1000,
        learningRate = 0.0001,
        datasetPath=None,
        savePath=None, restorePath=None):
    dataset = nolboDataset.nolboDataset(nolboConfig=nolboConfig, mode='singleObject', datasetPath=datasetPath)
    nb = nolboModule.nolbo_singleObject(config=nolboConfig)
    nb._setOptimizer(learningRate=learningRate)
    if restorePath != None:
        nb.restoreEncoderCore(restorePath)
        nb.restoreEncoderLastLayer(restorePath)
        nb.restoreDecoder(restorePath)
        # nb.restoreNetwork(restorePath)
    loss = np.zeros(4)
    epoch = 0
    iteration = 0
    run_time = 0.0
    print 'start training...'
    while epoch < training_epoch:
        start = time.time()
        batchData = dataset.getNextBatch(batchSize=batchSize)
        epochCurr = dataset._epoch
        dataStart = dataset._dataStart
        dataLength = dataset._dataLength
        if epochCurr != epoch:
            print ''
            iteration = 0
            loss = loss * 0.0
            run_time = 0.0
        epoch = epochCurr
        lossTemp = np.array(nb.fit(batchDict=batchData))
        end = time.time()
        loss = (loss * iteration + lossTemp) / (iteration + 1.0)
        run_time = (run_time * iteration + (end - start)) / (iteration + 1.0)
        # print loss
        sys.stdout.write("Epoch:{:03d} iter:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:05d}/{:05d} ".format(dataStart, dataLength))
        sys.stdout.write(
            "loss= tot:{:.3f}, vox:{:.3f}, reg:{:.3f}, nlb:{:.3f}\r".format(loss[0], loss[1], loss[2], loss[3]))
        sys.stdout.flush()

        if orgShapeTrain==True:
            batchData['outputImages'] = batchData['outputImagesOrg']
            batchData['EulerAngle'] = 0.0*batchData['EulerAngle']
            lossTemp = np.array(nb.fit(batchDict=batchData))
            end = time.time()
            loss = (loss * iteration + lossTemp) / (iteration + 1.0)
            run_time = (run_time * iteration + (end - start)) / (iteration + 1.0)

            sys.stdout.write(
                "orgEp:{:03d} iter:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
            sys.stdout.write("cur/tot:{:05d}/{:05d} ".format(dataStart, dataLength))
            sys.stdout.write(
                "loss= tot:{:.3f}, vox:{:.3f}, reg:{:.3f}, nlb:{:.3f}\r".format(loss[0], loss[1], loss[2], loss[3]))
            sys.stdout.flush()

        if (iteration+1) % 1000 == 0 and (iteration+1) != 1:
            print ''
            iteration = 0
            loss = loss * 0.0
            run_time = 0.0
            learningRate = learningRate*1.1
            nb._setOptimizer(learningRate=learningRate)
            if savePath != None:
                print 'save model...'
                nb.saveNetworks(savePath)
        iteration = iteration + 1.0

if __name__=="__main__":
    sys.exit(trainNolboSingleObject(
    nolboConfig=nolboConfig,
    batchSize=32,
    training_epoch = 1000,
    learningRate = 0.0001,
    datasetPath='/media/yonsei/4TB_HDD/dataset/suncg_data/training_data/',
    savePath='weights/nolbo_singleObject/',
    restorePath='weights/nolbo_singleObject/'
    ))