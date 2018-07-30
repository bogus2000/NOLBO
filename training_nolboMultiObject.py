import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import sys
import src.module.nolbo as nolboModule
import dataset_utils.dataset_loader.nolbo_dataset as nolboDataset

imgSizeList = [
    [320,320,3,96],
    [352,352,3,72],
    [384,384,3,72],
    [416,416,3,64],
    [448,448,3,56],
    [480,480,3,48],
    [512,512,3,40],
    [544,544,3,40],
    [576,576,3,40],
    [608,608,3,40],
]

nolboConfig = {
    'maxPoolNum':5,
    'bboxDim':5, #h,w,x,y,objectness
    'predictorNumPerGrid':2,
    'class':True, 'zClassDim':64, 'classDim':40,
    'inst':True, 'zInstDim':64, 'instDim':1000,
    'rot':True, 'zRotDim':3, 'rotDim':3,
    'isTraining':True,
    'nameScope' : 'nolbo',
    'encoder':{
        'inputImgDim':[352, 352, 3],
        'trainable':True,
        'activation':tf.nn.elu,
        'lastFilterNumList':[1024,1024,1024],
        'lastKernelSizeList':[3,3,3],
        'lastStridesList':[1,1,1],
        'lastPool':None,
    },
    'decoder':{
        'outputImgDim':[64,64,64,1],
        'trainable':True,
        'filterNumList':[512,256,128,64,1],
        'kernelSizeList':[4,4,4,4,4],
        'stridesList':[1,2,2,2,2],
        'activation':tf.nn.elu,
        'lastLayerActivation':tf.nn.sigmoid,
    },
    'prior':{
        'hiddenLayerNum':3,
        'trainable':True,
        'activation':tf.nn.elu,
        'constLogVar':0.0,
    },
}

def trainNolboMultiObject(
        nolboConfig, batchSize=32, training_epoch=1000,
        learningRate = 0.0001,
        savePath=None,
        encoderRestorePath=None,
        decoderRestorePath=None,
        priorRestorePath=None,
):
    dataPath_Object3D = '/media/yonsei/500GB_SSD/ObjectNet3D/'
    dataPath_Pascal3D = '/media/yonsei/500GB_SSD/PASCAL3D+_release1.1/'
    dataset = nolboDataset.nolboDatasetMultiObject(
        nolboConfig=nolboConfig,
        dataPath_Pascal3D=dataPath_Pascal3D,
        dataPath_ObjectNet3D=dataPath_Object3D,
    )
    dataset.setInputImageSize(nolboConfig['encoder']['inputImgDim'])
    model = nolboModule.nolbo_multiObject(config=nolboConfig)
    if encoderRestorePath!=None:
        print 'restore encoder core...'
        model.restoreEncoderCore(encoderRestorePath)
        print 'restore encoder last layer...'
        model.restoreEncoderLastLayer(encoderRestorePath)
    if decoderRestorePath!=None:
        print 'restore decoder...'
        model.restoreDecoder(decoderRestorePath)
    if priorRestorePath!=None:
        print 'restore priornet...'
        model.restorePriornet(priorRestorePath)
    loss = np.zeros(9+2)
    pr = np.zeros(2)
    epoch = 0
    iteration = 0
    run_time = 0.0
    print 'start training...'
    while epoch< training_epoch:
        start = time.time()
        # if iteration%10 == 0:
        #     np.random.shuffle(imgSizeList)
        #     dataset.setInputImageSize(imgSizeList[0][0:3])
        #     batchSize = imgSizeList[0][3]
        batchData = dataset.getNextBatch(batchSize=batchSize)
        inputData={
            'learningRate':learningRate,
            'inputImages':batchData['inputImages'],
            'bboxHWXY':batchData['bboxImages'][...,0:4],
            'objectness':batchData['bboxImages'][...,4:5],
            'outputImages':batchData['outputImagesOrg'],
            'classList':batchData['classList'],
            'instList':batchData['instList'],
            'EulerAngle':np.concatenate(
                [np.sin(batchData['EulerRad']), np.cos(batchData['EulerRad'])], axis=-1),
        }
        epochCurr = dataset._epoch
        dataStart = dataset._dataStart
        dataLength = dataset._dataLength
        if epochCurr!=epoch:
            iteration = 0
            loss = loss * 0.0
            run_time = 0.0
            if savePath!=None:
                print 'save model...'
                model.saveNetworks(savePath=savePath)

        epoch=epochCurr
        lossTemp, prTemp = model.fit(batchDict=inputData)
        lossTemp = np.array(lossTemp)
        prTemp = np.array(prTemp)
        end=time.time()
        loss = (loss * iteration + lossTemp) / (iteration + 1.0)
        pr = (pr * iteration + prTemp) / (iteration + 1.0)
        run_time = (run_time * iteration + (end - start)) / (iteration + 1.0)
        # print loss
        sys.stdout.write("Ep:{:03d} it:{:04d} rt:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:06d}/{:06d} ".format(dataStart, dataLength))
        sys.stdout.write(
            "loss=tot:{:.3f},bbx:{:.3f},obj:{:.3f},nob:{:.3f},vox:{:.3f},reg:{:.3f},cin:{:.3f},AEIin:{:.3f},sc:{:.3f},ob:{:.3f},nb:{:.3f}".format(
                loss[0], loss[1], loss[2],loss[3], loss[4], loss[5], loss[6], loss[7], loss[8], loss[9],loss[10]))
        sys.stdout.write(" p,r={:.4f},{:.4f}   \r".format(pr[0], pr[1]))
        sys.stdout.flush()

        if loss[0] != loss[0]:
            print ''
            return
        iteration = iteration + 1.0

# for 1st epoch : momentum, lr=1e-7, batchSize=56, gamma for binaryLoss=0.6

def main():
    trainNolboMultiObject(
        nolboConfig=nolboConfig,
        batchSize=64, training_epoch=1000,
        learningRate=(1e-4),
        savePath='weights/nolbo_multiObject/',
        # encoderRestorePath='weights/imagenet_classifier/',
        # encoderRestorePath='weights/renderforCNN_classifier/',
        # encoderRestorePath='weights/nolbo_singleObject/',
        encoderRestorePath='weights/nolbo_multiObject/',
        # decoderRestorePath='weights/VAE3D/',
        # priorRestorePath='weights/VAE3D/',
        decoderRestorePath='weights/nolbo_multiObject/',
        priorRestorePath='weights/nolbo_multiObject/',
    )
if __name__ == "__main__":
    sys.exit(main())




