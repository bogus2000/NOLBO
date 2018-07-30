import numpy as np
import tensorflow as tf
import cv2
import time, sys
import os, pickle
import src.module.nolbo as nolboModule
import dataset_utils.dataset_loader.nolbo_dataset as nolboDataset

nolboConfig = {
    'class':True, 'zClassDim':64, 'classDim':40,
    'inst':True, 'zInstDim':64, 'instDim':1000,
    'rot':True, 'zRotDim':3, 'rotDim':3,
    'isTraining':True,
    'nameScope' : 'nolbo',
    'encoder':{
        'inputImgDim':[None, None, 3],
        'trainable':True,
        'activation':tf.nn.elu,
        'lastPool':'average',
    },
    'decoder':{
        'outputImgDim':[64,64,64,1],
        'trainable':False,
        'filterNumList':[512,256,128,64,1],
        'kernelSizeList':[4,4,4,4,4],
        'stridesList':[1,2,2,2,2],
        'activation':tf.nn.elu,
        'lastLayerActivation':tf.nn.sigmoid,
    },
    'prior':{
        'hiddenLayerNum':3,
        'trainable':False,
        'activation':tf.nn.elu,
        'constLogVar':0.0,
    },
}

imgSizeList = [
    [480, 640, 3],
    [360, 480, 3],
    [300, 400, 3],
    [240, 320, 3],
    [180, 240, 3],
    [150, 200, 3],
    [120, 160, 3],
    [448, 448, 3],
    [416, 416, 3],
    [352, 352, 3],
    [320, 320, 3],
    [288, 288, 3],
    [224, 224, 3],
    [112, 112, 3]
]

def trainNolboSingleObject(
        nolboConfig,
        batchSize=32, training_epoch=1000,
        learningRate = 0.0001,
        savePath=None,
        encoderRestorePath=None,
        decoderRestorePath=None
):

    dataPath_Object3D = '/media/yonsei/500GB_SSD/ObjectNet3D/'
    dataPath_Pascal3D = '/media/yonsei/500GB_SSD/PASCAL3D+_release1.1/'

    dataset = nolboDataset.nolboDatasetSingleObject(
        nolboConfig=nolboConfig,
        dataPath_ObjectNet3D=dataPath_Object3D,
        dataPath_Pascal3D=dataPath_Pascal3D,
    )
    dataset.setInputImageSize([224,224,3])
    model = nolboModule.nolbo_singleObject(config=nolboConfig)
    if encoderRestorePath != None:
        print 'restore encoder weights...'
        model.restoreEncoderCore(encoderRestorePath)
        model.restoreEncoderLastLayer(encoderRestorePath)
    if decoderRestorePath != None:
        print 'restore decoder and prior weights...'
        model.restoreDecoder(decoderRestorePath)
        model.restorePriornet(decoderRestorePath)
    loss = np.zeros(6)
    pr = np.zeros(2)
    epoch = 0
    iteration = 0
    run_time = 0.0

    print 'start training...'
    while epoch < training_epoch:
        start = time.time()
        # np.random.shuffle(imgSizeList)
        # dataset.setInputImageSize(imgSizeList[0])
        batchData = dataset.getNextBatch(batchSize=batchSize)
        if learningRate==None:
            learningRate = 0.0001
        inputData={
            'learningRate' : learningRate,
            'inputImages' : batchData['inputImages'],
            'outputImages' : np.concatenate(
                [
                # batchData['outputImages'],
                 batchData['outputImagesOrg'],
                 ], axis=0),
            'classList' : batchData['classList'],
            'instList' : batchData['instList'],
            'AEIAngleSinCos' : np.concatenate([np.sin(batchData['AEIAngle']),np.cos(batchData['AEIAngle'])], axis=-1)
        }
        epochCurr = dataset._epoch
        dataStart = dataset._dataStart
        dataLength = dataset._dataLength
        if epochCurr != epoch:
            iteration = 0
            loss = loss * 0.0
            run_time = 0.0
            if savePath != None:
                print 'save model...'
                model.saveNetworks(savePath)
        epoch = epochCurr
        lossTemp, prTemp = model.fit(batchDict=inputData)
        lossTemp = np.array(lossTemp)
        prTemp = np.array(prTemp)
        end = time.time()
        loss = (loss * iteration + lossTemp) / (iteration + 1.0)
        pr = (pr * iteration + prTemp) / (iteration + 1.0)
        run_time = (run_time * iteration + (end - start)) / (iteration + 1.0)
        # print loss
        sys.stdout.write("Ep:{:03d} it:{:04d} rt:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:06d}/{:06d} ".format(dataStart, dataLength))
        sys.stdout.write(
            "loss=tot:{:.3f},vox:{:.3f},reg:{:.3f},cin:{:.3f},rn:{:.3f},sc:{:.3f}".format(loss[0], loss[1], loss[2],loss[3], loss[4], loss[5]))
        sys.stdout.write(" p,r={:.4f},{:.4f}   \r".format(pr[0], pr[1]))
        sys.stdout.flush()

        if loss[0] != loss[0]:
            print ''
            return
        iteration = iteration + 1.0

if __name__=="__main__":
    sys.exit(trainNolboSingleObject(
        nolboConfig=nolboConfig,
        batchSize=80,
        training_epoch = 1000,
        learningRate = 1e-5,
        savePath='weights/nolbo_singleObject/',
        encoderRestorePath='weights/nolbo_singleObject/',
        decoderRestorePath='weights/nolbo_singleObject/',
    ))
