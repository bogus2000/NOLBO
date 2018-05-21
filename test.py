import numpy as np
import tensorflow as tf
import cv2
import time, sys
import os, pickle
import src.module.nolbo as nolboModule
import dataset_utils.dataset_loader.nolbo_dataset as nolboDataset

nolboConfig = {
    'inputImgDim':[None, None, 3],
    'maxPoolNum':5,
    'predictorNumPerGrid':2,
    'bboxDim':5,
    'class':True, 'zClassDim':64, 'classDim':40,
    'inst':True, 'zInstDim':64, 'instDim':1000,
    'rot':True, 'zRotDim':3, 'rotDim':3,
    'trainable':True, 'learningRate':0.0001,
    'decoderStructure':{
        'outputImgDim':[64,64,64,1],
        'trainable':True,
        'filterNumList':[512,256,128,64,1],
        'kernelSizeList':[3,3,3,3,3],
        'stridesList':[1,2,2,2,2],
        'activation':tf.nn.elu,
        'lastLayerActivation':tf.nn.sigmoid
    }
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
        orgShapeTrain=False,
        batchSize=32, training_epoch=1000,
        learningRate = 0.0001,
        savePath=None, restorePath=None):

    dataPath_Object3D = '/media/yonsei/4TB_HDD/dataset/ObjectNet3D/'
    dataPath_Pascal3D = '/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/'
    dataPath_pix3D = '/media/yonsei/4TB_HDD/dataset/pix3d/'

    dataset = nolboDataset.nolboDatasetSingleObject(
        nolboConfig=nolboConfig,
        mode='nolbo',
        dataPath_ObjectNet3D=dataPath_Object3D,
        dataPath_Pascal3D=dataPath_Pascal3D,
        dataPath_pix3D=dataPath_pix3D
    )
    nb = nolboModule.nolbo_singleObject(config=nolboConfig)
    # nb._setOptimizer(learningRate=learningRate)
    if restorePath != None:
        print 'restore weights...'
        # nb.restoreEncoderCore(restorePath)
        # nb.restoreEncoderLastLayer(restorePath)
        # nb.restoreDecoder(restorePath)
        # np.restorePriornet(restorePath)
        nb.restoreNetworks(restorePath)
    loss = np.zeros(4)
    pr = np.zeros(2)
    epoch = 0
    iteration = 0
    run_time = 0.0

    print 'start training...'
    while epoch < training_epoch:
        start = time.time()
        np.random.shuffle(imgSizeList)
        dataset.setInputImageSize(imgSizeList[0])
        batchData = dataset.getNextBatch(batchSize=batchSize)
        if learningRate==None:
            learningRate = 0.0001
        batchData['learningRate'] = learningRate
        epochCurr = dataset._epoch
        dataStart = dataset._dataStart
        dataLength = dataset._dataLength
        if epochCurr != epoch:
            print ''
            iteration = 0
            loss = loss * 0.0
            run_time = 0.0
        epoch = epochCurr
        lossTemp, prTemp = nb.fit(batchDict=batchData)
        lossTemp = np.array(lossTemp)
        prTemp = np.array(prTemp)
        end = time.time()
        loss = (loss * iteration + lossTemp) / (iteration + 1.0)
        pr = (pr * iteration + prTemp) / (iteration + 1.0)
        run_time = (run_time * iteration + (end - start)) / (iteration + 1.0)
        # print loss
        sys.stdout.write("Epoch:{:03d} iter:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:06d}/{:06d} ".format(dataStart, dataLength))
        sys.stdout.write(
            "loss= tot:{:.3f}, vox:{:.3f}, reg:{:.3f}, nlb:{:.3f} ".format(loss[0], loss[1], loss[2], loss[3]))
        sys.stdout.write("p,r={:.4f},{:.4f}   \r".format(pr[0],pr[1]))
        sys.stdout.flush()

        if orgShapeTrain==True:
            batchData['outputImages'] = batchData['outputImagesOrg']
            batchData['EulerAngle'] = 0.0*batchData['EulerAngle']
            lossTemp, pr = nb.fit(batchDict=batchData)
            lossTemp = np.array(lossTemp)
            pr = np.array(pr)
            end = time.time()
            loss = (loss * iteration + lossTemp) / (iteration + 1.0)
            run_time = (run_time * iteration + (end - start)) / (iteration + 1.0)

            sys.stdout.write(
                "orgEp:{:03d} iter:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
            sys.stdout.write("cur/tot:{:06d}/{:06d} ".format(dataStart, dataLength))
            sys.stdout.write(
                "loss= tot:{:.3f}, vox:{:.3f}, reg:{:.3f}, nlb:{:.3f} ".format(loss[0], loss[1], loss[2], loss[3]))
            sys.stdout.write("p,r={:.4f},{:.4f}   \r".format(pr[0], pr[1]))
            sys.stdout.flush()

        if (iteration+1) % 1000 == 0 and (iteration+1) != 1:
            print ''
            iteration = 0
            loss = loss * 0.0
            pr = pr * 0.0
            run_time = 0.0
            # learningRate = learningRate*1.1
            # nb._setOptimizer(learningRate=learningRate)
            if savePath != None:
                print 'save model...'
                nb.saveNetworks(savePath)
        iteration = iteration + 1.0

if __name__=="__main__":
    sys.exit(trainNolboSingleObject(
        nolboConfig=nolboConfig,
        orgShapeTrain=False,
        batchSize=24,
        training_epoch = 1000,
        learningRate = 0.0001,
        savePath='weights/nolbo_singleObject/',
        restorePath='weights/nolbo_singleObject/'
    ))