import numpy as np
import copy
import time, sys
import dataset_utils.dataset_loader.Imagenet_dataset as imagenetDataset
import src.module.nolbo as nolboModule
import gc

nolboConfig = {
    'inputImgDim':[None, None, 3],
    'classDim': 1000,
}

imgSizeList = [
        # [480, 640, 3],
        # [360, 480, 3],
        # [300, 400, 3],
        [240, 320, 3],
        [180, 240, 3],
        [150, 200, 3],
        [120, 160, 3],
        # [448, 448, 3],
        # [416, 416, 3],
        # [352, 352, 3],
        # [320, 320, 3],
        [288, 288, 3],
        [224, 224, 3],
        [112, 112, 3]
    ]

def trainNolboClassifier(
        nolboConfig,
        batchSize=32, training_epoch=1000,
        learningRate = 0.0001,
        savePath=None, restorePath=None):

    datasetPath = '/media/yonsei/3A1FF8D034F3B6C8/ImageNet/'
    dataset = imagenetDataset.imagenetDataset(dataPath=datasetPath, classNum=nolboConfig['classDim'])
    dataset.setImageSize((224, 224))
    #
    nb = nolboModule.darknet_classifier(classNum=nolboConfig['classDim'])
    if restorePath != None:
        print 'restore weights...'
        # nb.restoreEncoderCore(restorePath)
        # nb.restoreEncoderLastLayer(restorePath)
        # nb.restoreDecoder(restorePath)
        # np.restorePriornet(restorePath)
        nb.restoreNetworks(restorePath)

    loss = 0.0
    acc, top5Acc = 0.0, 0.0
    epoch = 0
    iteration = 0
    run_time = 0.0
    if learningRate == None:
        learningRate = 0.0001

    print 'start training...'
    while epoch < training_epoch:

        start = time.time()
        # np.random.shuffle(imgSizeList)
        # dataset.setImageSize(imgSizeList[0])
        # dataset.setImageSize([112,112])
        batchData = dataset.getNextBatchPar(batchSize=batchSize)
        batchData['learningRate'] = learningRate
        # learningRate = learningRate*0.99995
        # lr = lr*0.9990
        epochCurr = dataset._epoch
        dataStart = dataset._dataStart
        dataLength = dataset._dataLength

        if ((iteration+1) % 2000 == 0 and (iteration+1) != 1) or epochCurr != epoch:
            print ''
            gc.collect()
            iteration = 0
            loss = loss * 0.0
            run_time = 0.0
            if savePath != None:
                print 'save model...'
                nb.saveNetworks(savePath)
        epoch = epochCurr

        # lossTemp, accTemp = loss, acc
        lossTemp, accTemp, top5AccTemp = nb.fit(batchData)
        end = time.time()

        acc = float(acc * iteration + accTemp) / float(iteration + 1.0)
        top5Acc = float(top5Acc * iteration + top5AccTemp) / float(iteration + 1.0)
        loss = float(loss * iteration + lossTemp) / float(iteration + 1.0)
        run_time = (run_time * iteration + (end - start)) / (iteration + 1.0)
        sys.stdout.write("Epoch:{:03d} iter:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("cur/tot:{:07d}/{:07d} ".format(dataStart, dataLength))
        sys.stdout.write(
            "loss={:.7f} ".format(loss))
        sys.stdout.write("acc=a:{:.5f},t:{:.5f}   \r".format(acc, top5Acc))
        sys.stdout.flush()

        iteration = iteration + 1.0


if __name__=="__main__":
    sys.exit(trainNolboClassifier(
        nolboConfig=nolboConfig,
        batchSize=256,
        training_epoch = 1000,
        learningRate = 0.1,
        savePath='weights/nolbo_classifier/',
        # restorePath=None
        restorePath = 'weights/nolbo_classifier/'
    ))