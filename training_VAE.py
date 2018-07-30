import numpy as np
import time, sys
import dataset_utils.dataset_loader.vae3D_dataset as vaeDataset
import src.module.vae as vae
import tensorflow as tf

classDim = 40
instDim = 1000

nolboConfig = {
    'classDim' : classDim,
    'instDim' : instDim,
}

# =========== autoencoder architecture ===============
networkStructure = {
    'class':True, 'zClassDim':64, 'classDim':classDim,
    'inst':True, 'zInstDim':64, 'instDim':instDim,
    'rot':True, 'zRotDim':3, 'rotDim':3,
    'isTraining' : True,
    'nameScope' : 'nolbo',
    'encoder' : {
        'inputImgDim':[64,64,64,1],
        'trainable':True,
        'activation':tf.nn.elu,
        'filterNumList':[64,128,256,512, 2*(64+64+3)],
        'kernelSizeList':[4,4,4,4,4],
        'stridesList':[2,2,2,2,1],
        'lastPool' : 'average',
    },
    'decoder' : {
        'outputImgDim':[64,64,64,1],
        'trainable':True,
        'activation':tf.nn.elu,
        'filterNumList':[512,256,128,64,1],
        'kernelSizeList':[4,4,4,4,4],
        'stridesList':[1,2,2,2,2],
        'lastLayerActivation':tf.nn.sigmoid
    },
    'prior':{
        'hiddenLayerNum':3,
        'trainable':True,
        'activation':tf.nn.elu,
        'constLogVar':0.0,
    },
}

def trainVAE(networkStructure, batchSize, trainingEpoch, learningRate, savePath, restorePath, loadOrg3DShape=False):
    dataPath = '/media/yonsei/120GB_SSD/CAD_training_data_numpy/'
    dataset = vaeDataset.ObjectNet3D_voxelRotationDataset(
        trainingDataPath=dataPath,
        partitionNum=6,
        loadVoxOrg=loadOrg3DShape
    )

    model = vae.VAE3D(network_architecture=networkStructure)
    if restorePath!=None:
        # model.restoreEncoder(restorePath=restorePath)
        # model.restorePriornet(restorePath=restorePath)
        model.restoreNetworks(restorePath=restorePath)
    loss = np.zeros(6)
    pr = np.zeros(2)
    epoch = 0
    epochCurr = 0
    iteration = 0
    run_time = 0.0

    print 'start training...'
    while epoch < trainingEpoch:
        start = time.time()
        batchData = dataset.getNextBatch(batchSize=batchSize)
        if learningRate == None:
            learningRate = 0.0001
        batchData['learningRate'] = learningRate
        inputData = {
            'learningRate' : learningRate,
            'inputImages' : batchData['inputImages'],
            'outputImages' : np.concatenate(
                [
                # batchData['outputImages'],
                # batchData['outputImages'],
                # batchData['outputImages'], #batchData['outputImages'],
                batchData['outputImagesOrg'],
                # batchData['outputImages'],
                ], axis=0),
            'classList' : batchData['classList'],
            'instList' : batchData['instList'],
            'EulerAngleSinCos' : np.concatenate(
                [np.sin(batchData['EulerAngle']), np.cos(batchData['EulerAngle'])], axis=-1),
        }
        epochCurr = dataset._epoch
        dataStart = dataset._batchStart
        dataLength = dataset._partitionLength
        if epochCurr != epoch:
            print ''
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
        sys.stdout.write(
            "Ep:{:03d} it:{:04d} rt:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
        sys.stdout.write("par:{:02d}/{:02d} ".format(int(dataset._curruntPartition), int(dataset._partitionNum)))
        sys.stdout.write("cur/tot:{:05d}/{:05d} ".format(dataStart, dataLength))
        sys.stdout.write(
            "loss=tot:{:.3f},vox:{:.3f},reg:{:.3f},cin:{:.3f},rn:{:.3f},sc:{:.3f}".format(loss[0],loss[1],loss[2],loss[3],loss[4],loss[5]))
        sys.stdout.write(" p,r={:.4f},{:.4f}   \r".format(pr[0], pr[1]))
        sys.stdout.flush()

        if loss[0] != loss[0]:
            print ''
            return
        iteration = iteration + 1.0

        # if iteration % 1000 == 0 and iteration != 1:
        #     print ''
        #     iteration = 0
        #     loss = loss * 0.0
        #     pr = pr * 0.0
        #     run_time = 0.0
        #     # learningRate = learningRate*1.1
        #     # nb._setOptimizer(learningRate=learningRate)
        #     if savePath != None:
        #         print 'save model...'
        #         model.saveNetworks(savePath)

if __name__ == "__main__":
    sys.exit(trainVAE(
        networkStructure=networkStructure,
        batchSize=96,
        trainingEpoch=1000,
        learningRate=1e-7,
        # learningRate=1e-6,
        # learningRate=0.0001,
        savePath='weights/VAE3D/',
        restorePath='weights/VAE3D/',
        # restorePath=None,
        loadOrg3DShape=True,
    ))




















