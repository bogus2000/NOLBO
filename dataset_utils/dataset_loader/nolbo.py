import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import dataset_utils.datasetUtils as datasetUtils

###example of nolboConfig for nolboDataset
# nolboConfig = {
#     'inputImgDim':[448,448,1],
#     'maxPoolNum':5,
#     'predictorNumPerGrid':11,
#     'bboxDim':5,
#     'class':True, 'zClassDim':64, 'classDim':24,
#     'inst':True, 'zInstDim':64, 'instDim':1000,
#     'rot':True, 'zRotDim':3, 'rotDim':3,
#     'trainable':True,
#     'decoderStructure':{
#         'outputImgDim':[64,64,64,1],
#         'trainable':True,    
#         'filterNumList':[512,256,128,64,1],
#         'kernelSizeList':[4,4,4,4,4],
#         'stridesList':[1,2,2,2,2],
#         'activation':tf.nn.leaky_relu,
#         'lastLayerActivation':tf.nn.sigmoid
#     }
# }

class nolboDataset(object):
    def __init__(self, datasetPath, nolboConfig):
        self._datasetPath = datasetPath
        self._nolboConfig = nolboConfig
        self._dataStart = 0
        self._dataLength = 0
        self._epoch = 0
        self._dataPathList = []
        self._objClassAndInst = None
        self._loadDataPath()
        self._dataPathShuffle()
    def _loadDataPath(self):
        print 'load data path...'
        with open(os.path.join(self._datasetPath, '../metadata','objClassAndInst.pkl'), 'rb') as objClassAndInstFile:
            self._objClassAndInst = pickle.load(objClassAndInstFile)
            objClassAndInstFile.close()
        houseList = os.listdir(self._datasetPath)
        houseList.sort(key=datasetUtils.natural_keys)
        houseNum = 0
        for house in houseList:
            print houseNum, house
            houseNum += 1
            housePath = os.path.join(self._datasetPath, house)
            roomList = os.listdir(housePath)
            roomList.sort(key=datasetUtils.natural_keys)
            if len(roomList)!=0:
                for room in roomList:
                    depthVisualizePath = os.path.join(housePath, room, 'depthVisualize')
                    objInfoPath = os.path.join(housePath, room, 'objInfo')
                    voxel3DPath = os.path.join(housePath, room, 'voxel3D')
                    depthVisualizeList = os.listdir(depthVisualizePath)
                    objInfoList = os.listdir(objInfoPath)
                    voxel3DList = os.listdir(voxel3DPath)
                    if len(depthVisualizeList)!=0:
                        for viewIdx in range(len(depthVisualizeList)):
                            dViewNum = depthVisualizeList[viewIdx].split("_")[-1].split(".")[0]
                            oViewNum = objInfoList[viewIdx].split("_")[-1].split(".")[0]
                            vViewNum = voxel3DList[viewIdx].split("_")[-1]
                            if str(dViewNum)==str(oViewNum) and str(dViewNum)==str(vViewNum):
                                depthPath = os.path.join(depthVisualizePath, depthVisualizeList[viewIdx])
                                objifPath = os.path.join(objInfoPath, objInfoList[viewIdx])
                                voxelPath = os.path.join(voxel3DPath, voxel3DList[viewIdx])
                                dataPath = [depthPath, objifPath, voxelPath]
                                self._dataPathList.append(dataPath)
        self._dataLength = len(self._dataPathList)
        print 'done!'
        
    def _dataPathShuffle(self):
        print 'data path shuffle...'
        random.shuffle(self._dataPathList)
        print 'done!'
    def printDataPath(self):
        for i in range(len(self._dataPathList)):
            if i%10 == 0 or i==len(self._dataPathList)-1:
                print 'dataNum', i
                print self._dataPathList[i]
    def getNextBatch(self, maximumBatchSize):
        print 'get next batch data...'
        self._gridSize = [self._nolboConfig['inputImgDim'][0]/2**self._nolboConfig['maxPoolNum'], 
                          self._nolboConfig['inputImgDim'][1]/2**self._nolboConfig['maxPoolNum']]
        batchSize = 0
        addedDataPathNum = 0
        inputImages, bboxHWXY, objectness = [],[],[]
        outputImages, classList, instList, EulerAngle = [],[],[],[]
        if self._dataStart + maximumBatchSize >= self._dataLength:
            self._dataStart = 0
            self._dataPathShuffle()
        for dataPath in self._dataPathList[self._dataStart:]:
            depthPath, objifPath, voxelPath = dataPath
            with open(objifPath) as fPointer:
                objifs = fPointer.readlines()
            batchSize += len(objifs)
            if batchSize>maximumBatchSize:
                break
            elif batchSize<=maximumBatchSize:
                # print depthPath
                inputImage = cv2.imread(depthPath, 0)
                # print inputImage.shape
                inputImage = cv2.resize(inputImage, (self._nolboConfig['inputImgDim'][0], self._nolboConfig['inputImgDim'][1]))
                # print inputImage.shape
                inputImage = cv2.normalize(inputImage, inputImage, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
                # print inputImage.shape
                inputImage = inputImage.reshape(self._nolboConfig['inputImgDim'])
                inputImage = self._imageAugmentation(inputImage)
                # print inputImage.shape
                bboxHWXYImage = np.zeros(self._gridSize +[self._nolboConfig['predictorNumPerGrid'], self._nolboConfig['bboxDim']-1])
                objectnessImage = np.zeros(self._gridSize + [self._nolboConfig['predictorNumPerGrid'], 1])
                outputImageIdx = 0
                for objif in objifs:
                    # print objif
                    objShapePath, objClass, objNYUClass, rowMin, rowMax, colMin, colMax, heading, pitch, roll = objif.split(" ")
                    objShapePath = os.path.join(*(objShapePath.split('/')[6:]))
                    rowCenter = (float(rowMax) + float(rowMin))/2.0
                    colCenter = (float(colMax) + float(colMin))/2.0
                    rowCenter = np.max((np.min((rowCenter, 1.0-1e-10)), 0.0)) * self._gridSize[0]
                    colCenter = np.max((np.min((colCenter, 1.0-1e-10)), 0.0)) * self._gridSize[1]
                    rowIdxInGrid, colIdxInGrid = int(rowCenter), int(colCenter)
                    dx, dy = colCenter - colIdxInGrid, rowCenter - rowIdxInGrid
                    bboxHeight = np.min((1.0, float(colMax)))-np.max((0.0, float(colMin)))
                    bboxWidth = np.min((1.0, float(rowMax)))-np.max((0.0, float(rowMin)))
                    for predictorIdx in range(self._nolboConfig['predictorNumPerGrid']):
                        if objectnessImage[rowIdxInGrid,colIdxInGrid, predictorIdx, 0] == 0:
                            objectnessImage[rowIdxInGrid,colIdxInGrid, predictorIdx, 0] = 1
                            bboxHWXYImage[rowIdxInGrid,colIdxInGrid, predictorIdx, 0] = bboxHeight
                            bboxHWXYImage[rowIdxInGrid,colIdxInGrid, predictorIdx, 1] = bboxWidth
                            bboxHWXYImage[rowIdxInGrid,colIdxInGrid, predictorIdx, 2] = dx
                            bboxHWXYImage[rowIdxInGrid,colIdxInGrid, predictorIdx, 3] = dy
                    objClass, objInst = self._objClassAndInst[objShapePath]                    
                    objClassVector = [0]*30
                    objInstVector = [0]*1300
                    # print objClass, objInst
                    objClassVector[objClass] = 1
                    objInstVector[objInst] = 1
                    objHpr = [float(heading), float(pitch), float(roll)]                    
                    obj3DShape = np.loadtxt(os.path.join(voxelPath, str(outputImageIdx)+'.txt'), np.float32)
                    obj3DShape = np.reshape(obj3DShape, self._nolboConfig['decoderStructure']['outputImgDim'])
                    outputImages.append(obj3DShape)
                    classList.append(objClassVector)
                    instList.append(objInstVector)
                    EulerAngle.append(objHpr)
                inputImages.append(inputImage)
                bboxHWXY.append(bboxHWXYImage)
                objectness.append(objectnessImage)                
                addedDataPathNum += 1
        self._dataStart += addedDataPathNum
        inputImages = np.array(inputImages)
        bboxHWXY = np.array(bboxHWXY)
        objectness = np.array(objectness)
        outputImages = np.array(outputImages)
        classList = np.array(classList)
        instList = np.array(instList)
        EulerAngle = np.array(EulerAngle)        
        batchDict = {
            'inputImages':inputImages,
            'bboxHWXY':bboxHWXY,
            'objectness':objectness,
            'outputImages':outputImages,
            'classList':classList,
            'instList':instList,
            'EulerAngle':EulerAngle
        }
        print 'done!'
        return batchDict
        
    def _imageAugmentation(self, inputImages):
        noiseTypeList = ['gaussian', 'salt&pepper', 'poisson', 'speckle']
        random.shuffle(noiseTypeList)
        select = np.random.randint(0,2,len(noiseTypeList))
        for i in range(len(noiseTypeList)):
            if select[i] == 1:
                # print noiseTypeList[i]
                inputImages = datasetUtils.noisy(image=inputImages,noise_typ=noiseTypeList[i])
        return inputImages

# dataset = nolboDataset(datasetPath='/media/yonsei/4TB_HDD/dataset/suncg_data/training_data/', nolboConfig=nolboConfig)
# dataset._dataPathShuffle()
# batchData = dataset.getNextBatch(maximumBatchSize=32)
# print batchData.keys()
# for key in batchData.keys():
#     if key != 'outputImages':
#         print key, batchData[key].shape
