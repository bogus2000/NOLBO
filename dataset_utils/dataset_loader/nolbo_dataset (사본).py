import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle, sys
import pandas
import dataset_utils.datasetUtils as datasetUtils
from pathos.multiprocessing import ProcessingPool as Pool

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

classConvertion = {
    'aeroplane' : 'aeroplane',
    'bed' : 'bed',
    'bench' : 'bench',
    'bicycle' : 'bicycle',
    'boat' : 'boat',
    'bookshelf' : 'bookshelf',
    'bookcase' : 'bookshelf',
    'bottle' : 'bottle',
    'bus' : 'bus',
    'cabinet' : 'cabinet',
    'can' : 'can',
    'cap' : 'cap',
    'car' : 'car',
    'chair' : 'chair',
    'computer' : 'computer',
    'cup' : 'cup',
    'desk' : 'desk',
    'diningtable' : 'table',
    'table' : 'table',
    'door' : 'door',
    'filing_cabinet' : 'cabinet',
    'fire_extinguisher' : 'fire_extinguisher',
    'jar' : 'jar',
    'keyboard' : 'keyboard',
    'laptop' : 'laptop',
    'microwave' : 'microwave',
    'motorbike' : 'motorbike',
    'mouse' : 'mouse',
    'piano' : 'piano',
    'pillow' : 'pillow',
    'printer' : 'printer',
    'refrigerator' : 'refrigerator',
    'rode_pole' : 'rode_pole',
    'sofa' : 'sofa',
    'speaker' : 'speaker',
    'suitcase' : 'suitcase',
    'teapot' : 'teapot',
    'toilet' : 'toilet',
    'train' : 'train',
    'trash_bin' : 'trash_bin',
    'tub' : 'bathtub',
    'tvmonitor' : 'tvmonitor',
    'wardrobe' : 'wardrobe',
}

classDict = {
    'aeroplane' : 1,
    'bed' : 2,
    'bench' : 3,
    'bicycle' : 4,
    'boat' : 5,
    'bookshelf' : 6,
    'bottle' : 7,
    'bus' : 8,
    'cabinet' : 9,
    'can' : 10,
    'cap' : 11,
    'car' : 12,
    'chair' : 13,
    'computer' : 14,
    'cup' : 15,
    'desk' : 16,
    'table' : 17,
    'door' : 18,
    'fire_extinguisher' : 19,
    'jar' : 20,
    'keyboard' : 21,
    'laptop' : 22,
    'microwave' : 23,
    'motorbike' : 24,
    'mouse' : 25,
    'piano' : 26,
    'pillow' : 27,
    'printer' : 28,
    'refrigerator' : 29,
    'rode_pole' : 30,
    'sofa' : 31,
    'speaker' : 32,
    'suitcase' : 33,
    'teapot' : 34,
    'toilet' : 35,
    'train' : 36,
    'trash_bin' : 37,
    'bathtub' : 38,
    'tvmonitor' : 39,
    'wardrobe' : 40,
}

# from memory_profiler import profile
def loadImg(imgPath, imgSize, bbox):
    inputImage = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    colMin, rowMin, colMax, rowMax = bbox
    imageCanvas = inputImage[int(rowMin):int(rowMax), int(colMin):int(colMax)]
    imageResult = datasetUtils.imgAug(imageCanvas,flip=False)
    imageResult = cv2.resize(imageResult, imgSize)
    return imageResult

def load3DShape(obj3DShapePath):
    obj3DShape = np.load(obj3DShapePath)
    return obj3DShape

class nolboDatasetSingleObject(object):
    def __init__(self, nolboConfig,
                 dataPath_ObjectNet3D=None, dataPath_Pascal3D=None
                 ):
        self._nolboConfig = nolboConfig
        self._dataPath_ObjectNet3D = dataPath_ObjectNet3D
        self._dataPath_Pascal3D = dataPath_Pascal3D
        self._dataStart = 0
        self._dataLength = 0
        self._epoch = 0
        self._dataPathList = []
        self._classConvertion = None
        self._classDict = None
        self._instDict = None

        self._createDict()
        self._loadDataPath()
        self._dataPathShuffle()

    def _createDict(self):
        print 'create dict...'
        self._classConvertion = classConvertion
        self._classDict = classDict
        self._instDict = dict()
        if os.path.isdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD')):
            print 'ObjectNet3D...'
            classList = os.listdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD'))
            classList.sort(key=datasetUtils.natural_keys)
            for orgClassName in classList:
                if os.path.isdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD', orgClassName)):
                    if orgClassName in self._classConvertion:
                        className = self._classConvertion[orgClassName]
                        if className not in self._instDict:
                            self._instDict[className] = dict()
                        CADModelList = os.listdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD', orgClassName))
                        CADModelList.sort(key=datasetUtils.natural_keys)
                        instIdx = 0
                        for CADModel in CADModelList:
                            if CADModel.endswith(".pcd"):
                                CADModelPath = os.path.join('CAD', orgClassName, os.path.splitext(CADModel)[0])
                                if CADModelPath not in self._instDict[className]:
                                    self._instDict[className][CADModelPath] = instIdx
                                    instIdx += 1
            print 'dict ready!'

    def _loadDataPath(self):
        print 'load data path...'
        dataFromList = [self._dataPath_Pascal3D, self._dataPath_ObjectNet3D]
        for dataFrom in dataFromList:
            if dataFrom!=None and os.path.isdir(os.path.join(dataFrom, 'training_data')):
                datasetTypeList = os.listdir(os.path.join(dataFrom, 'training_data'))
                datasetTypeList.sort(key=datasetUtils.natural_keys)
                print dataFrom
                for datasetType in datasetTypeList:
                    if os.path.isdir(os.path.join(dataFrom, 'training_data', datasetType)):
                        dataPointList = os.listdir(os.path.join(dataFrom, 'training_data', datasetType))
                        dataPointList.sort(key=datasetUtils.natural_keys)
                        dataPointIdx = 1
                        totalDataPointNum = len(dataPointList)
                        for dataPoint in dataPointList:
                            if os.path.isdir(os.path.join(dataFrom, 'training_data', datasetType, dataPoint)):
                                objFolderList = os.listdir(os.path.join(dataFrom, 'training_data', datasetType, dataPoint))
                                objFolderList.sort(key=datasetUtils.natural_keys)
                                for objFolder in objFolderList:
                                    objInfoPath = os.path.join(dataFrom, 'training_data', datasetType, dataPoint, objFolder, 'objInfo.txt')
                                    with open(objInfoPath) as objInfoFilePointer:
                                        className = objInfoFilePointer.readline().split(" ")[0]
                                    if className in self._classConvertion:
                                        self._dataPathList.append(
                                            [dataFrom ,os.path.join(dataFrom, 'training_data', datasetType, dataPoint, objFolder)])
                                    # self._dataPathList.append(
                                    #     [dataFrom, os.path.join(dataFrom, 'training_data', datasetType, dataPoint, objFolder)])
                                sys.stdout.write(datasetType + " {:05d}/{:05d}\r".format(dataPointIdx, totalDataPointNum))
                                dataPointIdx += 1
                        print ''
                # sys.stdout.write("\n")
        print 'done!'
        self._dataLength = len(self._dataPathList)

    def _dataPathShuffle(self):
        print ''
        print 'data path shuffle...'
        self._dataStart = 0
        np.random.shuffle(self._dataPathList)
        self._dataLength = len(self._dataPathList)
        print 'done! :', self._dataLength

    def setInputImageSize(self, imgSize):
        self._nolboConfig['inputImgDim'] = imgSize

    def getNextBatch(self, batchSize):
        checkedDataNum = 0
        addedDataNum = 0
        inputImages = []
        classList, instList, AEIAngle = [],[],[]
        outputImages, outputImagesOrg = [], []
        if self._dataStart + batchSize >= self._dataLength:
            self._epoch += 1
            self._dataPathShuffle()
        for dataFromAndPath in self._dataPathList[self._dataStart:]:
            if addedDataNum>=batchSize:
                break
            dataFrom = dataFromAndPath[0]
            dataPath = dataFromAndPath[1]
            objInfoPath = os.path.join(dataPath, 'objInfo.txt')
            obj3DShapePath = os.path.join(dataPath, 'voxel.npy')
            objOrg3DShapePath = os.path.join(dataPath, 'voxel_org.npy')
            with open(objInfoPath) as objInfoFilePointer:
                objInfo = objInfoFilePointer.readline()
            objClassOrg, imgPath, CADModelPath, colMin, rowMin, colMax, rowMax, azimuth, elevation, in_plane_rot = objInfo.split(" ")
            if objClassOrg in self._classConvertion:
                try:
                    # 2D image
                    imgPath = os.path.join(dataFrom, imgPath)
                    inputImage = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                    colMin, rowMin, colMax, rowMax = 0.9*float(colMin), 0.9*float(rowMin), 1.1*float(colMax), 1.1*float(rowMax)
                    bboxLengthSqr = np.max(((colMax-colMin), (rowMax-rowMin))) / 2.0
                    colCenter, rowCenter = (colMin+colMax)/2.0, (rowMin+rowMax)/2.0
                    colMinSqr, rowMinSqr = int(colCenter-bboxLengthSqr), int(rowCenter-bboxLengthSqr)
                    colMaxSqr, rowMaxSqr = int(colCenter+bboxLengthSqr), int(rowCenter+bboxLengthSqr)
                    colMin, rowMin = int(np.max((0.0, colMin))), int(np.max((0.0, rowMin)))
                    colMax, rowMax = int(np.min((len(inputImage[0]), colMax))), int(np.min((len(inputImage), rowMax)))
                    imageCanvas = inputImage[rowMin:rowMax, colMin:colMax]
                    # image augmentation
                    if np.random.rand() < 0.5:
                        imageCanvas = datasetUtils.imgAug(imageCanvas, crop=True, flip=False, gaussianBlur=True)
                    # #image normalization
                    # imageCanvas = (imageCanvas / 255.0) * 2.0 - 1.0
                    #append padding for image
                    imageCanvas = cv2.copyMakeBorder(imageCanvas,
                                                     top=np.max((0, rowMin - rowMinSqr))+np.random.randint(0,1+int(bboxLengthSqr*0.1)),
                                                     bottom=np.max((0, rowMaxSqr - rowMax))+np.random.randint(0,1+int(bboxLengthSqr*0.1)),
                                                     left=np.max((0, colMin - colMinSqr))+np.random.randint(0,1+int(bboxLengthSqr*0.1)),
                                                     right=np.max((0, colMaxSqr - colMax))+np.random.randint(0,1+int(bboxLengthSqr*0.1)),
                                                     borderType=cv2.BORDER_CONSTANT,
                                                     value=[0,0,0])
                    imageCanvas = cv2.resize(
                        imageCanvas,
                        dsize=(self._nolboConfig['inputImgDim'][1], self._nolboConfig['inputImgDim'][0]),
                        interpolation=cv2.INTER_CUBIC
                    )
                    if imageCanvas.shape[-1] != 3:
                        imageCanvas = np.reshape(imageCanvas,(self._nolboConfig['inputImgDim'][0],self._nolboConfig['inputImgDim'][1],1))
                        imageCanvas = np.concatenate([imageCanvas, imageCanvas, imageCanvas], axis=-1)
                    imageCanvas = imageCanvas.reshape(self._nolboConfig['inputImgDim'])

                    # class index
                    objClass = self._classConvertion[objClassOrg]
                    objClassIdx = self._classDict[objClass] - 1
                    objClassVector = np.zeros(self._nolboConfig['classDim'])
                    objClassVector[objClassIdx] = 1

                    # inst, rotated 3D shape, Euler angle
                    instIdx = -1
                    instIdx = self._instDict[objClass][CADModelPath]
                    objInstVector = np.zeros(self._nolboConfig['instDim'])
                    objInstVector[instIdx] = 1

                    classList.append(objClassVector.copy())
                    inputImages.append(imageCanvas.copy())
                    instList.append(objInstVector.copy())
                    obj3DShape = np.load(obj3DShapePath)
                    outputImages.append(obj3DShape.copy())
                    objAEI = np.array([float(azimuth),float(elevation),float(in_plane_rot)])/180.0*np.pi # angle to radian
                    AEIAngle.append(objAEI.copy())

                    # original 3D Shape (without rotation)
                    objOrg3DShape = np.load(objOrg3DShapePath)
                    outputImagesOrg.append(objOrg3DShape.copy())

                    addedDataNum += 1
                except:
                    pass
            checkedDataNum += 1

        self._dataStart = self._dataStart+checkedDataNum

        batchDict = {
            'classList' : np.array(classList).astype('float'),
            'instList' : np.array(instList).astype('float'),
            'AEIAngle' : np.array(AEIAngle).astype('float'),
            'inputImages' : np.array(inputImages).astype('float'),
            'outputImages' : np.array(outputImages).astype('float'),
            'outputImagesOrg' : np.array(outputImagesOrg).astype('float'),
        }

        return batchDict


class nolboDatasetMultiObject(object):
    def __init__(self, nolboConfig,
                 dataPath_ObjectNet3D=None, dataPath_Pascal3D=None
                 ):
        self._nolboConfig = nolboConfig
        self._dataPath_ObjectNet3D = dataPath_ObjectNet3D
        self._dataPath_Pascal3D = dataPath_Pascal3D
        self._dataStart = 0
        self._dataLength = 0
        self._epoch = 0
        self._dataPathList = []
        self._classConvertion = None
        self._classDict = None
        self._instDict = None

        self._createDict()
        self._loadDataPath()


    def _createDict(self):
        print 'create dict...'
        self._classConvertion = classConvertion
        self._classDict = classDict
        self._instDict = dict()
        if os.path.isdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD')):
            print 'ObjectNet3D...'
            classList = os.listdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD'))
            classList.sort(key=datasetUtils.natural_keys)
            for orgClassName in classList:
                if os.path.isdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD', orgClassName)):
                    if orgClassName in self._classConvertion:
                        className = self._classConvertion[orgClassName]
                        if className not in self._instDict:
                            self._instDict[className] = dict()
                        CADModelList = os.listdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD', orgClassName))
                        CADModelList.sort(key=datasetUtils.natural_keys)
                        instIdx = 0
                        for CADModel in CADModelList:
                            if CADModel.endswith(".pcd"):
                                CADModelPath = os.path.join('CAD', orgClassName, os.path.splitext(CADModel)[0])
                                if CADModelPath not in self._instDict[className]:
                                    self._instDict[className][CADModelPath] = instIdx
                                    instIdx += 1
            print 'dict ready!'

    def _loadDataPath(self):
        print 'load data path...'
        dataFromList = [self._dataPath_Pascal3D, self._dataPath_ObjectNet3D]
        for dataFrom in dataFromList:
            if dataFrom!=None and os.path.isdir(os.path.join(dataFrom, 'training_data')):
                datasetTypeList = os.listdir(os.path.join(dataFrom, 'training_data'))
                datasetTypeList.sort(key=datasetUtils.natural_keys)
                print dataFrom
                for datasetType in datasetTypeList:
                    if os.path.isdir(os.path.join(dataFrom, 'training_data', datasetType)):
                        dataPointList = os.listdir(os.path.join(dataFrom, 'training_data', datasetType))
                        dataPointList.sort(key=datasetUtils.natural_keys)
                        dataPointIdx = 1
                        totalDataPointNum = len(dataPointList)
                        for dataPoint in dataPointList:
                            if os.path.isdir(os.path.join(dataFrom, 'training_data', datasetType, dataPoint)):
                                dataPointPath = os.path.join(dataFrom, 'training_data', datasetType, dataPoint)
                                self._dataPathList.append([dataFrom, dataPointPath])
                                sys.stdout.write(datasetType + " {:05d}/{:05d}\r".format(dataPointIdx, totalDataPointNum))
                                dataPointIdx += 1
                        print ''
                # sys.stdout.write("\n")
        print 'done!'
        self._dataLength = len(self._dataPathList)

    def _dataPathShuffle(self):
        print ''
        print 'data path shuffle...'
        self._dataStart = 0
        np.random.shuffle(self._dataPathList)
        self._dataLength = len(self._dataPathList)
        print 'done! :', self._dataLength

    def setInputImageSize(self, imgSize):
        self._nolboConfig['inputImgDim'] = imgSize

    def getNextBatch(self, batchSize):
        self._gridSize = [
            self._nolboConfig['inputImgDim'][1]/(2**self._nolboConfig['maxPoolNum']),
            self._nolboConfig['inputImgDim'][0]/(2**self._nolboConfig['maxPoolNum'])]
        addedDataNum = 0
        addedObjNum = 0
        checkedDataNum = 0
        inputImages, bboxHWXY, objectness = [],[],[]
        outputImages, classList, instList, EulerAngle = [],[],[],[]
        outputImagesOrg = []
        for dataPath in self._dataPathList[self._dataStart:]:
            dataFrom = dataPath[0]
            dataPointPath = dataPath[1]
            objectFolderList = os.listdir(dataPointPath)
            objectFolderList.sort(key=datasetUtils.natural_keys)
            selectedObjInfo = []
            selectedObjFolderPath = []
            for objectFolder in objectFolderList:
                if os.path.isdir(os.path.join(dataPointPath, objectFolder)):
                    objFolderPath = os.path.join(dataPointPath, objectFolder, 'objInfo.txt')
                    with open(objFolderPath) as objInfoFilePointer:
                        objInfo = objInfoFilePointer.readline()
                    className = objInfo[0]
                    if className in self._classConvertion:
                        selectedObjInfo.append(objInfo)
                        selectedObjFolderPath.append(objFolderPath)
            addedObjNum += len(selectedObjInfo)
            if addedObjNum+len(selectedObjInfo)>batchSize:
                break
            elif len(selectedObjInfo)>0:
                addedDataNum += 1
                for i in range(len(selectedObjInfo)):
                    objInfo = selectedObjInfo[i]
                    objFolderPath = selectedObjFolderPath[i]
                    objClassOrg,imgPath,CADModelPath,colMin,rowMin,colMax,rowMax,azimuth,elevation,in_plane_rot=objInfo.split(" ")
                    


            if dataPath == self._dataPathList[-1]:
                self._epoch += 1
                self._dataStart = 0
                self._dataPathShuffle()
                break




















