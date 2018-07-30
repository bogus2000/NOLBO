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
        self._nolboConfig['encoder']['inputImgDim'] = imgSize

    def getNextBatch(self, batchSize):
        self._gridSize = [
            self._nolboConfig['encoder']['inputImgDim'][0]/(2**self._nolboConfig['maxPoolNum']),
            self._nolboConfig['encoder']['inputImgDim'][1]/(2**self._nolboConfig['maxPoolNum'])]
        addedObjNum = 0
        img2DPathList = []
        inputImages, bboxImages = [],[]
        outputImages, classList, instList, EulerRad = [],[],[],[]
        outputImagesOrg = []
        for dataPath in self._dataPathList[self._dataStart:]:
            dataFrom = dataPath[0]
            dataPointPath = dataPath[1]
            # print dataPointPath
            objectFolderList = os.listdir(dataPointPath)
            objectFolderList.sort(key=datasetUtils.natural_keys)
            selectedObjInfo = []
            selectedObjFolderPath = []
            for objectFolder in objectFolderList:
                if os.path.isdir(os.path.join(dataPointPath, objectFolder)):
                    objFolderPath = os.path.join(dataPointPath, objectFolder)
                    objInfoPath = os.path.join(objFolderPath, 'objInfo.txt')
                    with open(objInfoPath) as objInfoFilePointer:
                        objInfo = objInfoFilePointer.readline()
                    className = objInfo.split(" ")[0]
                    if className in self._classConvertion:
                        # print className
                        selectedObjInfo.append(objInfo)
                        selectedObjFolderPath.append(objFolderPath)
            if len(selectedObjInfo)>batchSize:
                # print 'pass'
                pass
            elif addedObjNum+len(selectedObjInfo)>batchSize:
                break
            elif len(selectedObjInfo)>0:
                try:
                    img2DPath = os.path.join(dataFrom, selectedObjInfo[0].split(" ")[1])
                    inputImage = cv2.imread(img2DPath, cv2.IMREAD_COLOR)
                    imageRow, imageCol, channel = inputImage.shape
                    imageRow, imageCol = float(imageRow), float(imageCol)
                    dRowRescale, dColRescale = 0,0
                    if np.random.rand()<0.7:
                        rescaleRatio = np.random.uniform(low=0.0, high=0.2)
                        if np.random.rand()<0.5:
                            rowCropMin,rowCropMax=int(imageRow*rescaleRatio),int(imageRow*(1-rescaleRatio))
                            colCropMin,colCropMax=int(imageCol*rescaleRatio),int(imageCol*(1-rescaleRatio))
                            dRowRescale = -rowCropMin
                            dColRescale = -colCropMin
                            inputImage = inputImage[rowCropMin:rowCropMax, colCropMin:colCropMax, :]
                        else:
                            #smaller size
                            rescaleRatio = rescaleRatio*2.0
                            left,right = int(imageRow*rescaleRatio), int(imageRow*rescaleRatio)
                            top,bottom = int(imageCol*rescaleRatio), int(imageCol*rescaleRatio)
                            dRowRescale = top
                            dColRescale = left
                            inputImage = cv2.copyMakeBorder(inputImage,top=top,bottom=bottom,left=left,right=right,
                                                            borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
                        imageRow, imageCol, channel = inputImage.shape
                        imageRow, imageCol = float(imageRow), float(imageCol)

                    dAngle, dRad = 0.0, 0.0
                    cosdRad, sindRad = 1.0, 0.0
                    # if np.random.rand() < 0.8:
                    #     dAngle = np.random.uniform(low=-30.0, high=30.0)
                    #     dRad = dAngle / 180.0 * np.pi
                    #     cosdRad, sindRad = np.cos(dRad), np.sin(dRad)
                    #     rotationMat = cv2.getRotationMatrix2D((imageCol / 2, imageRow / 2), dAngle, 1)
                    #     inputImage = cv2.warpAffine(inputImage, rotationMat, (int(imageCol), int(imageRow)))

                    dRow,dCol = 0,0
                    if np.random.rand()<0.7:
                        rowTransMax, colTransMax = imageRow*0.2, imageCol*0.2
                        dRow, dCol= np.random.randint(-int(rowTransMax),int(rowTransMax)), np.random.randint(-int(colTransMax),int(colTransMax))
                        translationMat = np.float32([[1,0,dCol],[0,1,dRow]])
                        inputImage = cv2.warpAffine(inputImage, translationMat, (int(imageCol),int(imageRow)))

                    if np.random.rand() < 0.7:
                        inputImage = datasetUtils.imgAug(inputImage, crop=False, flip=False, gaussianBlur=True)

                    inputImage = cv2.resize(
                        inputImage,
                        dsize=(self._nolboConfig['encoder']['inputImgDim'][1], self._nolboConfig['encoder']['inputImgDim'][0]),
                        interpolation=cv2.INTER_CUBIC
                    )
                    objOrderingImage = -1 * np.ones(self._gridSize + [self._nolboConfig['predictorNumPerGrid']])
                    bboxImage = np.zeros(self._gridSize + [self._nolboConfig['predictorNumPerGrid'], self._nolboConfig['bboxDim']])
                    outputImagesPerImg, outputImagesOrgPerImg, classListPerImg, instListPerImg, EulerRadPerImg = [],[],[],[],[]
                    outputImagesIdx = 0
                    for i in range(len(selectedObjInfo)):
                        objInfo = selectedObjInfo[i]
                        objFolderPath = selectedObjFolderPath[i]
                        objClassOrg,imgPath,CADModelPath,colMin,rowMin,colMax,rowMax,azimuth,elevation,in_plane_rot=objInfo.split(" ")
                        rowMin, rowMax = float(rowMin), float(rowMax)
                        colMin, colMax = float(colMin), float(colMax)

                        # Rescale(image random sizing)
                        rowMin,rowMax = rowMin+dRowRescale, rowMax+dRowRescale
                        colMin,colMax = colMin+dColRescale, colMax+dColRescale
                        # in_plane_rot - degree
                        in_plane_rot = float(in_plane_rot) + dAngle
                        # rotation
                        rowMin,rowMax = rowMin-imageRow/2.0,rowMax-imageRow/2.0
                        colMin,colMax = colMin-imageCol/2.0,colMax-imageCol/2.0
                        sindRad = -sindRad # convert to the image coordinate
                        col11 = colMin*cosdRad - rowMin*sindRad
                        row11 = colMin*sindRad + rowMin*cosdRad
                        col12 = colMax*cosdRad - rowMin*sindRad
                        row12 = colMax*sindRad + rowMin*cosdRad
                        col21 = colMin*cosdRad - rowMax*sindRad
                        row21 = colMin*sindRad + rowMax*cosdRad
                        col22 = colMax*cosdRad - rowMax*sindRad
                        row22 = colMax*sindRad + rowMax*cosdRad
                        rowMin,rowMax = np.min((row11,row12,row21,row22)), np.max((row11,row12,row21,row22))
                        colMin,colMax = np.min((col11,col12,col21,col22)), np.max((col11,col12,col21,col22))
                        rowMin,rowMax = rowMin+imageRow/2.0, rowMax+imageRow/2.0
                        colMin,colMax = colMin+imageCol/2.0, colMax+imageCol/2.0
                        # translation
                        rowMin, rowMax = rowMin + dRow, rowMax + dRow
                        colMin, colMax = colMin + dCol, colMax + dCol

                        rowCenterOnGrid = (rowMax + rowMin)/2.0*self._gridSize[0]/imageRow
                        colCenterOnGrid = (colMax + colMin)/2.0*self._gridSize[1]/imageCol
                        rowIdxOnGrid = int(rowCenterOnGrid)
                        colIdxOnGrid = int(colCenterOnGrid)
                        # print 'objIdx:{:01d}'.format(i)
                        # print rowMin, rowMax, colMin, colMax
                        # print imageRow, imageCol
                        # print rowCenterOnGrid, colCenterOnGrid
                        # print rowIdxOnGrid, colIdxOnGrid
                        dx, dy = colCenterOnGrid - colIdxOnGrid, rowCenterOnGrid - rowIdxOnGrid # (0,1)
                        bboxHeight = np.min((1.0, (rowMax-rowMin)/imageRow)) # (0,1)
                        bboxWidth = np.min((1.0, (colMax-colMin)/imageCol)) # (0,1)
                        # if not (rowIdxOnGrid>=0 and rowIdxOnGrid<self._gridSize[0]) or not (colIdxOnGrid>=0 and colIdxOnGrid<self._gridSize[1]):
                        #     print '???', img2DPath, str(outputImagesIdx)+'/'+str(len(selectedObjInfo)), 'r,c:'+str(rowIdxOnGrid)+','+str(colIdxOnGrid), 'dr,dc:'+str(dRow)+','+str(dCol), 'drS,dcS:'+str(dRowRescale)+','+str(dColRescale)
                        for predictorIdx in range(self._nolboConfig['predictorNumPerGrid']):
                            # is center on grids? and is the grid not occupied?
                            if (rowIdxOnGrid>=0 and rowIdxOnGrid<self._gridSize[0]) \
                                and (colIdxOnGrid>=0 and colIdxOnGrid<self._gridSize[1])\
                                and (bboxImage[rowIdxOnGrid, colIdxOnGrid, predictorIdx, 4] != 1):
                                    bboxImage[rowIdxOnGrid, colIdxOnGrid, predictorIdx, 4] = 1
                                    bboxImage[rowIdxOnGrid, colIdxOnGrid, predictorIdx, 0] = bboxHeight
                                    bboxImage[rowIdxOnGrid, colIdxOnGrid, predictorIdx, 1] = bboxWidth
                                    bboxImage[rowIdxOnGrid, colIdxOnGrid, predictorIdx, 2] = dx
                                    bboxImage[rowIdxOnGrid, colIdxOnGrid, predictorIdx, 3] = dy
                                    #object class vector
                                    objClass = self._classConvertion[objClassOrg]
                                    objClassIdx = self._classDict[objClass] - 1
                                    objClassVector = np.zeros(self._nolboConfig['classDim'])
                                    objClassVector[objClassIdx] = 1
                                    #object inst vector
                                    instIdx = self._instDict[objClass][CADModelPath]
                                    objInstVector = np.zeros(self._nolboConfig['instDim'])
                                    objInstVector[instIdx] = 1
                                    #Euler angle - radian
                                    objEulerRad = np.array([float(azimuth), float(elevation), float(in_plane_rot)])/180.0*np.pi
                                    #object 3D shape
                                    outputImage = np.load(os.path.join(objFolderPath, 'voxel.npy'))
                                    outputImageOrg = np.load(os.path.join(objFolderPath, 'voxel_org.npy'))

                                    #append items
                                    outputImagesPerImg.append(outputImage)
                                    outputImagesOrgPerImg.append(outputImageOrg)
                                    classListPerImg.append(objClassVector)
                                    instListPerImg.append(objInstVector)
                                    EulerRadPerImg.append(objEulerRad)
                                    #set obj order
                                    objOrderingImage[rowIdxOnGrid, colIdxOnGrid, predictorIdx] = outputImagesIdx
                                    outputImagesIdx += 1
                                    break

                    if outputImagesIdx>0:
                        addedObjNum += outputImagesIdx
                        for gridRow in range(self._gridSize[0]):
                            for gridCol in range(self._gridSize[1]):
                                for detectorIdx in range(self._nolboConfig['predictorNumPerGrid']):
                                    objOrder = int(objOrderingImage[gridRow,gridCol, detectorIdx])
                                    if objOrder>=0:
                                        outputImages.append(outputImagesPerImg[objOrder])
                                        outputImagesOrg.append(outputImagesOrgPerImg[objOrder])
                                        classList.append(classListPerImg[objOrder])
                                        instList.append(instListPerImg[objOrder])
                                        EulerRad.append(EulerRadPerImg[objOrder])
                        img2DPathList.append(img2DPath)
                        inputImages.append(inputImage)
                        bboxImages.append(bboxImage)
                except:
                    # img2DPath = os.path.join(dataFrom, selectedObjInfo[0].split(" ")[1])
                    # print ''
                    # print img2DPath
                    # return
                    pass
            self._dataStart += 1
            if self._dataStart >= len(self._dataPathList):
                self._epoch += 1
                self._dataStart = 0
                self._dataPathShuffle()
                break
        img2DPathList = np.array(img2DPathList)
        inputImages = np.array(inputImages)
        # if inputImages.shape[0] == 0:
        #     print 'batch size = 0!!'
        #     print self._dataPathList[self._dataStart-1]
        bboxImages = np.array(bboxImages)
        outputImages = np.array(outputImages)
        outputImagesOrg = np.array(outputImagesOrg)
        classList = np.array(classList)
        instList = np.array(instList)
        EulerRad = np.array(EulerRad)
        batchDict = {
            'inputImages':inputImages,
            'bboxImages':bboxImages,
            'outputImages':outputImages,
            'outputImagesOrg':outputImagesOrg,
            'classList':classList,
            'instList':instList,
            'EulerRad':EulerRad,
            'img2DPath':img2DPathList,
        }

        return batchDict



















