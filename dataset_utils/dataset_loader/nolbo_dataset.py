import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle, sys
import pandas
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
    'car' : 'car',
    'chair' : 'chair',
    'computer' : 'computer',
    'cup' : 'cup',
    'desk' : 'desk',
    'diningtable' : 'table',
    'door' : 'door',
    'filing_cabinet' : 'cabinet',
    'fire_extinguisher' : 'fire_extinguisher',
    'jar' : 'jar',
    'keyboard' : 'keyboard',
    'laptop' : 'laptop',
    'motorbike' : 'motorbike',
    'microwave' : 'microwave',
    'mouse' : 'mouse',
    'piano' : 'piano',
    'pillow' : 'pillow',
    'printer' : 'printer',
    'refrigerator' : 'refrigerator',
    'rode_pole' : 'rode_pole',
    'sofa' : 'sofa',
    'speaker' : 'speaker',
    'suitcase' : 'suitcase',
    'table' : 'table',
    'teapot' : 'teapot',
    'toilet' : 'toilet',
    'train' : 'train',
    'trash_bin' : 'trash_bin',
    'tub' : 'bathtub',
    'tvmonitor' : 'tvmonitor',
    'wardrobe' : 'wardrobe',
    ##################################################################
    # 'ashtray' : 'ashtray',
    # 'backpack' : 'backpack',
    # 'basket' : 'basket',
    # 'blackboard' : 'blackboard',
    # 'bucket' : 'bucket',
    # 'calculator' : 'calculator',
    # 'camera' : 'camera',
    # 'cellphone' : 'cellphone',
    # 'clock' : 'clock',
    # 'coffee_maker' : 'coffee_maker',
    # 'comb' : 'comb',
    # 'desk_lamp' : 'desk_lamp',
    # 'dishwasher' : 'dishwasher',
    # 'eraser' : 'eraser',
    # 'eyeglasses' : 'eyeglasses',
    # 'fan' : 'fan',
    # 'faucet' : 'faucet',
    # 'fish_tank' : 'fish_tank',
    # 'flashlight' : 'flashlight',
    # 'fork' : 'fork',
    # 'guitar' : 'guitar',
    # 'hair_dryer' : 'hair_dryer',
    # 'hammer' : 'hammer',
    # 'headphone' : 'headphone',
    # 'helmet' : 'helmet',
    # 'iron' : 'iron',
    # 'kettle' : 'kettle',
    # 'key' : 'key',
    # 'knife' : 'knife',
    # 'lighter' : 'lighter',
    # 'mailbox' : 'mailbox',
    # 'microphone' : 'microphone',
    # 'paintbrush' : 'paintbrush',
    # 'pan' : 'pan',
    # 'pen' : 'pen',
    # 'pencil' : 'pencil',
    # 'plate' : 'plate',
    # 'pot' : 'pot',
    # 'racket' : 'racket',
    # 'remote_control' : 'remote_control',
    # 'rifle' : 'rifle',
    # 'satellite_dish' : 'satellite_dish',
    # 'scissors' : 'scissors',
    # 'screwdriver' : 'screwdriver',
    # 'shoe' : 'shoe',
    # 'shovel' : 'shovel',
    # 'sign' : 'sign',
    # 'skate' : 'skate',
    # 'skateboard' : 'skateboard',
    # 'slipper' : 'slipper',
    # 'spoon' : 'spoon',
    # 'stapler' : 'stapler',
    # 'stove' : 'stove',
    # 'telephone' : 'telephone',
    # 'toaster' : 'toaster',
    # 'toothbrush' : 'toothbrush',
    # 'trophy' : 'trophy',
    # 'vending_machine' : 'vending_machine',
    # 'washing_machine' : 'washing_machine',
    # 'watch' : 'watch',
    # 'wheelchair' : 'wheelchair',
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
    'car' : 11,
    'chair' : 12,
    'computer' : 13,
    'cup' : 14,
    'desk' : 15,
    'table' : 16,
    'door' : 17,
    'fire_extinguisher' : 18,
    'jar' : 19,
    'keyboard' : 20,
    'laptop' : 21,
    'motorbike' : 22,
    'microwave' : 23,
    'mouse' : 24,
    'piano' : 25,
    'pillow' : 26,
    'printer' : 27,
    'refrigerator' : 28,
    'rode_pole' : 29,
    'sofa' : 30,
    'speaker' : 31,
    'suitcase' : 32,
    'teapot' : 33,
    'toilet' : 34,
    'train' : 35,
    'trash_bin' : 36,
    'bathtub' : 37,
    'tvmonitor' : 38,
    'wardrobe' : 39,
    ##################################################################
    'ashtray' : 40,
    'backpack' : 41,
    'basket' : 42,
    'blackboard' : 43,
    'bucket' : 44,
    'calculator' : 45,
    'camera' : 46,
    'cellphone' : 47,
    'clock' : 48,
    'coffee_maker' : 49,
    'comb' : 50,
    'desk_lamp' : 51,
    'dishwasher' : 52,
    'eraser' : 53,
    'eyeglasses' : 54,
    'fan' : 55,
    'faucet' : 56,
    'fish_tank' : 57,
    'flashlight' : 58,
    'fork' : 59,
    'guitar' : 60,
    'hair_dryer' : 61,
    'hammer' : 62,
    'headphone' : 63,
    'helmet' : 64,
    'iron' : 65,
    'kettle' : 66,
    'key' : 67,
    'knife' : 68,
    'lighter' : 69,
    'mailbox' : 70,
    'microphone' : 71,
    'paintbrush' : 72,
    'pan' : 73,
    'pen' : 74,
    'pencil' : 75,
    'plate' : 76,
    'pot' : 77,
    'racket' : 78,
    'remote_control' : 79,
    'rifle' : 80,
    'satellite_dish' : 81,
    'scissors' : 82,
    'screwdriver' : 83,
    'shoe' : 84,
    'shovel' : 85,
    'sign' : 86,
    'skate' : 87,
    'skateboard' : 88,
    'slipper' : 89,
    'spoon' : 90,
    'stapler' : 91,
    'stove' : 92,
    'telephone' : 93,
    'toaster' : 94,
    'toothbrush' : 95,
    'trophy' : 96,
    'vending_machine' : 97,
    'washing_machine' : 98,
    'watch' : 99,
    'wheelchair' : 100,
}

# from memory_profiler import profile


class nolboDatasetSingleObject(object):
    def __init__(self, nolboConfig,
                 mode='classification',
                 dataPath_ObjectNet3D=None, dataPath_Pascal3D=None, dataPath_pix3D=None,
                 loadOrg3DShape = False
                 ):
        self._nolboConfig = nolboConfig
        self._mode = mode
        self._dataPath_ObjectNet3D = dataPath_ObjectNet3D
        self._dataPath_Pascal3D = dataPath_Pascal3D
        self._dataPath_pix3D = dataPath_pix3D
        self._dataStart = 0
        self._dataLength = 0
        self._epoch = 0
        self._dataPathList = []
        self._classConvertion = None
        self._classDict = None
        self._instDict_ObjectNet3D = None
        self._instDict_pix3D = None
        self._loadOrg3DShape = loadOrg3DShape
        if self._mode == 'classification' or self._mode == 'nolbo' or self._mode == 'autoencoder':
            self._createDict()
            self._loadDataPath()
            self._dataPathShuffle()
        else:
            print 'mode must be `classification`, `autoencoder` or `nolbo`!'

    def _createDict(self):
        print 'create dict...'
        self._classConvertion = classConvertion
        self._classDict = classDict
        self._instDict_ObjectNet3D = dict()
        self._instDict_pix3D = dict()
        classList = os.listdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD'))
        print 'ObjectNet3D...'
        for orgClassName in classList:
            if os.path.isdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD', orgClassName)):
                if orgClassName in self._classConvertion:
                    className = self._classConvertion[orgClassName]
                    classIdx = classDict[className]
                    # print classIdx, className
                    if className not in self._instDict_ObjectNet3D:
                        self._instDict_ObjectNet3D[className] = dict()
                    CADModelList = os.listdir(os.path.join(self._dataPath_ObjectNet3D, 'CAD', orgClassName))
                    CADModelList.sort()
                    instIdx = 1
                    for CADModel in CADModelList:
                        if CADModel.endswith(".pcd"):
                            CADModelPath = os.path.join('CAD', orgClassName, os.path.splitext(CADModel)[0])
                            if CADModelPath not in self._instDict_ObjectNet3D[className]:
                                self._instDict_ObjectNet3D[className][CADModelPath] = instIdx
                                instIdx += 1
        print 'pix3D...'
        classList = os.listdir(os.path.join(self._dataPath_pix3D, 'model'))
        for orgClassName in classList:
            if os.path.isdir(os.path.join(self._dataPath_pix3D, 'model', orgClassName)):
                if orgClassName in self._classConvertion:
                    className = self._classConvertion[orgClassName]
                    classIdx = classDict[className]
                    # print classIdx, className
                    if className not in self._instDict_pix3D:
                        self._instDict_pix3D[className] = dict()
                    instIdx = 1
                    for path, dirs, files in os.walk(os.path.join(self._dataPath_pix3D, 'model', orgClassName)):
                        files.sort()
                        for fileName in files:
                            if fileName.endswith('.pcd'):
                                filePath = os.path.join('model', orgClassName, path.split('/')[-1],
                                                        os.path.splitext(fileName)[0])
                                if filePath not in self._instDict_pix3D[className]:
                                    self._instDict_pix3D[className][filePath] = instIdx
                                    instIdx += 1
        print 'dict ready!'

    def _loadDataPath(self):
        print 'load data path...'
        dataFromList = [self._dataPath_Pascal3D, self._dataPath_ObjectNet3D, self._dataPath_pix3D]
        for dataFrom in dataFromList:
            if dataFrom!=None:
                datasetTypeList = os.listdir(os.path.join(dataFrom, 'training_data'))
                datasetTypeList.sort()
                # print dataFrom
                for datasetType in datasetTypeList:
                    if os.path.isdir(os.path.join(dataFrom, 'training_data', datasetType)):
                        dataPointList = os.listdir(os.path.join(dataFrom, 'training_data', datasetType))
                        dataPointList.sort()
                        # dataPointIdx = 0
                        totalDataPointNum = len(dataPointList)
                        for dataPoint in dataPointList:
                            if os.path.isdir(os.path.join(dataFrom, 'training_data', datasetType, dataPoint)):
                                objFolderList = os.listdir(os.path.join(dataFrom, 'training_data', datasetType, dataPoint))
                                objFolderList.sort()
                                for objFolder in objFolderList:
                                    objInfoPath = os.path.join(dataFrom, 'training_data', datasetType, dataPoint, objFolder, 'objInfo.txt')
                                    # with open(objInfoPath) as objInfoFilePointer:
                                    #     className = objInfoFilePointer.readline().split(" ")[0]
                                    # if className in self._classConvertion:
                                    #     self._dataPathList.append(
                                    #         [dataFrom ,os.path.join(dataFrom, 'training_data', datasetType, dataPoint, objFolder)])
                                    self._dataPathList.append(
                                        [dataFrom, os.path.join(dataFrom, 'training_data', datasetType, dataPoint, objFolder)])
                #                     sys.stdout.write(datasetType)
                #                     sys.stdout.write(" ")
                #                     # sys.stdout.write(" {:05d}/{:05d} ".format(dataPointIdx, totalDataPointNum))
                #                     sys.stdout.write(os.path.join(dataPoint, objFolder))
                #                     sys.stdout.write("\r")
                #         sys.stdout.write("\n")
                # # sys.stdout.write("\n")
                del datasetTypeList
        print 'done!'
        self._dataLength = len(self._dataPathList)

    def _dataPathShuffle(self):
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
                objInfoFilePointer.close()
            objClassOrg, imgPath, CADModelPath, colMin, rowMin, colMax, rowMax, azimuth, elevation, in_plane_rot = objInfo.split(" ")
            if objClassOrg in self._classConvertion:

                #class index
                objClass = self._classConvertion[objClassOrg]
                objClassIdx = self._classDict[objClass] - 1
                objClassVector = np.zeros(self._nolboConfig['classDim'])
                objClassVector[objClassIdx] = 1
                classList.append(objClassVector.copy())


                #2D image
                if self._mode == 'nolbo' or self._mode == 'classification':
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
                    del inputImage
                    # image augmentation
                    if np.random.rand() < 0.5:
                        imageCanvas = datasetUtils.imageAugmentation(imageCanvas)
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
                    inputImages.append(imageCanvas.copy())
                    del imageCanvas
                    # del inputImage
                    # del imageCanvas

                # inst, rotated 3D shape, Euler angle
                if self._mode == 'nolbo' or self._mode == 'autoencoder':
                    pix3DOrPascal = 0
                    instIdx = -1
                    if dataFrom != self._dataPath_pix3D:
                        instIdx = self._instDict_ObjectNet3D[objClass][CADModelPath]
                    elif dataFrom == self._dataPath_pix3D:
                        pix3DOrPascal = 1
                        instIdx = self._instDict_pix3D[objClass][CADModelPath]
                    objInstVector = np.zeros(self._nolboConfig['instDim'])
                    objInstVector[0] = pix3DOrPascal
                    objInstVector[instIdx] = 1
                    instList.append(objInstVector.copy())
                    obj3DShape = np.load(obj3DShapePath)
                    outputImages.append(obj3DShape.copy())
                    # azRad = float(azimuth)/180.0*np.pi
                    # elRad = float(elevation)/180.0*np.pi
                    # ipRad = float(in_plane_rot)/180.0*np.pi
                    # objAEI = np.array(
                    #     [np.sin(azRad), np.sin(elRad), np.sin(ipRad),
                    #      np.cos(azRad), np.cos(elRad), np.cos(ipRad)])
                    objAEI = np.array([float(azimuth),float(elevation),float(in_plane_rot)])
                    AEIAngle.append(objAEI.copy())

                    # original 3D Shape (without rotation)
                    if self._loadOrg3DShape:
                        #add org shape
                        objOrg3DShape = np.load(objOrg3DShapePath)
                        if self._mode == 'autoencoder':
                            # classList.append(np.array(objClassVector).copy())
                            # instList.append(np.array(objInstVector).copy())
                            # AEIAngle.append(np.array([0.0, 0.0, 0.0]).copy())
                            outputImagesOrg.append(objOrg3DShape.copy())
                            del objOrg3DShape
                        elif self._mode == 'nolbo':
                            pass
                    del objInstVector
                    del obj3DShape
                    del objAEI
                del objClassVector

                addedDataNum += 1
            checkedDataNum += 1

        self._dataStart = self._dataStart+checkedDataNum

        batchDict = dict()
        batchDict['classList'] = (np.array(classList)).astype('float')
        del classList
        if self._mode == 'nolbo' or self._mode == 'classification':
            batchDict['inputImages'] = (np.array(inputImages)).astype('float')
            del inputImages

        if self._mode == 'nolbo' or self._mode == 'autoencoder':
            batchDict['instList'] = (np.array(instList)).astype('float')
            batchDict['outputImages'] = (np.array(outputImages)).astype('float')
            batchDict['AEIAngle'] = (np.array(AEIAngle)).astype('float')
            del instList
            del outputImages
            del AEIAngle
            if self._loadOrg3DShape:
                batchDict['outputImagesOrg'] = np.array(outputImagesOrg).astype('float')
                del outputImagesOrg

        return batchDict




















