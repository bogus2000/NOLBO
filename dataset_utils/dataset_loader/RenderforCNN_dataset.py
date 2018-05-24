import os, cv2
import numpy as np
import sys
import dataset_utils.datasetUtils as datasetUtils
from pathos.multiprocessing import ProcessingPool as Pool

def imageResize(imagePath, imageSize):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    image = cv2.resize(image, imageSize)
    image = datasetUtils.imgAug(image, flip=False)
    return image
class RenderforCNNDataset(object):
    def __init__(self, dataPath, classNum=12, instNum=7500, rotDim=360):
        self._dataPath = dataPath
        self._classNum = classNum
        self._instNum = instNum
        self._rotDim = rotDim
        self._epoch = 0
        self._dataStart = 0
        self._dataLength = 0
        self._dataPointPathList = None
        self._classIdxConverter = None
        self._instIdxConverter = None
        self._imageSize = (480, 640)
        self._loadDataPointPath()
        self._dataShuffle()
    def setImageSize(self, size=(480,640)):
        self._imageSize = (size[0], size[1])
    def _loadDataPointPath(self):
        print 'load data point path...'
        self._dataPointPathList = []
        self._classIdxConverter = dict()
        self._instIdxConverter = dict()
        trainPath = os.path.join(self._dataPath, 'train')
        classNameList = os.listdir(trainPath)
        classNameList.sort(key=datasetUtils.natural_keys)
        classIdx = 0
        for className in classNameList:
            classPath = os.path.join(trainPath, className)
            if os.path.isdir(classPath):
                if className not in self._classIdxConverter:
                    self._classIdxConverter[className] = classIdx
                    classIdx += 1
                if className not in self._instIdxConverter:
                    self._instIdxConverter[className] = dict()
                instNameList = os.listdir(classPath)
                instNameList.sort(key=datasetUtils.natural_keys)
                instIdx = 0
                for instName in instNameList:
                    instPath = os.path.join(classPath, instName)
                    if os.path.isdir(instPath):
                        if instName not in self._instIdxConverter[className]:
                            self._instIdxConverter[className][instName] = instIdx
                            instIdx += 1
                        rotInstList = os.listdir(instPath)
                        rotInstList.sort(key=datasetUtils.natural_keys)
                        for rotInstName in rotInstList:
                            rotInstPath = os.path.join(instPath, rotInstName)
                            self._dataPointPathList.append(rotInstPath)
                        sys.stdout.write('c:{:02d}/{:02d} i:{:04d}/{:04d}\r'.format(classIdx, len(classNameList), instIdx, len(instNameList)))
        self._dataLength = len(self._dataPointPathList)
        print 'done!'

    def _dataShuffle(self):
        self._dataStart = 0
        np.random.shuffle(self._dataPointPathList)

    ''' https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class '''
    def getNextBatchPar(self, batchSize=32):
        self._pool = Pool()
        if self._dataStart + batchSize >= self._dataLength:
            self._epoch += 1
            self._dataStart = 0
            self._dataShuffle()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart+batchSize
        dataPathTemp = self._dataPointPathList[dataStart:dataEnd]

        classList = np.zeros((batchSize, self._classNum))
        instList = np.zeros((batchSize, self._instNum))
        azimuthList = np.zeros((batchSize, self._rotDim))
        elevationList = np.zeros((batchSize, self._rotDim))
        in_plane_rotList = np.zeros((batchSize, self._rotDim))
        for i in range(len(dataPathTemp)):
            imagePath = dataPathTemp[i]
            imagePathSplit = imagePath.split("/")
            fileName = imagePathSplit[-1]
            instName = imagePathSplit[-2]
            className = imagePathSplit[-3]

            classIdx = self._classIdxConverter[className]
            instIdx = self._instIdxConverter[className][instName]
            classList[i,classIdx] = 1
            instList[i,instIdx] = 1

            modelId, instId, azimuth, elevation, tilt, dAndEXE = fileName.split("_")
            azimuth, elevation, tilt = int(azimuth[1:]), int(elevation[1:]), int(tilt[1:])
            if azimuth<0:
                azimuth += 360
            azimuth = azimuth%360
            if elevation<0:
                elevation += 360
            elevation = elevation%360
            if tilt<0:
                tilt += 360
            tilt = tilt%360
            azimuthList[i,azimuth] = 1
            elevationList[i,elevation] = 1
            in_plane_rotList[i,tilt] = 1

        imageSize = [self._imageSize] * batchSize
        inputImages = self._pool.map(imageResize, dataPathTemp, imageSize)

        batchData = {
            'inputImages' : np.array(inputImages).astype('float'),
            'classList' : np.array(classList).astype('float'),
            'instList' : np.array(instList).astype('float'),
            'azimuth' : np.array(azimuthList).astype('float'),
            'elevation' : np.array(elevationList).astype('float'),
            'in_plane_rot' : np.array(in_plane_rotList).astype('float'),
        }
        return batchData




















