import os, cv2
import numpy as np
import sys
from xml.etree.cElementTree import parse
import dataset_utils.datasetUtils as datasetUtils
from pathos.multiprocessing import ProcessingPool as Pool
# from multiprocessing import Pool
#
# pool = Pool(processes=8)
def imageResize(imagePath, imageSize, bbox):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if bbox!=None:
        imageBbox = image[bbox[2]:bbox[3], bbox[0]:bbox[1],:]
        if len(imageBbox)==0 or len(imageBbox[0])==0:
            imageResult = image
        else:
            imageResult = imageBbox
    else:
        imageResult = image
    imageResult = datasetUtils.imgAug(imageResult)
    imageResult = cv2.resize(imageResult, imageSize)
    return imageResult

class imagenetDataset(object):
    def __init__(self, dataPath, classNum=1000):
        self._dataPath = dataPath
        self._classNum = classNum
        self._epoch = 0
        self._dataStart = 0
        self._dataLength = 0
        self._dataPointPathList = None
        self._classIdxConverter = None
        self._imageSize = (480, 640)
        self._loadDataPointPath()
        self._dataShuffle()

    def setImageSize(self, size=(480, 640)):
        self._imageSize = (size[0],size[1])

    def _loadDataPointPath(self):
        print 'load data point path...'
        self._dataPointPathList = []
        self._classIdxConverter = dict()
        trainPath = os.path.join(self._dataPath, 'train')
        classNameList = os.listdir(trainPath)
        classNameList.sort(key=datasetUtils.natural_keys)
        classIdx = 0
        for className in classNameList:
            classPath = os.path.join(trainPath, className)
            if os.path.isdir(classPath):
                if className in self._classIdxConverter:
                    pass
                else:
                    self._classIdxConverter[className] = classIdx
                    classIdx += 1
                instNameList = os.listdir(classPath)
                instNameList.sort(key=datasetUtils.natural_keys)
                for instName in instNameList:
                    instPath = os.path.join(classPath, instName)
                    self._dataPointPathList.append(instPath)
                sys.stdout.write('{:04d}/{:04d}\r'.format(classIdx, 1000))

            if classIdx == self._classNum:
                break
        self._dataLength = len(self._dataPointPathList)
        print 'done!'

    def _dataShuffle(self):
        # 'data list shuffle...'
        self._dataStart = 0
        np.random.shuffle(self._dataPointPathList)
        # print 'done!'

    ''' https://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class '''
    def getNextBatchPar(self, batchSize=32):
        self._pool = Pool()
        if self._dataStart + batchSize >= self._dataLength:
            self._epoch += 1
            self._dataStart = 0
            self._dataShuffle()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart + batchSize
        dataPathTemp = self._dataPointPathList[dataStart:dataEnd]

        bboxList = []
        classIndexList = np.zeros((batchSize, self._classNum), dtype=np.float32)
        for i in range(len(dataPathTemp)):
            imagePath = dataPathTemp[i]
            className = imagePath.split("/")[-2]
            classIdx = self._classIdxConverter[className]
            # classIndexVector = np.zeros(1000)
            # classIndexVector[classIdx] = 1
            classIndexList[i, classIdx] = 1.0
            # classIndexList.append(classIndexVector)
            xmlPath = imagePath.split("train")
            xmlPath = os.path.join(xmlPath[0], 'bbox', xmlPath[1].split(".")[0][1:] + '.xml')
            if os.path.isfile(xmlPath):
                note = parse(xmlPath).getroot()
                xMin = np.max((0, int(note.find('object').find('bndbox').findtext('xmin'))))
                xMax = np.max((0, int(note.find('object').find('bndbox').findtext('xmax'))))
                yMin = np.max((0, int(note.find('object').find('bndbox').findtext('ymin'))))
                yMax = np.max((0, int(note.find('object').find('bndbox').findtext('ymax'))))
                bboxList.append([xMin,xMax,yMin,yMax])
            else:
                bboxList.append(None)

        imageSize = [self._imageSize] * batchSize
        inputImages = self._pool.map(imageResize, dataPathTemp, imageSize, bboxList)

        batchData = {
            'inputImages': np.array(inputImages).astype('float'),
            'classIndexList': np.array(classIndexList).astype('float')
        }
        return batchData

    def getNextBatch(self, batchSize=32):
        if self._dataStart + batchSize >= self._dataLength:
            self._epoch += 1
            self._dataStart = 0
            self._dataShuffle()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart + batchSize
        dataPathTemp = self._dataPointPathList[dataStart:dataEnd]
        inputImages = []
        classIndexList = np.zeros((batchSize, self._classNum), dtype=np.float32)
        for i in range(len(dataPathTemp)):
            imagePath = dataPathTemp[i]
            className = imagePath.split("/")[-2]
            classIdx = self._classIdxConverter[className]
            # classIndexVector = np.zeros(1000)
            # classIndexVector[classIdx] = 1
            classIndexList[i,classIdx] = 1
            # classIndexList.append(classIndexVector)

            image = cv2.imread(imagePath, cv2.IMREAD_COLOR).astype('float')
            # image = 2.0*image/255.0 - 1.0
            # print self._imageSize
            image = cv2.resize(image, self._imageSize)
            # inputImages.append(image)
            inputImages.append(image.copy())

        batchData = {
            'inputImages': np.array(inputImages),
            'classIndexList': np.array(classIndexList)
        }
        return batchData