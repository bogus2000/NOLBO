import numpy as np
import os, re
import sys

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\\d+)', text) ]

acceptedClassList = {
    'aeroplane',
    'bed',
    'bench',
    'bicycle',
    'boat', 'bookshelf', 'bottle', 'bus',
    ############################################
    'cabinet', 'can', 'cap', 'car', 'chair',
    'computer', 'cup',
    # 'desk',
    'diningtable',
    'door', 'fire_extinguisher', 'jar',
    #############################################
    'keyboard', 'laptop', 'microwave', 'motorbike', 'mouse',
    'piano', 'pillow', 'printer', 'refrigerator', 'road_pole',
    ##############################################
    'sofa', 'speaker', 'suitcase', 'teapot', 'toilet', 'train',
    'trash_bin', 'tub', 'tvmonitor',
    # 'wardrobe',
}

class ObjectNet3D_voxelRotationDataset(object):
    def __init__(self, trainingDataPath, partitionNum=5, loadVoxOrg=True):
        self._epoch = 0
        self._trainingDataPath = trainingDataPath
        self._partitionNum = partitionNum
        self._loadVoxOrg = loadVoxOrg
        self._curruntPartition = 0.0
        self._partitionLength = 0
        self._batchStart = 0
        self._classListData = None
        self._instListData = None
        self._vox3DData = None
        self._EulerAngleData = None
        self._vox3DOrg = None
        self._dataListPerClass = dict()
        self._loadPartition()

    def _loadPartition(self):
        print ''
        print 'load a partition of dataset...'
        if int(self._curruntPartition+1.0) > self._partitionNum:
            self._curruntPartition = 0.0
            self._epoch += 1
        self._classListData = []
        self._instListData = []
        self._vox3DData = []
        self._EulerAngleData = []
        self._vox3DOrg = dict()
        classNameList = os.listdir(self._trainingDataPath)
        classNameList.sort(key=natural_keys)
        # classNum = 1
        for className in classNameList:
            if className in acceptedClassList and \
                    os.path.isdir(os.path.join(self._trainingDataPath, className)):

                # print className
                # sys.stdout.write(className)
                # sys.stdout.write('{:02d}/{:02d}\r'.format(classNum, 40))
                # classNum += 1
                # print 'load class List...'
                classListDataTemp = np.load(os.path.join(self._trainingDataPath, className, 'classIdx.npy')).astype('bool')
                # print 'load inst List...'
                instListDataTemp = np.load(os.path.join(self._trainingDataPath, className, 'instIdx.npy')).astype('bool')
                # print 'load voxels...'
                vox3DDataTemp = np.load(os.path.join(self._trainingDataPath, className, 'vox3D.npy')).astype('bool')
                # print 'load Euler angles...'
                EulerAngleDataTemp = np.load(os.path.join(self._trainingDataPath, className, 'EulerAngle.npy'))
                # print 'load voxel original...'
                if self._curruntPartition == 0:
                    listTemp = np.arange(len(classListDataTemp))
                    np.random.shuffle(listTemp)
                    self._dataListPerClass[className] = listTemp

                vox3DOrgTemp = np.load(os.path.join(self._trainingDataPath, className, 'vox3DOrg.npy')).astype('bool')
                dataLength = len(classListDataTemp)
                dataStart = int(dataLength * (self._curruntPartition / float(self._partitionNum)))
                dataEnd = int(dataLength * ((self._curruntPartition + 1.0) / float(self._partitionNum)))
                # self._classListData = np.vstack((self._classListData, classListDataTemp[dataStart:dataEnd]))
                # self._instListData = np.vstack((self._instListData, instListDataTemp[dataStart:dataEnd]))
                # self._vox3DData = np.vstack((self._vox3DData, vox3DDataTemp[dataStart:dataEnd]))
                # self._EulerAngleData = np.vstack((self._EulerAngleData, EulerAngleDataTemp[dataStart:dataEnd]/180.0*np.pi))
                self._classListData += [classListDataTemp[self._dataListPerClass[className][dataStart:dataEnd]].copy()]
                self._instListData += [instListDataTemp[self._dataListPerClass[className][dataStart:dataEnd]].copy()]
                self._vox3DData += [vox3DDataTemp[self._dataListPerClass[className][dataStart:dataEnd]].copy()]
                self._EulerAngleData += [EulerAngleDataTemp[self._dataListPerClass[className][dataStart:dataEnd]].copy()/180.0*np.pi] # angle to radian
                # print 'data length :', len(self._classListData[-1])
                classIdx = np.argmax(classListDataTemp[0])
                self._vox3DOrg[classIdx] = vox3DOrgTemp.copy()
                del classListDataTemp
                del instListDataTemp
                del vox3DDataTemp
                del EulerAngleDataTemp
                del vox3DOrgTemp
        self._curruntPartition += 1.0
        self._classListData = np.concatenate(self._classListData, axis=0)
        self._instListData = np.concatenate(self._instListData, axis=0)
        self._vox3DData = np.concatenate(self._vox3DData, axis=0)
        self._EulerAngleData = np.concatenate(self._EulerAngleData)
        self._partitionLength = len(self._classListData)
        # print self._partitionLength
        self._shuffleList = np.arange(self._partitionLength)
        np.random.shuffle(self._shuffleList)
        # self._classListData = self._classListData[s]
        # self._instListData = self._instListData[s]
        # self._vox3DData = self._vox3DData[s]
        # self._EulerAngleData = self._EulerAngleData[s]
        # print ''
        print 'done!'

    def getNextBatch(self, batchSize=32):
        if self._batchStart + batchSize > self._partitionLength:
            self._batchStart = 0
            self._loadPartition()
        dataStart = self._batchStart
        dataEnd = self._batchStart + batchSize
        self._batchStart += batchSize
        dataList = self._shuffleList[dataStart:dataEnd]
        # batch_dict = {
        #     'inputImages': np.array([self._vox3DData[i] for i in dataList]).astype('float'),
        #     'classList': np.array([self._classListData[i] for i in dataList]).astype('float'),
        #     'instList': np.array([self._instListData[i] for i in dataList]).astype('float'),
        #     'EulerAngle': np.array([self._EulerAngleData[i] for i in dataList]).astype('float'),
        #     'outputImages': np.array([self._vox3DData[i] for i in dataList]).astype('float'),
        # }
        batch_dict = {
            'inputImages': (self._vox3DData[dataList]).astype('float'),
            'classList': (self._classListData[dataList]).astype('float'),
            'instList': (self._instListData[dataList]).astype('float'),
            'EulerAngle': (self._EulerAngleData[dataList]).astype('float'),
            'outputImages' : (self._vox3DData[dataList]).astype('float'),
        }
        if self._loadVoxOrg:
            vox3DOrgData = []
            for i in range(batchSize):
                classIdx = np.argmax(batch_dict['classList'][i])
                instIdx = np.argmax(batch_dict['instList'][i])
                vox3DOrgData.append(self._vox3DOrg[classIdx][instIdx])
            vox3DOrgData = np.array(vox3DOrgData)
            batch_dict['outputImagesOrg'] = vox3DOrgData.astype('float')
        return batch_dict