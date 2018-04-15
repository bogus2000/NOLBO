import scipy.io
import numpy as np
import os, re
import cv2
import dataset_utils.datasetUtils as datasetUtils

def build_pretraining_data():
    mat = scipy.io.loadmat('/media/yonsei/4TB_HDD/downloads/SUNRGBDMeta3DBB_v2.mat')
    print mat.keys()
    mat = mat['SUNRGBDMeta'][0]
    print mat.shape
    
    imgList = []
    imgPathList = []
    imgClassList = []
    classNameList = os.listdir('/media/yonsei/4TB_HDD/dataset/suncg_data/modelNet/')
    classNameList.sort(key=datasetUtils.natural_keys)
    for img in mat:
        depthPath = '/media/yonsei/4TB_HDD/downloads'+img['depthpath'][0][16:]
        print depthPath    
        bbox = img['groundtruth3DBB']
        if len(bbox)!= 0:
            depthImg = cv2.imread(depthPath,0)
            depthImg = depthImg.reshape((depthImg.shape[0], depthImg.shape[1], 1))
            depthImg = cv2.normalize(depthImg, depthImg, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
            classIndex = [0]*len(classNameList)
            classNameListInImg = bbox['classname'][0]
            for classNameInImg in classNameListInImg:
                classNameInImg = str(classNameInImg[0])
                if classNameInImg == 'cabinet':
                    classNameInImg = 'wardrobe'
                for classNameIdx in range(len(classNameList)):
                    if classNameInImg == classNameList[classNameIdx]:
                        classIndex[classNameIdx] = 1
                        break
            imgList.append(depthImg)
            imgPathList.append(depthPath)
            imgClassList.append(classIndex)
    imgList = np.array(imgList)
    imgPathList = np.array(imgPathList)
    imgClassList = np.array(imgClassList)
    np.save('./pretraining_data/imgList.npy', imgList)
    np.save('./pretraining_data/imgClassList.npy', imgClassList)
    np.save('./pretraining_data/imgPathList.npy', imgPathList)

