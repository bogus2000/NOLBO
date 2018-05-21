import numpy as np
import tensorflow as tf
import cv2
import time
import os, pickle
import src.module.nolbo as nolboModule
import dataset_utils.dataset_loader.nolbo_dataset as nolboDataset

def myFunc():
	nolboConfig = {
	    'inputImgDim':[416,416,3],
	    'maxPoolNum':5,
	    'predictorNumPerGrid':2,
	    'bboxDim':5,
	    'class':True, 'zClassDim':64, 'classDim':101,
	    'inst':True, 'zInstDim':64, 'instDim':1000,
	    'rot':True, 'zRotDim':3, 'rotDim':3,
	    'trainable':True,
	    'decoderStructure':{
		'outputImgDim':[64,64,64,1],
		'trainable':True,    
		'filterNumList':[512,256,128,64,1],
		'kernelSizeList':[4,4,4,4,4],
		'stridesList':[1,2,2,2,2],
		'activation':tf.nn.leaky_relu,
		'lastLayerActivation':tf.nn.sigmoid
	    }
	}

	dataPath_Object3D = '/media/yonsei/4TB_HDD/dataset/ObjectNet3D/'
	dataPath_Pascal3D = '/media/yonsei/4TB_HDD/dataset/PASCAL3D+_release1.1/'
	dataPath_pix3D = '/media/yonsei/4TB_HDD/dataset/pix3d/'

	dataset = nolboDataset.nolboDatasetSingleObject(
	    nolboConfig=nolboConfig, 
	    dataPath_ObjectNet3D=dataPath_Object3D,
	    dataPath_Pascal3D=dataPath_Pascal3D,
	    dataPath_pix3D=dataPath_pix3D
	)
	for i in range(3):
		batchData = dataset.getNextBatch(batchSize=32)

if __name__=="__main__":
	myFunc()
