import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import sys
import src.net_core.darknet as darknet
import src.net_core.autoencoder as ae

# nolbo_multiObjectConfig = {
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

def sampling(mu, logVar):
    epsilon = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    samples = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(logVar)), epsilon))
    return samples


def binary_loss(xPred, xTarget):
    gamma = 0.50
    b_range = 0.0
    voxelDimTotal = np.prod(xPred.get_shape().as_list()[1:])
    yTarget = -b_range + (2.0 * b_range + 1.0) * tf.reshape(xTarget, (-1, voxelDimTotal))
    yPred = tf.clip_by_value(tf.reshape(xPred, (-1, voxelDimTotal)), clip_value_min=1e-7, clip_value_max=1.0 - 1e-7)
    bce_loss = - tf.reduce_sum(gamma * yTarget * tf.log(yPred) + (1.0 - gamma) * (1.0 - yTarget) * tf.log(1.0 - yPred),
                               axis=-1)
    return bce_loss


def lb_loss(mean, logVar, mean_target, logVar_target, dim):
    vectorDimTotal = np.prod(mean.get_shape().as_list()[1:])
    m = tf.reshape(mean, (-1, vectorDimTotal))
    lV = tf.reshape(logVar, (-1, vectorDimTotal))
    m_t = tf.reshape(mean_target, (-1, vectorDimTotal))
    lV_t = tf.reshape(logVar_target, (-1, vectorDimTotal))
    loss = tf.reduce_sum(0.5 * (lV_t - lV) + tf.div((tf.exp(lV) + tf.square(m - m_t)), (2.0 * tf.exp(lV_t))) - 0.5,
                         axis=-1)
    return loss

class nolbo_singleObject(object):
    def __init__(self, config=None, isTraining=True):
        #==========set parameters===========
        self._config = config
        self._isTraining = isTraining
        #==========declare networks=========
        self._encoder = None
        self._decoder = None
        #==========declare inputs===========
        self._inputImages = tf.placeholder(tf.float32, shape=np.concatenate([[None], self._config['inputImgDim']]))
        #==========declare predicted outputs===========
        self._meanPred = None
        self._variancePred = None
        self._zs = None
        self._outputImagesPred = None
        #==========declare parameters=========
        self._latentDim = None
        #==========declare Ground Truth (GT) inputs for training==========
        self._outputImagesGT = None
        self._classListGT = None
        self._instListGT = None
        self._EulerAngleGT = None
        # ==========declare parameters for training==============
        self._batchSize = None
        self._lr = None
        self._variables = None
        self._update_ops = None
        self._saver = None
        self._optimizer = None
        self._totalLoss = None

        if self._isTraining == True:
            self._initTraining()
        self._buildNetwork()
        if self._isTraining == True:
            self._createLoss()
            self._setOptimizer()

        # init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # launch the session
        self._sess.run(init)

    def _initTraining(self):
        print "init training..."
        self._outputImagesGT = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._config['decoderStructure']['outputImgDim']])))
        self._classListGT = None
        self._instListGT = None
        self._EulerAngleGT = None
        print 'done!'

    def _getEncOutChannelDimAndLatentDim(self):
        self._latentDim = 0
        if self._config['class'] == True:
            self._latentDim += self._config['zClassDim']
        if self._config['inst'] == True:
            self._latentDim += self._config['zInstDim']
        if self._config['rot'] == True:
            self._latentDim += self._config['zRotDim']

    def _buildNetwork(self):
        print "build network..."
        self._getEncOutChannelDimAndLatentDim()
        self._inputImages = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._config['inputImgDim']])))
        self._encoder = darknet.encoder_singleVectorOutput(outputVectorDim= 2 * self._latentDim) #multiply 2 as we want both mean and variance
        self._decoder = ae.decoder(architecture=self._config['decoderStructure'], decoderName='nolboDec')
        encOutput = self._encoder(self._inputImages)
        self._meanPred, self._variancePred = tf.split(encOutput, num_or_size_splits=2, axis=0)
        self._zs = sampling(mu=self._meanPred, logVar=self._variancePred)
        self._outputImagesPred = self._decoder(self._zs)
        print "done!"

    def _createLoss(self):
        print "create loss..."
        self._binaryLoss = tf.reduce_mean(binary_loss(self._outputImagesPred, self._outputImagesGT))
        self._totalLoss = self._binaryLoss
    def _setOptimizer(self, learningRate=0.0001):
        print "set optimizer..."
        self._lr = learningRate
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        with tf.control_dependencies(self._encoder.allUpdate_ops + self._decoder.update_ops):
            self._optimizer = optimizer.minimize(
                self._totalLoss,
                var_list= self._encoder.allVariables + self._decoder.variables
            )
    def fit(self, batchDict):
        feed_dict = {
            self._inputImages : batchDict['inputImages'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT : batchDict['classList'],
            self._instListGT : batchDict['instList'],
            self._EulerAngleGT : batchDict['EulerAngle']
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._binaryLoss
        opt, loss = self._sess.run([optimizer, lossList], feed_dict=feed_dict)
        return loss

class nolbo_multiObject(object):
    def __init__(self, config=None, isTraining=True):
        #==========set parameters===========
        self._config = config
        self._isTraining = isTraining
        #==========declare networks==========
        self._encoder = None
        self._decoder = None
        #==========declare inputs============
        self._inputImages = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._config['inputImgDim']])))        
        #==========declare predicted outputs====================
        self._encOutput = None
        self._bboxHWXYPred = None
        self._objectnessPred = None
        self._meanPred = None
        self._variancePred = None
        self._zs = None
        self._outputImagesPred = None
        #==========declare parameters ==========================
        self._latentDim = None
        self._objMask = None
        #==========declare Ground Truth (GT) inputs for training============        
        self._bboxHWXYGT = None
        self._objectnessGT = None
        self._outputImagesGT = None
        self._classListGT = None
        self._instListGT = None
        self._EulerAngleGT = None
        #==========declare parameters for training==============
        self._batchSize = None
        self._lr = None
        self._variables = None
        self._update_ops = None
        self._saver = None
        self._optimizer = None
        self._encOutChannelDim = None
        self._gridSize = None
        self._IOU = None
        self._totalLoss = None

        if self._isTraining==True:
            self._initTraining()
        self._buildNetwork()
        if self._isTraining==True:
            self._createLoss()
            self._setOptimizer()
        
        #init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        #initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        #launch the session
        self._sess.run(init)
        
    def _buildNetwork(self):
        print "build network..."
        self._getEncOutChannelDimAndLatentDim()
        self._inputImages = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._config['inputImgDim']])))
        self._encoder = darknet.encoder_gridOutput(lastLayserChannelNum=self._encOutChannelDim, nameScope='nolboEnc')
        self._decoder = ae.decoder(architecture=self._config['decoderStructure'], decoderName='nolboDec')
        self._encOutput = self._encoder(self._inputImages)
        #bboxGT = (None,13,13,predictorNumPerGrid, bboxDim)        
        self._bboxHWXYPred, self._objectnessPred, self._meanPred, self._variancePred = [],[],[],[]
        partStart = 0
        for predictorIndex in range(self._config['predictorNumPerGrid']):
            self._bboxHWXYPred.append(self._encOutput[:,:,:,partStart:partStart+self._config['bboxDim']-1])
            partStart += self._config['bboxDim']-1
            self._objectnessPred.append(self._encOutput[:,:,:,partStart:partStart+1])
            partStart += 1
            self._meanPred.append(self._encOutput[:,:,:,partStart:partStart+self._latentDim])
            partStart += self._latentDim
            self._variancePred.append(self._encOutput[:,:,:,partStart:partStart+self._latentDim])
        self._bboxHWXYPred = tf.transpose(tf.stack(self._bboxHWXYPred), [1,2,3,0,4])
        self._objectnessPred = tf.transpose(tf.stack(self._objectnessPred), [1,2,3,0,4])
        self._meanPred = tf.transpose(tf.stack(self._meanPred), [1,2,3,0,4])
        self._variancePred = tf.transpose(tf.stack(self._variancePred), [1,2,3,0,4])        
        print 'self._bboxHWXYs.shape',self._bboxHWXYPred.shape
        print 'self._objectnesses.shape',self._objectnessPred.shape
        print 'self._means.shape',self._meanPred.shape
        print 'self._variances.shape',self._variancePred.shape
        self._zs = self._getLatent()
        self._outputImagesPred = self._decoder(self._zs)
        print "done!"
        
    def _createLoss(self):
        print "create loss..."
        #declare variables for training
        self._variables = self._encoder.allVariables + self._decoder.variables
        self._update_ops= self._encoder.allUpdate_ops + self._decoder.update_ops
        self._saver = tf.train.Saver(var_list=self._variables)
        
        diffBboxX = self._objMask * (self._bboxHWXYPredTile[:,:,:,:,2] - self._bboxHWXYGTTile[:,:,:,:,2])
        diffBboxY = self._objMask * (self._bboxHWXYPredTile[:,:,:,:,3] - self._bboxHWXYGTTile[:,:,:,:,3])
        diffBboxH = self._objMask * (tf.square(self._bboxHWXYPredTile[:,:,:,:,0]) - tf.square(self._bboxHWXYGTTile[:,:,:,:,0]))
        diffBboxW = self._objMask * (tf.square(self._bboxHWXYPredTile[:,:,:,:,1]) - tf.square(self._bboxHWXYGTTile[:,:,:,:,1]))
        objTileShape = np.concatenate([[-1], self._objectnessPredTile.get_shape().as_list()[1:-1]])
        diffObjPrObjxIOU = self._objMask * (self._IOU * tf.reshape(self._objectnessPredTile, objTileShape) - tf.reshape(self._objectnessGTTile, objTileShape))
        diffNoObj = (1-self._objMask) * (tf.reshape(self._objectnessPredTile, objTileShape) - tf.reshape(self._objectnessGTTile, objTileShape))
                
        self._bboxLoss = tf.reduce_mean(tf.reduce_sum(tf.square(diffBboxX+diffBboxY+diffBboxH+diffBboxW), axis=[1,2,3]))
        self._objLoss = tf.reduce_mean(tf.reduce_sum(tf.square(diffObjPrObjxIOU), axis=[1,2,3]))
        self._noObjLoss = tf.reduce_mean(tf.reduce_sum(tf.square(diffNoObj), axis=[1,2,3]))
        self._binaryLoss = tf.reduce_mean(self._binary_loss(self._outputImagesPred, self._outputImagesGT))
        self._totalLoss = self._bboxLoss + self._objLoss + self._noObjLoss + self._binaryLoss
        print 'done!'
        
    def _setOptimizer(self, learningRate=0.0001):
        print "set optimizer..."
        self._lr = learningRate        
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        with tf.control_dependencies(self._update_ops):
            self._optimizer = optimizer.minimize(self._totalLoss, var_list=self._variables)
        print 'done!'
        
    def fit(self, batchDict):
        feed_dict = {
            self._inputImages : batchDict['inputImages'],
            self._bboxHWXYGT : batchDict['bboxHWXY'],
            self._objectnessGT : batchDict['objectness'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT : batchDict['classList'],
            self._instListGT : batchDict['instList'],
            self._EulerAngleGT : batchDict['EulerAngle']
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._bboxLoss, self._objLoss, self._noObjLoss, self._binaryLoss
        opt, loss = self._sess.run([optimizer, lossList], feed_dict=feed_dict)
        return loss
    
    def _getEncOutChannelDimAndLatentDim(self):
        self._latentDim = 0
        self._encOutChannelDim = self._config['bboxDim']
        if self._config['class'] == True:
            self._encOutChannelDim += 2*self._config['zClassDim'] #should multiply 2 as we need both mean and variance
            self._latentDim += self._config['zClassDim']
        if self._config['inst'] == True:
            self._encOutChannelDim += 2*self._config['zInstDim'] #should multiply 2 as we need both mean and variance
            self._latentDim += self._config['zInstDim']
        if self._config['rot'] == True:
            self._encOutChannelDim += 2*self._config['zRotDim'] #should multiply 2 as we need both mean and variance
            self._latentDim += self._config['zRotDim']
        self._encOutChannelDim = self._encOutChannelDim * self._config['predictorNumPerGrid']
    

    def _getLatent(self):
        if self._isTraining==True:
            self._getTiles()
            self._getObjMask()
#             we use dynamic_partition instead of boolean_mask according to the following sites:
#             https://stackoverflow.com/questions/44380727/get-userwarning-while-i-use-tf-boolean-mask?noredirect=1&lq=1
#             https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation#35896823
#             mean = tf.boolean_mask(self._meanPredTile, tf.cast(self._objMask, tf.bool))
#             variance = tf.boolean_mask(self._variancePredTile, tf.cast(self._objMask, tf.bool))
            mean = tf.dynamic_partition(self._meanPredTile, tf.cast(self._objMask, tf.int32), 2)[1]
            variance = tf.dynamic_partition(self._variancePredTile, tf.cast(self._objMask, tf.int32), 2)[1]
            latentVariables = self._sampling(mean, variance)
            return latentVariables
        


    def _initTraining(self):
        print "init training..."
        self._gridSize = [self._config['inputImgDim'][0]/2**self._config['maxPoolNum'], self._config['inputImgDim'][1]/2**self._config['maxPoolNum']]
        self._bboxHWXYGT = tf.placeholder(tf.float32, shape=([None, self._gridSize[0], self._gridSize[1], self._config['predictorNumPerGrid'], self._config['bboxDim']-1]))
        self._objectnessGT = tf.placeholder(tf.float32, shape=(np.concatenate([[None,self._gridSize[0], self._config['predictorNumPerGrid'], self._gridSize[1]], [1]])))
        self._outputImagesGT = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._config['decoderStructure']['outputImgDim']])))
        self._classListGT = None
        self._instListGT = None
        self._EulerAngleGT = None
        print 'done!'
    
    def _getTiles(self):
        #tiles for prediction
        self._bboxHWXYPredTile = tf.tile(self._bboxHWXYPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._objectnessPredTile = tf.tile(self._objectnessPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._meanPredTile = tf.tile(self._meanPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._variancePredTile = tf.tile(self._variancePred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG        
        self._bboxHWXYGTTile = []
        self._objectnessGTTile = []
        for objInBboxGTIdx in range(self._config['predictorNumPerGrid']):
            for predictorIndex in range(self._config['predictorNumPerGrid']):
                self._bboxHWXYGTTile+=[self._bboxHWXYGT[:,:,:,objInBboxGTIdx:objInBboxGTIdx+1,:]]
                self._objectnessGTTile+=[self._objectnessGT[:,:,:,objInBboxGTIdx:objInBboxGTIdx+1,:]]
        self._bboxHWXYGTTile = tf.concat(self._bboxHWXYGTTile, axis=-2)
        self._objectnessGTTile = tf.concat(self._objectnessGTTile, axis=-2)
        
    def _getObjMask(self):
        print 'get ObjMask...'
        self._getIOU() #IOU = (None, 13,13, prdNPG*objNPG)
        self._objMask = []
        for objIdx in range(self._config['predictorNumPerGrid']):
            #IOUPerObj = (None, 13,13, prdNPG)
            IOUPerObj = self._IOU[:,:,:,objIdx*self._config['predictorNumPerGrid']:(objIdx+1)*self._config['predictorNumPerGrid']]
            #maxPerObj = (None, 13,13)
            maxPerObj = tf.reduce_max(IOUPerObj, -1, keep_dims=True)
            maskPerObj = tf.cast((IOUPerObj>=maxPerObj),tf.float32) * tf.cast((IOUPerObj!=0.0),tf.float32) #True for maximum and non-zero, False for non-maximum and zero
            self._objMask+=[maskPerObj] # maskPerObj = (None,13,13,prdNPG)
        self._objMask = tf.concat(self._objMask, axis=-1) #objMask=(None,13,13,prdNPG*objNPG)
        print 'done!'
    
    def _getIOU(self):
        print 'get IOU...'
        # bboxGT.shape = (None, 13,13,predictorNumPerGrid, bboxDim)
        # predictBbox.shape = (None, 13,13,predictorNumPerGrid, bboxDim)
        # bbox = [H,W,x,y]
        # get bboxGTTile = (None, 13,13,predictorNumPerGrid*objNumPerGrid, bboxDim)
        #box = [colMin, rowMin, colMax, rowMax]
        boxGT = tf.stack([self._bboxHWXYGTTile[:,:,:,:,2] - self._bboxHWXYGTTile[:,:,:,:,0]/2.0,
                          self._bboxHWXYGTTile[:,:,:,:,3] - self._bboxHWXYGTTile[:,:,:,:,1]/2.0,
                          self._bboxHWXYGTTile[:,:,:,:,2] + self._bboxHWXYGTTile[:,:,:,:,0]/2.0,
                          self._bboxHWXYGTTile[:,:,:,:,3] + self._bboxHWXYGTTile[:,:,:,:,1]/2.0]) # (rcMinMax, None, 13,13, prdNPG*objNPG)
        boxGT = tf.transpose(boxGT, [1,2,3,4,0]) # (None, 13,13, prdNPG*objNPG, rcMinMax)
        boxPr = tf.stack([self._bboxHWXYPredTile[:,:,:,:,2] - self._bboxHWXYPredTile[:,:,:,:,0]/2.0,
                          self._bboxHWXYPredTile[:,:,:,:,3] - self._bboxHWXYPredTile[:,:,:,:,1]/2.0,
                          self._bboxHWXYPredTile[:,:,:,:,2] + self._bboxHWXYPredTile[:,:,:,:,0]/2.0,
                          self._bboxHWXYPredTile[:,:,:,:,3] + self._bboxHWXYPredTile[:,:,:,:,1]/2.0])
        boxPr = tf.transpose(boxPr, [1,2,3,4,0])
        #get leftup point and rightdown point of intersection
        lu = tf.maximum(boxGT[:,:,:,:,:2], boxPr[:,:,:,:,:2])
        rd = tf.minimum(boxGT[:,:,:,:,2:], boxPr[:,:,:,:,2:])
        #get intersection Area
        intersection = tf.maximum(0.0, rd - lu)
        intersectionArea = intersection[:,:,:,:,0] * intersection[:,:,:,:,1]
        #get boxGTArea and boxPrArea
        boxGTArea = (boxGT[:,:,:,:,2] - boxGT[:,:,:,:,0]) * (boxGT[:,:,:,:,3] - boxGT[:,:,:,:,1])
        boxPrArea = (boxPr[:,:,:,:,2] - boxPr[:,:,:,:,0]) * (boxPr[:,:,:,:,3] - boxPr[:,:,:,:,1])
        unionArea = tf.maximum(boxGTArea + boxPrArea - intersectionArea, 1e-7)        
        self._IOU = tf.clip_by_value(intersectionArea/unionArea, 0.0, 1.0) #IOU = (None, 13,13, prdNPG*objNPG)
        print 'done!'
            
    def _sampling(self, mu, logVar):
        epsilon = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
        samples = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(logVar)), epsilon))
        return samples
    
    def _binary_loss(self, xPred, xTarget):
        gamma = 0.50
        b_range = 0.0
        voxelDimTotal = np.prod(xPred.get_shape().as_list()[1:])
        yTarget = -b_range + (2.0*b_range+1.0)*tf.reshape(xTarget, (-1,voxelDimTotal))
        yPred = tf.clip_by_value(tf.reshape(xPred, (-1,voxelDimTotal)), clip_value_min=1e-7, clip_value_max=1.0-1e-7)
        bce_loss = - tf.reduce_sum(gamma * yTarget * tf.log(yPred) + (1.0-gamma) * (1.0 - yTarget) * tf.log(1.0-yPred), axis=-1)
        return bce_loss
    def _lb_loss(self, mean,logVar, mean_target,logVar_target, dim):
        vectorDimTotal = np.prod(mean.get_shape().as_list()[1:])
        m = tf.reshape(mean, (-1, vectorDimTotal))
        lV = tf.reshape(logVar, (-1, vectorDimTotal))
        m_t = tf.reshape(mean_target, (-1, vectorDimTotal))
        lV_t = tf.reshape(logVar_target, (-1, vectorDimTotal))
        loss = tf.reduce_sum(0.5*(lV_t - lV)+tf.div((tf.exp(lV) + tf.square(m - m_t)),(2.0 * tf.exp(lV_t)))- 0.5, axis=-1)
        return loss

class darknet_classifier(object):
    def __init__(self, dataPath='./', imgSize = (416,416), batchSize = 64, learningRate = 0.0001, lastLayerActivation = tf.nn.sigmoid):
        self._imgList = None
        self._imgClassList = None
        self._dataPath = dataPath
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
        self._lastAct = lastLayerActivation
        self._classNum = None
        self.variables = None
        self.update_ops = None
        self._inputImg = None
        self._outputClass = None
        self._outputClassGT = None
        self._optimizer = None
        self._loss = None
        self._loadDataset()
        self._buildNetwork()
        self._createLossAndOptimizer()
        #init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        #launch the session
        self._sess.run(init)
    def _loadDataset(self):
        print "load Dataset..."
        self._imgList = []
        imgListTemp = np.load(os.path.join(self._dataPath,'imgList.npy'))
        self._imgClassList = np.load(os.path.join(self._dataPath+'imgClassList.npy'))
        self._classNum = self._imgClassList.shape[1]
        for i in range(len(imgListTemp)):
            img = cv2.resize(imgListTemp[i], self._imgSize)
            img = img.reshape((self._imgSize[0], self._imgSize[1],1))
            self._imgList.append(img)
        self._imgList = np.array(self._imgList)
        print "done!"
    def _buildNetwork(self):
        print "build network..."
        self._inputImg = tf.placeholder(tf.float32, shape=(None, self._imgSize[0], self._imgSize[1], 1))
        self._outputClassGT = tf.placeholder(tf.float32, shape=(None, self._classNum))
        self._classifier = darknet.encoder_singleVectorOutput(outputVectorDim=self._classNum)
        self._outputClass = self._classifier(self._inputImg)
        if self._lastAct != None:
            self._outputClass = self._lastAct(self._outputClass)
        print "done!"
    def _createLossAndOptimizer(self):
        print "create loss and optimizer..."
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        def binaryLoss(xPred, xTarget, epsilon=1e-7):
            yTarget = xTarget
            yPred = tf.clip_by_value(xPred, clip_value_min=epsilon, clip_value_max=1.0-epsilon)
            bce_loss = - tf.reduce_sum(yTarget*tf.log(yPred) + (1.0-yTarget)*tf.log(1.0-yPred), axis=-1)
            return bce_loss
        self._loss = tf.reduce_mean(binaryLoss(xPred=self._outputClass, xTarget=self._outputClassGT))
        with tf.control_dependencies(self._classifier.allUpdate_ops):
            self._optimizer = self._optimizer.minimize(
                self._loss, var_list = self._classifier.allVariables
            )
        print "done!"
    def _saveNetwork(self, savePath='./'):
        dCorePath = os.path.join(savePath,'classifierCore.ckpt')
        pretrainPath = os.path.join(savePath,'classifierLastLayer.ckpt')
        self._classifier.coreSaver.save(self._sess, dCorePath)
        self._classifier.lastLayerSaver.save(self._sess, pretrainPath)
    def _restoreNetwork(self, restorePath='./'):
        dCorePath = os.path.join(restorePath,'classifierCore.ckpt')
        pretrainPath = os.path.join(restorePath,'classifierLastLayer.ckpt')
        self._classifier.coreSaver.restore(self._sess, dCorePath)
        self._classifier.lastLayerSaver.restore(self._sess, pretrainPath)
    def _fit(self, batchImg, batchClassIndex):
        feed_dict = {
            self._inputImg : batchImg,
            self._outputClassGT : batchClassIndex
        }
        accAll = (tf.reduce_sum((1-self._outputClass)*(1-self._outputClassGT))+tf.reduce_sum(self._outputClass*self._outputClassGT))\
        /(tf.reduce_sum(self._outputClassGT)+tf.reduce_sum(1-self._outputClassGT))
        accPositive = tf.reduce_sum(self._outputClass*self._outputClassGT)/tf.reduce_sum(self._outputClassGT)
        _, lossResult = self._sess.run([self._optimizer, self._loss], feed_dict=feed_dict)
        accAllResult, accPositiveResult = self._sess.run([accAll, accPositive], feed_dict=feed_dict)
        return lossResult, accAllResult, accPositiveResult

    def train(self, epoch = 1000, weightSavePath='./'):
        currEpoch = 0
        iteration = 0.0
        loss = 0
        accAll, accPositive = 0, 0
        runTime = 0
        for i in range(int(epoch/self._batchSize)):
            for i in range(int(len(self._imgList)/self._batchSize)):
                startTime = time.time()
                start = i * self._batchSize
                end = np.min((start+self._batchSize, len(self._imgList)))
                lossTemp, accAllTemp, accPositiveTemp = self._fit(self._imgList[start:end], self._imgClassList[start:end])
                endTime = time.time()
                runTimeTemp = endTime - startTime
                currIter = iteration%1000
                if iteration!=0 and currIter == 0:
                    sys.stdout.write('\nsaveWeights...\n')
                    self._saveNetwork(weightSavePath)
                    loss = 0
                    accAll, accPositive = 0, 0
                    runTime = 0
                accAll = float(accAll*currIter + accAllTemp)/float(currIter+1.0)
                accPositive = float(accPositive*currIter + accPositiveTemp)/float(currIter+1.0)
                loss = float(loss*currIter + lossTemp)/float(currIter+1.0)
                runTime = float(runTime*currIter + runTimeTemp)/(currIter+1.0)
                sys.stdout.write('Epoch:{:04d} iter:{:06d} runtime:{:.3f} '.format(int(currEpoch+1), int(iteration+1), runTime))
                sys.stdout.write('curr/total:{:05d}/{:05d} '.format(start, len(self._imgList)))
                sys.stdout.write('loss:{:.3f} accAll:{:.3f} accPos:{:.3f}\r'.format(loss, accAll, accPositive))
                iteration += 1.0
            currEpoch +=1