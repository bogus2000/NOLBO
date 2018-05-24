import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle
import sys
import src.net_core.darknet as darknet
import src.net_core.autoencoder as ae
import src.net_core.priornet as priornet
import dataset_utils.datasetUtils as datasetUtils

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
def categorical_crossentropy(gt, pred):
    loss = -tf.reduce_mean(tf.reduce_sum(gt*tf.log(pred+1e-8), reduction_indices=1))
    return loss

def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)

def sampling(mu, logVar):
    epsilon = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    samples = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(logVar)), epsilon))
    return samples


def regulizer_loss(z_mean, z_logVar, dist_in_z_space, class_input = None):
    dim_z = tf.shape(z_mean)[-1]
    batch_size = tf.shape(z_mean)[0]
    z_m_repeat = tf.reshape(z_mean, tf.stack([batch_size, 1, dim_z]))
    z_m_repeat_tr = tf.reshape(z_mean, tf.stack([1, batch_size, dim_z]))
    z_logVar_repeat = tf.reshape(z_logVar, tf.stack([batch_size, 1, dim_z]))
    z_m_repeat = tf.tile(z_m_repeat, tf.stack([1, batch_size, 1]))
    z_m_repeat_tr = tf.tile(z_m_repeat_tr, tf.stack([batch_size, 1, 1]))
    z_logVar_repeat = tf.tile(z_logVar_repeat, tf.stack([1, batch_size, 1]))

    diff = tf.abs(z_m_repeat - z_m_repeat_tr) / tf.exp(0.5 * z_logVar_repeat)
    diff = tf.reduce_sum(diff, axis=-1)

    diff_in_z = diff - dist_in_z_space * tf.cast(dim_z, tf.float32) * tf.ones_like(diff)
    diff_in_z = tf.where(
        tf.greater(diff_in_z, tf.zeros_like(diff_in_z)), tf.zeros_like(diff_in_z), tf.square(diff_in_z))

    dot_cos = tf.reduce_sum(z_m_repeat*z_m_repeat_tr,axis=-1)/(
        tf.norm(z_m_repeat,axis=-1)*tf.norm(z_m_repeat_tr, axis=-1))
    dot_cos_abs = tf.abs(dot_cos)
    loss_reg = diff_in_z + diff_in_z * dot_cos_abs

    if class_input != None:
        c_i_repeat = tf.reshape(
            class_input, tf.stack([batch_size, 1, class_input.get_shape().as_list()[-1]]))
        c_i_repeat_tr = tf.reshape(
            class_input, tf.stack([1, batch_size, class_input.get_shape().as_list()[-1]]))
        c_i_repeat = tf.tile(c_i_repeat, tf.stack([1, batch_size, 1]))
        c_i_repeat_tr = tf.tile(c_i_repeat_tr, tf.stack([batch_size, 1, 1]))
        c_i_diff_abs = tf.abs(c_i_repeat - c_i_repeat_tr)
        c_i_diff_sum = tf.reduce_sum(c_i_diff_abs, axis=-1)
        # if categories are the same, get 1
        # else, get zero
        c_i_diff = tf.where(tf.greater(c_i_diff_sum, 0.0), tf.zeros_like(c_i_diff_sum), tf.ones_like(c_i_diff_sum))
        loss_reg = loss_reg * c_i_diff

    # loss_reg = tf.reduce_mean(loss_reg)
    return loss_reg

def binary_loss(xPred, xTarget, epsilon = 1e-7, gamma=0.5, b_range=False):
    b_range = float(b_range)
    voxelDimTotal = np.prod(xPred.get_shape().as_list()[1:])
    yTarget = -b_range + (2.0 * b_range + 1.0) * tf.reshape(xTarget, (-1, voxelDimTotal))
    yPred = tf.clip_by_value(tf.reshape(xPred, (-1, voxelDimTotal)), clip_value_min=epsilon, clip_value_max=1.0 - epsilon)
    bce_loss = - tf.reduce_sum(gamma * yTarget * tf.log(yPred) + (1.0 - gamma) * (1.0 - yTarget) * tf.log(1.0 - yPred),
                               axis=-1)
    return bce_loss


def nlb_loss(mean, logVar, mean_target, logVar_target):
    vectorDimTotal = np.prod(mean.get_shape().as_list()[1:])
    m = tf.reshape(mean, (-1, vectorDimTotal))
    lV = tf.reshape(logVar, (-1, vectorDimTotal))
    m_t = tf.reshape(mean_target, (-1, vectorDimTotal))
    lV_t = tf.reshape(logVar_target, (-1, vectorDimTotal))
    loss = tf.reduce_sum(0.5 * (lV_t - lV) + tf.div((tf.exp(lV) + tf.square(m - m_t)), (2.0 * tf.exp(lV_t))) - 0.5,
                         axis=-1)
    return loss

def create_evaluation(xTarget, xPred, prob=0.5):
    yTarget = tf.cast(tf.reshape(xTarget, (-1, np.prod(xTarget.get_shape().as_list()[1:]))), tf.float32)
    yPred = tf.cast(tf.greater_equal(tf.reshape(xPred, (-1, np.prod(xPred.get_shape().as_list()[1:]))), prob), tf.float32)
    TP = tf.reduce_mean(tf.reduce_sum(yTarget * yPred, axis=-1))
    FP = tf.reduce_mean(tf.reduce_sum((1.0 - yTarget) * yPred, axis=-1))
    FN = tf.reduce_mean(tf.reduce_sum(yTarget * (1.0 - yPred), axis=-1))
    p = TP / (TP + FP + 1e-10)
    r = TP / (TP + FN + 1e-10)
    return p,r

class nolbo_singleObject(object):
    def __init__(self, config=None, nameScope='nolbo', isTraining=True):
        #==========set parameters===========
        self._config = config
        self._nameScope=nameScope
        self._isTraining = isTraining
        #==========declare networks=========
        self._encoder = None
        self._decoder = None
        self._classPriornet = None
        self._instPriornet = None
        #==========declare inputs===========
        self._inputImages = tf.placeholder(tf.float32, shape=np.concatenate([[None], self._config['inputImgDim']]))
        #==========declare predicted outputs===========
        self._classMeanPred, self._instMeanPred = None, None
        self._classLogVarPred, self._instLogVarPred = None, None
        self._zs = None
        self._outputImagesPred = None
        self._classMeanPrior, self._instMeanPrior = None,None
        self._classLogVarPrior, self._instLogVarPrior = None,None
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
        self._outputImagesGT = tf.placeholder(
            tf.float32, shape=(np.concatenate([[None], self._config['decoderStructure']['outputImgDim']])))
        self._classListGT = tf.placeholder(
            tf.float32, shape=([None, self._config['classDim']]))
        self._instListGT = tf.placeholder(
            tf.float32, shape=([None, self._config['instDim']]))
        self._EulerAngleGT = tf.placeholder(
            tf.float32, shape=([None, self._config['rotDim']]))
        # self._lr = tf.Variable(self._config['learningRate'])
        self._lr = tf.placeholder(tf.float32, shape=[])
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
        self._encoder = darknet.encoder_singleVectorOutput(
            outputVectorDim= 2 * self._latentDim,
            coreActivation=tf.nn.elu,
            trainable=self._config['trainable'],
            nameScope=self._nameScope+'-enc') #multiply 2 as we want both mean and variance
        self._decoder = ae.decoder(
            architecture=self._config['decoderStructure'],
            decoderName=self._nameScope+'-dec')
        encOutput = self._encoder(self._inputImages)
        self._meanPred, self._logVarPred = tf.split(encOutput, num_or_size_splits=2, axis=-1)
        # self._classMeanPred = self._meanPred[...,0:self._config['zClassDim']]
        # self._instMeanPred = self._meanPred[...,self._config['zClassDim']:self._config['zClassDim']+self._config['zInstDim']]
        # self._rotMeanPred = self._meanPred[...,self._config['zClassDim']+self._config['zInstDim']:]
        # self._classLogVarPred = self._logVarPred[..., 0:self._config['zClassDim']]
        # self._instLogVarPred = self._logVarPred[...,self._config['zClassDim']:self._config['zClassDim'] + self._config['zInstDim']]
        # self._rotLogVarPred = self._logVarPred[..., self._config['zClassDim'] + self._config['zInstDim']:]
        self._zs = sampling(mu=self._meanPred, logVar=self._logVarPred)
        self._outputImagesPred = self._decoder(self._zs)

        if self._isTraining:
            self._classPriornet = priornet.priornet(
                inputDim=self._config['classDim'], outputDim=self._config['zClassDim'],
                hiddenLayerNum=2, scopeName=self._nameScope+'-classPrior', training=self._isTraining,
                coreActivation=tf.nn.elu,
                constLogVar= 0.0)
            self._instPriornet = priornet.priornet(
                inputDim=self._config['classDim'] + self._config['instDim'], outputDim=self._config['zInstDim'],
                hiddenLayerNum=2, scopeName=self._nameScope+'-instPrior', training=self._isTraining,
                coreActivation=tf.nn.elu,
                constLogVar= 0.0)
            self._classMeanPrior, self._classLogVarPrior = \
                self._classPriornet(self._classListGT)
            self._instMeanPrior, self._instLogVarPrior = \
                self._instPriornet(tf.concat([self._classListGT, self._instListGT], axis=-1))
            self._p, self._r = create_evaluation(xTarget=self._outputImagesGT, xPred=self._outputImagesPred)
        print "done!"

    def _createLoss(self):
        print "create loss..."
        self._binaryLoss = tf.reduce_mean(
            binary_loss(xPred=self._outputImagesPred, xTarget=self._outputImagesGT, gamma=0.97, b_range=False))
        self._regulizerLoss\
            = tf.reduce_mean(
                regulizer_loss(z_mean=self._classMeanPrior, z_logVar=self._classLogVarPrior, dist_in_z_space=10.0)
                +
                regulizer_loss(
                    z_mean=self._instMeanPrior, z_logVar=self._instLogVarPrior,
                    dist_in_z_space=3.0, class_input=self._classListGT))
        self._meanPrior = tf.concat([self._classMeanPrior, self._instMeanPrior, self._EulerAngleGT], axis=-1)
        self._meanPrior_p360 = tf.concat([self._classMeanPrior, self._instMeanPrior, self._EulerAngleGT+360.0], axis=-1)
        self._meanPrior_n360 = tf.concat([self._classMeanPrior, self._instMeanPrior, self._EulerAngleGT-360.0], axis=-1)
        self._logVarPrior = tf.concat(
            [self._classLogVarPrior, self._instLogVarPrior, 2.0*tf.log(0.1)*tf.ones_like(self._EulerAngleGT)], axis=-1)

        nlb0 = nlb_loss(mean=self._meanPred, logVar=self._logVarPred,
                     mean_target=self._meanPrior, logVar_target=self._logVarPrior)
        nlbp360 = nlb_loss(mean=self._meanPred, logVar=self._logVarPred,
                        mean_target=self._meanPrior_p360, logVar_target=self._logVarPrior)
        nlbn360 = nlb_loss(mean=self._meanPred, logVar=self._logVarPred,
                        mean_target=self._meanPrior_n360, logVar_target=self._logVarPrior)
        self._nlbLoss = tf.reduce_mean(tf.minimum(tf.minimum(nlb0, nlbp360), nlbn360))
        self._totalLoss = 10.0*self._binaryLoss + 0.1*self._nlbLoss + 0.01*self._regulizerLoss

    def _setOptimizer(self):
        print "set optimizer..."
        with tf.control_dependencies(
            self._encoder.allUpdate_ops + self._decoder.update_ops
            + self._classPriornet.update_ops + self._instPriornet.update_ops
        ):
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(
                self._totalLoss,
                var_list
                =
                self._encoder.allVariables + self._decoder.variables
                +
                self._classPriornet.variables + self._instPriornet.variables
            )
    def fit(self, batchDict):
        feed_dict = {
            self._lr: batchDict['learningRate'],
            self._inputImages : batchDict['inputImages'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT : batchDict['classList'],
            self._instListGT : batchDict['instList'],
            self._EulerAngleGT : batchDict['EulerAngle']
        }

        opt, loss, pr = self._sess.run(
            [self._optimizer,
             self._totalLoss, self._binaryLoss, self._regulizerLoss, self._nlbLoss,
             self._p, self._r], feed_dict=feed_dict)
        return loss, pr

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._nameScope + '_encoderCore.ckpt')
        self._encoder.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.save(self._sess, eLastPath)
    def saveDecoder(self, savePath='./'):
        dPath = os.path.join(savePath, self._nameScope + '_decoder.ckpt')
        self._decoder.saver.save(self._sess, dPath)
    def savePriornet(self, savePath='./'):
        pCPath = os.path.join(savePath, self._nameScope + '_classPrior.ckpt')
        pIPath = os.path.join(savePath, self._nameScope + '_instPrior.ckpt')
        self._classPriornet.saver.save(self._sess, pCPath)
        self._instPriornet.saver.save(self._sess, pIPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
        self.saveDecoder(savePath)
        self.savePriornet(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._encoder.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreDecoder(self, restorePath='./'):
        dPath = os.path.join(restorePath, self._nameScope + '_decoder.ckpt')
        self._decoder.saver.restore(self._sess, dPath)
    def restorePriornet(self, restorePath='./'):
        pCPath = os.path.join(restorePath, self._nameScope + '_classPrior.ckpt')
        pIPath = os.path.join(restorePath, self._nameScope + '_instPrior.ckpt')
        self._classPriornet.saver.restore(self._sess, pCPath)
        self._instPriornet.saver.restore(self._sess, pIPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)
        self.restoreDecoder(restorePath)
        self.restorePriornet(restorePath)

class nolbo_multiObject(object):
    def __init__(self, config=None, nameScope='nolbo', isTraining=True):
        #==========set parameters===========
        self._config = config
        self._nameScope = nameScope
        self._isTraining = isTraining
        #==========declare networks==========
        self._encoder = None
        self._decoder = None
        self._classPriornet, self._instPriornet = None,None
        #==========declare inputs============
        self._inputImages = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._config['inputImgDim']])))        
        #==========declare predicted outputs====================
        self._encOutput = None
        self._bboxHWXYPred = None
        self._objectnessPred = None
        self._meanPred = None
        self._logVarPred = None
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
        self._encoder = darknet.encoder_gridOutput(
            lastLayserChannelNum=self._encOutChannelDim, nameScope=self._nameScope+'-enc')
        self._decoder = ae.decoder(architecture=self._config['decoderStructure'], decoderName=self._nameScope+'-dec')
        self._encOutput = self._encoder(self._inputImages)
        #bboxGT = (None,13,13,predictorNumPerGrid, bboxDim)        
        self._bboxHWXYPred, self._objectnessPred, self._meanPred, self._logVarPred = [],[],[],[]
        partStart = 0
        for predictorIndex in range(self._config['predictorNumPerGrid']):
            self._bboxHWXYPred.append(self._encOutput[:,:,:,partStart:partStart+self._config['bboxDim']-1])
            partStart += self._config['bboxDim']-1
            self._objectnessPred.append(self._encOutput[:,:,:,partStart:partStart+1])
            partStart += 1
            self._meanPred.append(self._encOutput[:,:,:,partStart:partStart+self._latentDim])
            partStart += self._latentDim
            self._logVarPred.append(self._encOutput[:,:,:,partStart:partStart+self._latentDim])
            partStart += self._latentDim
        self._bboxHWXYPred = tf.nn.sigmoid(tf.transpose(tf.stack(self._bboxHWXYPred), [1,2,3,0,4]))
        self._objectnessPred = tf.nn.sigmoid(tf.transpose(tf.stack(self._objectnessPred), [1,2,3,0,4]))
        self._meanPred = tf.transpose(tf.stack(self._meanPred), [1,2,3,0,4])
        self._logVarPred = tf.transpose(tf.stack(self._logVarPred), [1,2,3,0,4])
        print 'self._bboxHWXYs.shape',self._bboxHWXYPred.shape
        print 'self._objectnesses.shape',self._objectnessPred.shape
        print 'self._means.shape',self._meanPred.shape
        print 'self._logVars.shape',self._logVarPred.shape
        self._selectObjAndFracMeanLogVar()
        self._zs = sampling(mu=self._meanPred, logVar=self._logVarPred)
        print 'self._zs.shape',self._zs.shape
        self._outputImagesPred = self._decoder(self._zs)

        if self._isTraining:
            self._classPriornet = priornet.priornet(
                inputDim=self._config['classDim'], outputDim=self._config['zClassDim'],
                hiddenLayerNum=2, scopeName=self._nameScope+'-classPrior',
                constLogVar= 0.0)
            self._instPriornet = priornet.priornet(
                inputDim=self._config['classDim'] + self._config['instDim'], outputDim=self._config['zInstDim'],
                hiddenLayerNum=2, scopeName=self._nameScope+'-instPrior',
                constLogVar= 0.0)
            self._classMeanPrior, self._classLogVarPrior = \
                self._classPriornet(self._classListGT)
            self._instMeanPrior, self._instLogVarPrior = \
                self._instPriornet(tf.concat([self._classListGT, self._instListGT], axis=-1))
        print "done!"
        
    def _createLoss(self):
        print "create loss..."
        #declare variables for training
        self._variables = self._encoder.allVariables + self._decoder.variables
        self._update_ops= self._encoder.allUpdate_ops + self._decoder.update_ops
        
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
        self._binaryLoss = tf.reduce_mean(binary_loss(self._outputImagesPred, self._outputImagesGT))

        self._regulizerLoss \
            = tf.reduce_mean(
            regulizer_loss(z_mean=self._classMeanPrior, z_logVar=self._classLogVarPrior, dist_in_z_space=20.0)
            + regulizer_loss(
                z_mean=self._instMeanPrior, z_logVar=self._instLogVarPrior, dist_in_z_space=5.0, 
                class_input=self._classListGT))
        self._meanPrior = tf.concat([self._classMeanPrior, self._instMeanPrior, self._EulerAngleGT], axis=-1)
        self._logVarPrior = tf.concat(
            [self._classLogVarPrior, self._instLogVarPrior, 2.0 * tf.log(0.1) * tf.ones_like(self._EulerAngleGT)])
        self._nlbLoss = tf.reduce_mean(
            nlb_loss(mean=self._meanPred, logVar=self._logVarPred,
                     mean_target=self._meanPrior, logVar_target=self._logVarPrior))
        self._totalLoss = \
            self._bboxLoss + 5.0*self._objLoss + 0.5*self._noObjLoss + \
            self._binaryLoss + self._regulizerLoss + self._nlbLoss
        print 'done!'
        
    def _setOptimizer(self):
        print "set optimizer..."
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        with tf.control_dependencies(self._update_ops):
            self._optimizer = optimizer.minimize(self._totalLoss, var_list=self._variables)
        print 'done!'
        
    def fit(self, batchDict):
        feed_dict = {
            self._lr : batchDict['learningRate'],
            self._inputImages : batchDict['inputImages'],
            self._bboxHWXYGT : batchDict['bboxHWXY'],
            self._objectnessGT : batchDict['objectness'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT : batchDict['classList'],
            self._instListGT : batchDict['instList'],
            self._EulerAngleGT : batchDict['EulerAngle']
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._bboxLoss, self._objLoss, self._noObjLoss, \
                   self._binaryLoss, self._regulizerLoss, self._nlbLoss
        opt, loss = self._sess.run([optimizer, lossList], feed_dict=feed_dict)
        return loss

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._nameScope + '_encoderCore.ckpt')
        self._encoder.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.save(self._sess, eLastPath)
    def saveDecoder(self, savePath='./'):
        dPath = os.path.join(savePath, self._nameScope + '_decoder.ckpt')
        self._decoder.saver.save(self._sess, dPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
        self.saveDecoder(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._encoder.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._encoder.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreDecoder(self, restorePath='./'):
        dPath = os.path.join(restorePath, self._nameScope + '_decoder.ckpt')
        self._decoder.saver.restore(self._sess, dPath)
    def restoreNetwork(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)
        self.restoreDecoder(restorePath)
    
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

    def _selectObjAndFracMeanLogVar(self):
        if self._isTraining==True:
            self._getTiles()
            self._getObjMask()
#             we use dynamic_partition instead of boolean_mask according to the following sites:
#             https://stackoverflow.com/questions/44380727/get-userwarning-while-i-use-tf-boolean-mask?noredirect=1&lq=1
#             https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation#35896823
#             mean = tf.boolean_mask(self._meanPredTile, tf.cast(self._objMask, tf.bool))
#             variance = tf.boolean_mask(self._variancePredTile, tf.cast(self._objMask, tf.bool))
            self._meanPred = tf.dynamic_partition(self._meanPredTile, tf.cast(self._objMask, tf.int32), 2)[1]
            self._logVarPred = tf.dynamic_partition(self._logVarPredTile, tf.cast(self._objMask, tf.int32), 2)[1]
            self._classMeanPred = self._meanPred[..., 0:self._config['zClassDim']]
            self._instMeanPred = self._meanPred[...,
                                 self._config['zClassDim']:self._config['zClassDim'] + self._config['zInstDim']]
            self._rotMeanPred = self._meanPred[..., self._config['zClassDim'] + self._config['zInstDim']:]
            self._classLogVarPred = self._logVarPred[..., 0:self._config['zClassDim']]
            self._instLogVarPred = self._logVarPred[...,
                                   self._config['zClassDim']:self._config['zClassDim'] + self._config['zInstDim']]
            self._rotLogVarPred = self._logVarPred[..., self._config['zClassDim'] + self._config['zInstDim']:]

    def _initTraining(self):
        print "init training..."
        self._gridSize = [self._config['inputImgDim'][0]/2**self._config['maxPoolNum'], self._config['inputImgDim'][1]/2**self._config['maxPoolNum']]
        self._bboxHWXYGT = tf.placeholder(tf.float32, shape=([None, self._gridSize[0], self._gridSize[1], self._config['predictorNumPerGrid'], self._config['bboxDim']-1]))
        self._objectnessGT = tf.placeholder(tf.float32, shape=([None,self._gridSize[0], self._gridSize[1], self._config['predictorNumPerGrid'], 1]))
        self._outputImagesGT = tf.placeholder(tf.float32, shape=(np.concatenate([[None], self._config['decoderStructure']['outputImgDim']])))
        self._classListGT = tf.placeholder(tf.float32, shape=([None,self._config['classDim']]))
        self._instListGT = tf.placeholder(tf.float32, shape=([None, self._config['instDim']]))
        self._EulerAngleGT = tf.placeholder(tf.float32, shape=([None, self._config['rotDim']]))
        # self._lr = tf.Variable(self._config['learningRate'])
        self._lr = tf.placeholder(tf.float32, shape=[])
        offset = np.transpose(
            np.reshape(
                np.array(
                    [np.arange(self._gridSize[0])] * self._gridSize[1] * self._config['predictorNumPerGrid']),
                (self._config['predictorNumPerGrid'], self._gridSize[0], self._gridSize[1])
            )
            ,(1, 2, 0)
        )
        #offset for columns
        self._offset = tf.reshape(
            tf.constant(offset, dtype=tf.float32),
            [1, self._gridSize[0], self._gridSize[1], self._config['predictorNumPerGrid']]
        )
        print 'done!'
    
    def _getTiles(self):
        #tiles for prediction
        self._bboxHWXYPredTile = tf.tile(self._bboxHWXYPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._objectnessPredTile = tf.tile(self._objectnessPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._meanPredTile = tf.tile(self._meanPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
        self._logVarPredTile = tf.tile(self._logVarPred, [1,1,1,self._config['predictorNumPerGrid'],1]) #objNPG
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
            IOUPerObj = self._IOU[...,objIdx*self._config['predictorNumPerGrid']:(objIdx+1)*self._config['predictorNumPerGrid']]
            objectnessGTPerObj = self._objectnessGTTile[...,objIdx*self._config['predictorNumPerGrid']:(objIdx+1)*self._config['predictorNumPerGrid'],0]
            #maxPerObj = (None, 13,13)
            _, indices = tf.nn.top_k(IOUPerObj, self._config['predictorNumPerGrid'])
            print "indices.shape",indices.shape
            indices = tf.reshape(indices, (-1,self._config['predictorNumPerGrid']))
            indices = tf.map_fn(tf.invert_permutation, indices)
            print "indices.shape", indices.shape
            indices = tf.reshape(indices,(-1, self._gridSize[0],self._gridSize[1],self._config['predictorNumPerGrid']))
            print "indices.shape", indices.shape
            # maxPerObj = tf.reduce_max(IOUPerObj, -1, keep_dims=True)
            # # maskPerObj = tf.cast((IOUPerObj>=maxPerObj and IOUPerObj!=0.0),tf.float32)
            # maskPerObj = tf.where(IOUPerObj>=maxPerObj, tf.ones_like(IOUPerObj), tf.zeros_like(IOUPerObj))
            maskPerObj = tf.cast(tf.where(indices<=0, tf.ones_like(indices), tf.zeros_like(indices)), tf.float32)
            maskPerObj = maskPerObj * objectnessGTPerObj
            self._objMask+=[maskPerObj] # maskPerObj = (None,13,13,prdNPG)
            print "maskPerObj.shape",maskPerObj.shape
        self._objMask = tf.concat(self._objMask, axis=-1) #objMask=(None,13,13,prdNPG*objNPG)
        print 'done!'
    
    def _getIOU(self):
        print 'get IOU...'
        # bboxGT.shape = (None, 13,13,predictorNumPerGrid, bboxDim)
        # predictBbox.shape = (None, 13,13,predictorNumPerGrid, bboxDim)
        # bboxHWXY = [0:H, 1:W, 2:x, 3:y]
        # get bboxGTTile = (None, 13,13,predictorNumPerGrid*objNumPerGrid, bboxDim)
        #box = [colMin, rowMin, colMax, rowMax]
        #   = [x-W/2, y-H/2, x+W/2, y+H/2]
        offset_column = tf.tile(
            self._offset, [self._bboxHWXYGT.get_shape().as_list()[0], 1,1,self._config['predictorNumPerGrid']])
        offset_row = tf.transpose(offset_column, (0, 2, 1, 3))
        boxGT = tf.stack([(self._bboxHWXYGTTile[:,:,:,:,2]+offset_column)/float(self._gridSize[1])
                          - self._bboxHWXYGTTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYGTTile[:,:,:,:,3]+offset_row)/float(self._gridSize[0])
                          - self._bboxHWXYGTTile[:,:,:,:,0]/2.0,
                          (self._bboxHWXYGTTile[:, :, :, :, 2] + offset_column) / float(self._gridSize[1])
                          + self._bboxHWXYGTTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYGTTile[:, :, :, :, 3] + offset_row) / float(self._gridSize[0])
                          + self._bboxHWXYGTTile[:,:,:,:,0]/2.0],
                         axis=-1) # (None, 13,13, prdNPG*objNPG, rcMinMax)
        boxPr = tf.stack([(self._bboxHWXYPredTile[:,:,:,:,2]+offset_column)/float(self._gridSize[1])
                          - self._bboxHWXYPredTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYPredTile[:,:,:,:,3]+offset_row)/float(self._gridSize[0])
                          - self._bboxHWXYPredTile[:,:,:,:,0]/2.0,
                          (self._bboxHWXYPredTile[:, :, :, :, 2] + offset_column) / float(self._gridSize[1])
                          + self._bboxHWXYPredTile[:,:,:,:,1]/2.0,
                          (self._bboxHWXYPredTile[:, :, :, :, 3] + offset_row) / float(self._gridSize[0])
                          + self._bboxHWXYPredTile[:,:,:,:,0]/2.0],
                         axis=-1) # (None, 13,13, prdNPG*objNPG, rcMinMax)
        #get leftup point and rightdown point of intersection
        lu = tf.maximum(boxGT[...,:2], boxPr[...,:2])
        rd = tf.minimum(boxGT[...,2:], boxPr[...,2:])
        #get intersection Area
        intersection = tf.maximum(0.0, rd - lu)
        intersectionArea = intersection[...,0] * intersection[...,1]
        #get boxGTArea and boxPrArea
        boxGTArea = (boxGT[...,2] - boxGT[...,0]) * (boxGT[...,3] - boxGT[...,1])
        boxPrArea = (boxPr[...,2] - boxPr[...,0]) * (boxPr[...,3] - boxPr[...,1])
        unionArea = tf.maximum(boxGTArea + boxPrArea - intersectionArea, 1e-7)        
        self._IOU = tf.clip_by_value(intersectionArea/unionArea, 0.0, 1.0) #IOU = (None, 13,13, prdNPG*objNPG)
        print 'self._IOU.shape', self._IOU.shape
        print 'done!'

class darknet_classifier(object):
    def __init__(
            self,
            dataPath='./',
            nameScope='nolbo',
            imgSize = (None,None), batchSize = 64, learningRate = 0.0001,
            classNum = 40,
            coreLayerActivation = tf.nn.elu,
            lastLayerActivation = tf.nn.softmax):
        self._imgList = None
        self._imgClassList = None
        self._dataPath = dataPath
        self._nameScope = nameScope
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
        self._coreAct = coreLayerActivation
        self._lastAct = lastLayerActivation
        self._classNum = classNum
        self.variables = None
        self.update_ops = None
        self._inputImg = None
        self._outputClass = None
        self._outputClassGT = None
        self._optimizer = None
        self._loss = None
        self._buildNetwork()
        self._createLossAndOptimizer()
        self._createEvaluation()
        #init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        #initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # init = tf.initialize_all_variables()
        #launch the session
        self._sess.run(init)

    def _buildNetwork(self):
        print "build network..."
        self._inputImg = tf.placeholder(tf.float32, shape=(None, self._imgSize[0], self._imgSize[1], 3))
        self._outputClassGT = tf.placeholder(tf.float32, shape=(None, self._classNum))
        self._classifier = darknet.encoder_singleVectorOutput(
            outputVectorDim=self._classNum,
            coreActivation=self._coreAct,
            lastLayerActivation=self._lastAct,
            lastLayerPooling='average',
            nameScope=self._nameScope+'-enc')
        self._outputClass = self._classifier(self._inputImg)
        print "done!"
    def _createLossAndOptimizer(self):
        print "create loss and optimizer..."
        self._lr = tf.placeholder(tf.float32, shape=[])
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        # self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr)
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.9, use_nesterov=True)
        # self._optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=self._lr)
        # self._optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._lr)
        # self._optimizer = tf.train.AdagradOptimizer(learning_rate=self._lr)
        # self._loss = tf.reduce_mean(
        #     tf.losses.softmax_cross_entropy(logits=self._outputClass, onehot_labels=self._outputClassGT))

        # self._loss = tf.reduce_sum(
        #     tf.losses.softmax_cross_entropy(onehot_labels=self._outputClassGT, logits=self._outputClass))

        self._loss = tf.reduce_mean(-tf.reduce_sum(self._outputClassGT * tf.log(self._outputClass+1e-8), reduction_indices=1))
        # self._loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._outputClassGT, logits=self._outputClass))
        with tf.control_dependencies(self._classifier.allUpdate_ops):
            self._optimizer = self._optimizer.minimize(
                self._loss, var_list = self._classifier.allVariables
            )
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self._optimizer = self._optimizer.minimize(
        #         self._loss
        #     )
        print "done!"
    def _createEvaluation(self):
        prediction = tf.argmax(self._outputClass, -1)
        gt = tf.argmax(self._outputClassGT, -1)
        equality = tf.equal(prediction, gt)
        self._acc = tf.reduce_mean(tf.cast(equality, tf.float32))
        self._top5Acc = tf.reduce_mean(
            tf.cast(
                tf.nn.in_top_k(
                    predictions=self._outputClass,
                    targets=gt,
                    k=5),
                tf.float32))

    def fit(self, batchDict):
        feed_dict = {
            self._inputImg : batchDict['inputImages'],
            self._outputClassGT : batchDict['classIndexList'],
            self._lr : batchDict['learningRate']
        }
        opt, lossResult, accResult, top5AccResult = self._sess.run([self._optimizer, self._loss, self._acc, self._top5Acc], feed_dict=feed_dict)
        return lossResult, accResult, top5AccResult

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.save(self._sess, eLastPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)

class VAE3D(object):
    def __init__(self, network_architecture):
        self._net_arc = network_architecture
        self._enc_arc = network_architecture['encoder']
        self._dec_arc = network_architecture['decoder']

        #encoder input
        self._inputImages = tf.placeholder(tf.float32, shape=np.concatenate([[None], self._enc_arc['inputImgDim']]))

        #priornet input
        self._latentDim = 0
        if self._net_arc['class']:
            self._latentDim += self._net_arc['zClassDim']
            self._classListGT = tf.placeholder(tf.float32, shape=([None, self._net_arc['classDim']]))
        if self._net_arc['inst']:
            self._latentDim += self._net_arc['zInstDim']
            self._instListGT = tf.placeholder(tf.float32, shape=([None, self._net_arc['instDim']]))
        if self._net_arc['rot']:
            self._latentDim += self._net_arc['zRotDim']
            self._EulerAngleGT = tf.placeholder(tf.float32, shape=([None, self._net_arc['rotDim']]))

        #decoder input
        self._z = None
        #decoder output
        self._outputImages = None
        #decoder output GT
        self._outputImagesGT = tf.placeholder(tf.float32, shape=np.concatenate([[None], self._enc_arc['inputImgDim']]))

        #for training
        self._lr = None

        self._buildNetwork()
        self._createLoss()
        self._setOptimizer()
        # init the session

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        # initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # launch the session
        self._sess.run(init)

    def _buildNetwork(self):
        print 'build network...'
        if self._net_arc['isTraining']:
            self._classPriornet = priornet.priornet(
                inputDim=self._net_arc['classDim'], outputDim=self._net_arc['zClassDim'],
                hiddenLayerNum=2, scopeName=self._net_arc['nameScope'] + '-classPrior', training=self._net_arc['isTraining'],
                coreActivation=tf.nn.elu,
                constLogVar=0.0)
            self._instPriornet = priornet.priornet(
                inputDim=self._net_arc['classDim'] + self._net_arc['instDim'], outputDim=self._net_arc['zInstDim'],
                hiddenLayerNum=2, scopeName=self._net_arc['nameScope'] + '-instPrior', training=self._net_arc['isTraining'],
                coreActivation=tf.nn.elu,
                constLogVar=0.0)
            self._classMeanPrior, self._classLogVarPrior = \
                self._classPriornet(self._classListGT)
            self._instMeanPrior, self._instLogVarPrior = \
                self._instPriornet(tf.concat([self._classListGT, self._instListGT], axis=-1))


        self._encoder = ae.encoder(self._enc_arc, encoderName='encoder3D')
        self._decoder = ae.decoder(self._dec_arc, decoderName='nolbo-dec')
        encOutput = self._encoder(self._inputImages)
        self._meanPredWOTanh, self._logVarPred = tf.split(encOutput, num_or_size_splits=2, axis=-1)
        self._classMeanPred = self._meanPredWOTanh[...,0:self._net_arc['zClassDim']]
        self._instMeanPred = self._meanPredWOTanh[...,self._net_arc['zClassDim']:self._net_arc['zClassDim']+self._net_arc['zInstDim']]
        self._rotMeanPred = self._meanPredWOTanh[..., self._net_arc['zClassDim'] + self._net_arc['zInstDim']:]
        self._rotMeanSinPred = self._meanPredWOTanh[...,self._net_arc['zClassDim']+self._net_arc['zInstDim']:self._net_arc['zClassDim']+self._net_arc['zInstDim']+self._net_arc['zRotDim']/2]
        self._rotMeanSinPred = tf.nn.tanh(self._rotMeanSinPred)
        self._rotMeanCosPred = self._meanPredWOTanh[...,self._net_arc['zClassDim']+self._net_arc['zInstDim']+self._net_arc['zRotDim'] / 2 :]
        self._rotMeanCosPred = tf.nn.tanh(self._rotMeanCosPred)
        self._meanPred = tf.concat([self._classMeanPred, self._instMeanPred, self._rotMeanSinPred, self._rotMeanCosPred], axis=-1)

        self._classLogVarPred = self._logVarPred[..., 0:self._net_arc['zClassDim']]
        self._instLogVarPred = self._logVarPred[...,self._net_arc['zClassDim']:self._net_arc['zClassDim'] + self._net_arc['zInstDim']]
        self._rotLogVarPred = self._logVarPred[..., self._net_arc['zClassDim'] + self._net_arc['zInstDim']:]

        self._zsAll = []
        self._zs = sampling(mu=self._meanPred, logVar=self._logVarPred)
        self._zsAll += [self._zs]

        self._meanPredwithPriorClass = tf.concat(
            [self._classMeanPrior, self._instMeanPred, self._rotMeanSinPred, self._rotMeanCosPred], axis=-1)
        self._logVarPredwithPriorClass = tf.concat(
            [self._classLogVarPrior, self._instLogVarPred, self._rotLogVarPred], axis=-1)
        self._zsPriorClass = sampling(mu=self._meanPredwithPriorClass, logVar=self._logVarPredwithPriorClass)
        self._zsAll += [self._zsPriorClass]

        self._meanPredwithPriorInst = tf.concat(
            [self._classMeanPred, self._instMeanPrior, self._rotMeanSinPred, self._rotMeanCosPred], axis=-1)
        self._logVarPredwithPriorInst = tf.concat(
            [self._classLogVarPred, self._instLogVarPrior, self._rotLogVarPred], axis=-1)
        self._zsPriorInst = sampling(mu=self._meanPredwithPriorInst, logVar=self._logVarPredwithPriorInst)
        self._zsAll += [self._zsPriorInst]

        self._meanPredOrg = tf.concat(
            [self._classMeanPred, self._instMeanPred, tf.zeros_like(self._rotMeanSinPred), tf.ones_like(self._rotMeanCosPred)], axis=-1)
        self._zsOrg = sampling(mu=self._meanPredOrg, logVar=self._logVarPred)
        self._zsAll += [self._zsOrg]

        self._zsAll = tf.concat(self._zsAll, axis=0)
        print self._zsAll.shape
        self._outputImages = self._decoder(self._zsAll)

        if self._net_arc['isTraining']:
            self._p, self._r = create_evaluation(xTarget=self._outputImagesGT, xPred=self._outputImages)

        print "done!"

    def _createLoss(self):
        print "create loss..."
        self._binaryLoss = tf.reduce_mean(
            binary_loss(xPred=self._outputImages, xTarget=self._outputImagesGT, gamma=0.60, b_range=False))
        self._regulizerLoss \
            = tf.reduce_mean(
            regulizer_loss(z_mean=self._classMeanPrior, z_logVar=self._classLogVarPrior, dist_in_z_space=5.0)
            +
            regulizer_loss(
                z_mean=self._instMeanPrior, z_logVar=self._instLogVarPrior,
                dist_in_z_space=2.0, class_input=self._classListGT)
        )
        self._meanPrior = tf.concat([self._classMeanPrior, self._instMeanPrior, self._EulerAngleGT], axis=-1)
        # self._meanPrior_p360 = tf.concat([self._classMeanPrior, self._instMeanPrior, self._EulerAngleGT + 360.0],
        #                                  axis=-1)
        # self._meanPrior_n360 = tf.concat([self._classMeanPrior, self._instMeanPrior, self._EulerAngleGT - 360.0],
        #                                  axis=-1)
        self._logVarPrior = tf.concat(
            [self._classLogVarPrior, self._instLogVarPrior, 2.0 * tf.log(0.1) * tf.ones_like(self._EulerAngleGT)],
            axis=-1)
        #
        # nlb0 = nlb_loss(mean=self._meanPred, logVar=self._logVarPred,
        #                 mean_target=self._meanPrior, logVar_target=self._logVarPrior)
        # nlbp360 = nlb_loss(mean=self._meanPred, logVar=self._logVarPred,
        #                    mean_target=self._meanPrior_p360, logVar_target=self._logVarPrior)
        # nlbn360 = nlb_loss(mean=self._meanPred, logVar=self._logVarPred,
        #                    mean_target=self._meanPrior_n360, logVar_target=self._logVarPrior)
        # self._nlbLoss = tf.reduce_mean(tf.minimum(tf.minimum(nlb0, nlbp360), nlbn360))

        self._nlbLoss = tf.reduce_mean(
            nlb_loss(mean=self._meanPred,
                     logVar=self._logVarPred,mean_target=self._meanPrior, logVar_target=self._logVarPrior))

        self._totalLoss = self._binaryLoss + self._nlbLoss + 0.01 * self._regulizerLoss

    def _setOptimizer(self):
        print "set optimizer..."
        self._lr = tf.placeholder(tf.float32, shape=[])
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.9, use_nesterov=True)
        with tf.control_dependencies(
                                        self._encoder.update_ops + self._decoder.update_ops
                                + self._classPriornet.update_ops + self._instPriornet.update_ops
        ):
            self._optimizer = self._optimizer.minimize(
                self._totalLoss,
                var_list
                =
                self._encoder.variables + self._decoder.variables
                +
                self._classPriornet.variables + self._instPriornet.variables
            )

    def fit(self, batchDict):
        feed_dict = {
            self._lr: batchDict['learningRate'],
            self._inputImages: batchDict['inputImages'],
            self._outputImagesGT : batchDict['outputImages'],
            self._classListGT: batchDict['classList'],
            self._instListGT: batchDict['instList'],
            self._EulerAngleGT: batchDict['EulerAngleSinCos']
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._binaryLoss, self._regulizerLoss, self._nlbLoss
        precisionRecall = self._p, self._r
        opt, loss, pr = self._sess.run([optimizer, lossList, precisionRecall], feed_dict=feed_dict)
        return loss, pr

    def saveEncoder(self, savePath='./'):
        ePath = os.path.join(savePath, self._net_arc['nameScope'] + '_encoderCore.ckpt')
        self._encoder.saver.save(self._sess, ePath)
    def saveDecoder(self, savePath='./'):
        dPath = os.path.join(savePath, self._net_arc['nameScope'] + '_decoder.ckpt')
        self._decoder.saver.save(self._sess, dPath)
    def savePriornet(self, savePath='./'):
        pCPath = os.path.join(savePath, self._net_arc['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(savePath, self._net_arc['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.save(self._sess, pCPath)
        self._instPriornet.saver.save(self._sess, pIPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoder(savePath)
        self.saveDecoder(savePath)
        self.savePriornet(savePath)
    def restoreEncoder(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._net_arc['nameScope'] + '_encoderCore.ckpt')
        self._encoder.saver.restore(self._sess, eCorePath)
    def restoreDecoder(self, restorePath='./'):
        dPath = os.path.join(restorePath, self._net_arc['nameScope'] + '_decoder.ckpt')
        self._decoder.saver.restore(self._sess, dPath)
    def restorePriornet(self, restorePath='./'):
        pCPath = os.path.join(restorePath, self._net_arc['nameScope'] + '_classPrior.ckpt')
        pIPath = os.path.join(restorePath, self._net_arc['nameScope'] + '_instPrior.ckpt')
        self._classPriornet.saver.restore(self._sess, pCPath)
        self._instPriornet.saver.restore(self._sess, pIPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoder(restorePath)
        self.restoreDecoder(restorePath)
        self.restorePriornet(restorePath)

class RenderforCNN_classifier(object):
    def __init__(self, classNum, instNum, rotDim, coreActivation=tf.nn.elu, nameScope='nolbo'):
        self._classNum = classNum
        self._instNum = instNum
        self._rotDim = rotDim
        self._coreAct = coreActivation
        self._nameScope = nameScope
        #inputs
        self._inputImages = None
        #outputs
        self._classPred = None
        self._instPred = None
        self._azimuthPred = None
        self._elevationPred = None
        self._in_plane_rotPred = None
        #output ground truth
        self._classGT = None
        self._instGT = None
        self._azimuthGT = None
        self._elevationGT = None
        self._in_plane_rotGT = None
        #parameters for training
        self._learningRate = None

        self._initVariables()
        self._buildNetwork()
        self._createEvaluation()
        self._createLoss()
        self._setOptimizer()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        # initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # launch the session
        self._sess.run(init)

    def _initVariables(self):
        # self._inputImages = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self._learningRate = tf.placeholder(tf.float32, shape=[])
        self._classGT = tf.placeholder(tf.float32, shape=(None, self._classNum))
        self._instGT = tf.placeholder(tf.float32, shape=(None, self._instNum))
        self._azimuthGT = tf.placeholder(tf.float32, shape=(None, self._rotDim))
        self._elevationGT = tf.placeholder(tf.float32, shape=(None, self._rotDim))
        self._in_plane_rotGT = tf.placeholder(tf.float32, shape=(None, self._rotDim))
    def _buildNetwork(self):
        print "build network..."
        self._inputImages = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self._classifier = darknet.encoder_singleVectorOutput(
            outputVectorDim = self._classNum + self._instNum + 3*self._rotDim,
            coreActivation=self._coreAct,
            lastLayerActivation=None,
            nameScope=self._nameScope + '-enc')
        self._encoderOutput = self._classifier(self._inputImages)
        self._classPred = tf.nn.softmax(self._encoderOutput[...,:self._classNum])
        self._instPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum:self._classNum+self._instNum])
        self._azimuthPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum+self._instNum:self._classNum+self._instNum+self._rotDim])
        self._elevationPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum+self._instNum+self._rotDim:self._classNum+self._instNum+2*self._rotDim])
        self._in_plane_rotPred = tf.nn.softmax(
            self._encoderOutput[...,self._classNum+self._instNum+2*self._rotDim:])
        print "done!"

    def _createLoss(self):
        print "create loss..."
        # self._classLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._classGT, logits=self._classPred))
        # self._instLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._instGT, logits=self._instPred))
        # self._azimuthLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._azimuthGT, logits=self._azimuthPred))
        # self._elevationLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._elevationGT, logits=self._elevationPred))
        # self._in_plane_rotLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._in_plane_rotGT, logits=self._in_plane_rotPred))
        self._classLoss = categorical_crossentropy(gt=self._classGT, pred=self._classPred)
        self._instLoss = categorical_crossentropy(gt=self._instGT, pred=self._instPred)
        self._azimuthLoss = categorical_crossentropy(gt=self._azimuthGT, pred=self._azimuthPred)
        self._elevationLoss = categorical_crossentropy(gt=self._elevationGT, pred=self._elevationPred)
        self._in_plane_rotLoss = categorical_crossentropy(gt=self._in_plane_rotGT, pred=self._in_plane_rotPred)
        self._loss = self._classLoss + self._instLoss + \
                     (self._azimuthLoss + self._elevationLoss + self._in_plane_rotLoss)
        print "done!"

    def _setOptimizer(self):
        print "set optimizer..."
        # self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learningRate)
        # self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._learningRate)
        self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._learningRate, momentum=0.9, use_nesterov=True)
        with tf.control_dependencies(self._classifier.allUpdate_ops):
            self._optimizer = self._optimizer.minimize(
                self._loss, var_list = self._classifier.allVariables
            )
        print "done!"

    def _createEvaluation(self):
        classPrediction = tf.argmax(self._classPred, -1)
        classGT = tf.argmax(self._classGT, -1)
        classEquality = tf.equal(classPrediction, classGT)
        self._classAcc = tf.reduce_mean(tf.cast(classEquality, tf.float32))
        instPrediction = tf.argmax(self._instPred, -1)
        instGT = tf.argmax(self._instGT, -1)
        instEquality = tf.equal(instPrediction, instGT)
        self._instAcc = tf.reduce_mean(tf.cast(instEquality, tf.float32))

        azimuthPrediction = tf.argmax(self._azimuthPred, -1)
        azimuthGT = tf.argmax(self._azimuthGT, -1)
        azimuthEquality = tf.equal(azimuthPrediction, azimuthGT)
        self._azimuthAcc = tf.reduce_mean(tf.cast(azimuthEquality, tf.float32))
        self._azimuthTop5Acc = tf.reduce_mean(tf.cast(
            tf.nn.in_top_k(predictions=self._azimuthPred, targets=azimuthGT, k=5),tf.float32))

        elevationPrediction = tf.argmax(self._elevationPred, -1)
        elevationGT = tf.argmax(self._elevationGT, -1)
        elevationEquality = tf.equal(elevationPrediction, elevationGT)
        self._elevationAcc = tf.reduce_mean(tf.cast(elevationEquality, tf.float32))
        self._elevationTop5Acc = tf.reduce_mean(tf.cast(
            tf.nn.in_top_k(predictions=self._elevationPred, targets=elevationGT, k=5), tf.float32))

        in_plane_rotPrediction = tf.argmax(self._in_plane_rotPred, -1)
        in_plane_rotGT = tf.argmax(self._in_plane_rotGT, -1)
        in_plane_rotEquality = tf.equal(in_plane_rotPrediction, in_plane_rotGT)
        self._in_plane_rotAcc = tf.reduce_mean(tf.cast(in_plane_rotEquality, tf.float32))
        self._in_plane_rotTop5Acc = tf.reduce_mean(tf.cast(
            tf.nn.in_top_k(predictions=self._in_plane_rotPred, targets=in_plane_rotGT, k=5), tf.float32))

    def fit(self, batchDict):
        feed_dict = {
            self._inputImages : batchDict['inputImages'],
            self._classGT : batchDict['classList'],
            self._instGT : batchDict['instList'],
            self._azimuthGT : batchDict['azimuth'],
            self._elevationGT : batchDict['elevation'],
            self._in_plane_rotGT : batchDict['in_plane_rot'],
            self._learningRate : batchDict['learningRate']
        }
        opt, totalLoss, classAcc, instAcc, azAcc, azTop5Acc, elAcc, elTop5Acc, ipAcc, ipTop5Acc = self._sess.run(
            [self._optimizer, self._loss,
             self._classAcc, self._instAcc,
             self._azimuthAcc, self._azimuthTop5Acc,
             self._elevationAcc, self._elevationTop5Acc,
             self._in_plane_rotAcc, self._in_plane_rotTop5Acc
             ], feed_dict=feed_dict)
        return totalLoss, classAcc, instAcc, azAcc, azTop5Acc, elAcc, elTop5Acc, ipAcc, ipTop5Acc

    def saveEncoderCore(self, savePath='./'):
        eCorePath = os.path.join(savePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.save(self._sess, eCorePath)
    def saveEncoderLastLayer(self, savePath='./'):
        eLastPath = os.path.join(savePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.save(self._sess, eLastPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoderCore(savePath)
        self.saveEncoderLastLayer(savePath)
    def restoreEncoderCore(self, restorePath='./'):
        eCorePath = os.path.join(restorePath, self._nameScope + '_encoderCore.ckpt')
        self._classifier.coreSaver.restore(self._sess, eCorePath)
    def restoreEncoderLastLayer(self, restorePath='./'):
        eLastPath = os.path.join(restorePath, self._nameScope + '_encoderLastLayer.ckpt')
        self._classifier.lastLayerSaver.restore(self._sess, eLastPath)
    def restoreNetworks(self, restorePath='./'):
        self.restoreEncoderCore(restorePath)
        self.restoreEncoderLastLayer(restorePath)


