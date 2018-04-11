import numpy as np
import tensorflow as tf
import cv2
import time
import os
import src.darknet.darknet_core as darknet_core

class darknet_classifier(object):
    def __init__(self, dataPath='./', imgSize = (416,416), batchSize = 64, learningRate = 0.0001):
        self._imgList = None
        self._imgClassList = None
        self._dataPath = dataPath
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
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
        self._darknetCore = darknet_core.darknet19_core()
        self._pretraining = darknet_core.darknet19_pretraining(self._classNum)
        coreOutput = self._darknetCore(self._inputImg)
        self._outputClass = self._pretraining(coreOutput)
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
        with tf.control_dependencies(self._darknetCore.update_ops + self._pretraining.update_ops):
            self._optimizer = self._optimizer.minimize(
                self._loss, var_list = self._darknetCore.variables + self._pretraining.variables
            )
        print "done!"
    def _saveNetwork(self, savePath='./'):
        dCorePath = os.path.join(savePath,'dCore.ckpt')
        pretrainPath = os.path.join(savePath,'pretrain.ckpt')
        self._darknetCore.saver.save(self._sess, dCorePath)
        self._pretraining.saver.save(self._sess, pretrainPath)
    def _restoreNetwork(self, restorePath='./'):
        dCorePath = os.path.join(restorePath,'dCore.ckpt')
        pretrainPath = os.path.join(restorePath,'pretrain.ckpt')
        self._darknetCore.saver.restore(self._sess, dCorePath)
        self._pretraining.saver.restore(self._sess, pretrainPath)
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
