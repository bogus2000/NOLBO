import numpy as np
import tensorflow as tf

class priornet(object):
    def __init__(self, inputDim, outputDim, hiddenLayerNum, scopeName="priornet", training=False,
                 coreActivation=tf.nn.leaky_relu, lastLayerActivation=None,
                 constLogVar = None,
                 reuse=False):
        self._inputDim = inputDim
        self._outputDim = outputDim
        self._hiddenLayerNum = hiddenLayerNum
        self._scopeName = scopeName
        self._training=training
        self._coreAct = coreActivation
        self._lastAct = lastLayerActivation
        self._constLogVar = constLogVar
        self._reuse = reuse
    def __call__(self, inputVector, bhPhase=True, trainable=True):
        print "priornet - "+self._scopeName
        with tf.variable_scope(self._scopeName, reuse=self._reuse):
            ratio = np.power(float(self._outputDim)/float(self._inputDim), 1.0/float(self._hiddenLayerNum))
            layerDim = self._inputDim
            print inputVector.shape
            hidden = 2.0*inputVector-1.0
            print "mean prior"
            for i in range(self._hiddenLayerNum-1):
                layerDim = layerDim * ratio
                hidden = tf.layers.dense(inputs=hidden,units=int(layerDim),activation=None,use_bias=True)
                hidden = tf.layers.batch_normalization(hidden)
                hidden = tf.layers.dropout(hidden, rate=0.5, training=self._training)
                if self._coreAct != None:
                    hidden = self._coreAct(hidden)
                print hidden.shape
            meanPrior = tf.layers.dense(inputs=hidden,units=self._outputDim,activation=None,use_bias=True)
            if self._lastAct != None:
                meanPrior = self._lastAct(meanPrior)
            print meanPrior.shape

            if self._constLogVar == None:
                print "logVar prior"
                layerDim = self._inputDim
                print inputVector.shape
                hidden = 2.0*inputVector - 1.0
                for i in range(self._hiddenLayerNum - 1):
                    layerDim = layerDim * ratio
                    hidden = tf.layers.dense(inputs=hidden, units=int(layerDim), activation=None, use_bias=True)
                    hidden = tf.layers.batch_normalization(hidden)
                    hidden = tf.layers.dropout(hidden, rate=0.5, training=self._training)
                    if self._coreAct != None:
                        hidden = self._coreAct(hidden)
                    print hidden.shape
                logVarPrior = tf.layers.dense(inputs=hidden, units=self._outputDim, activation=None, use_bias=True)
                if self._lastAct != None:
                    logVarPrior = self._lastAct(logVarPrior)
                print logVarPrior.shape
            elif self._constLogVar == self._constLogVar:
                print "logVar prior : constant "+str(self._constLogVar)
                logVarPrior = self._constLogVar * tf.ones_like(meanPrior)
            else:
                logVarPrior = None
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scopeName)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._scopeName)
        self.saver = tf.train.Saver(var_list=self.variables)
        return meanPrior, logVarPrior