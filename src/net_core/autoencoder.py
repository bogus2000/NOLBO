import numpy as np
import tensorflow as tf
import random as rd
import sys
import time
import os,re

#=========== autoencoder architecture example (3D GAN) ===============
# encoderStructure = {
#     'inputImgDim':[64,64,64,1],
#     'trainable':True,
#     'activation':tf.nn.leaky_relu,
#     'filterNumList':[64,128,256,512,400],
#     'kernelSizeList':[4,4,4,4,4],
#     'stridesList':[2,2,2,2,1]
# }
# decoderStructure = {
#     'outputImgDim':[64,64,64,1],
#     'trainable':True,
#     'activation':tf.nn.leaky_relu,
#     'filterNumList':[512,256,128,64,1],
#     'kernelSizeList':[4,4,4,4,4],
#     'stridesList':[1,2,2,2,2],
#     'lastLayerActivation':tf.nn.sigmoid
# }

class encoder(object):
    def __init__(self, architecture, encoderName='encoder'):
        self._arc = architecture
        self._reuse = False
        self._scopeName = encoderName
        self.variables, self.update_ops, self.saver = None,None,None
        self._trainable = None
        self._conv = None
        self._imgRank = len(self._arc['inputImgDim'])
        if self._imgRank==2+1:
            self._conv=tf.layers.conv2d
        elif self._imgRank==3+1:
            self._conv=tf.layers.conv3d
        self._trainable = self._arc['trainable']
        self._activation = self._arc['activation']
        self._filterNumList = self._arc['filterNumList']
        self._kernelSizeList = self._arc['kernelSizeList']
        self._stridesList = self._arc['stridesList']
    def _convEnc(self, inputs, filters, kernelSize, strides=2, padding='same'):
        hiddenC = self._conv(inputs=inputs, filters=filters, kernel_size=kernelSize, strides=strides, padding=padding, activation=None, trainable=self._trainable, use_bias=False)
        hiddenC = tf.layers.batch_normalization(inputs=hiddenC, training=self._bnPhase, trainable=self._trainable)
        hiddenC = self._activation(hiddenC)
        print hiddenC.shape
        return hiddenC
    def __call__(self, inputs, bnPhase=True):
        print 'encoder - '+self._scopeName
        print inputs.shape
        self._bnPhase = bnPhase
        with tf.variable_scope(self._scopeName, reuse=self._reuse):            
            totalDepth = len(self._filterNumList)
            for depth in range(totalDepth-1):
                filterNum = self._filterNumList[depth]
                kernelSize = self._kernelSizeList[depth]
                strides = self._stridesList[depth]
                if depth==0:
                    hidden = inputs
                hidden = self._convEnc(hidden, filters=filterNum, kernelSize=kernelSize, strides=strides)
            hidden = self._conv(hidden, self._filterNumList[totalDepth-1], self._kernelSizeList[totalDepth-1], self._stridesList[totalDepth-1], padding='valid', activation=None, trainable=self._trainable, use_bias=True)
            print hidden.shape
            hidden = tf.layers.flatten(hidden)
            print hidden.shape
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scopeName)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._scopeName)
        self.saver = tf.train.Saver(var_list=self.variables)
        return hidden

class decoder(object):
    def __init__(self, architecture, decoderName='decoder'):
        self._arc = architecture
        self._reuse = False
        self._scopeName = decoderName
        self.variables, self.update_ops, self.saver = None,None,None
        self._trainable = None
        self._convTrans = None
        self._outputImgDim = self._arc['outputImgDim']
        self._outputImgRank = len(self._outputImgDim)
        if self._outputImgRank==2+1:
            self._convTrans =tf.layers.conv2d_transpose
        elif self._outputImgRank==3+1:
            self._convTrans =tf.layers.conv3d_transpose
        self._trainable = self._arc['trainable']
        self._activation = self._arc['activation']
        self._filterNumList = self._arc['filterNumList']
        self._kernelSizeList = self._arc['kernelSizeList']
        self._stridesList = self._arc['stridesList']
        self._lastLayerAct = self._arc['lastLayerActivation']
    def _linearTransform(self, inputs, outputDim):
        inputs = tf.reshape(inputs, (-1, np.prod(inputs.get_shape().as_list()[1:])))
        hiddenL = tf.layers.dense(inputs=inputs, units=outputDim, trainable=self._trainable, use_bias=True)
        print hiddenL.shape
        return hiddenL
    def _convDec(self, inputs, filters, kernelSize, strides, padding='same'):
        hiddenC = self._convTrans(inputs=inputs, filters=filters, kernel_size=kernelSize, strides=strides, padding=padding, activation=None, trainable=self._trainable, use_bias=False)
        hiddenC = tf.layers.batch_normalization(inputs=hiddenC, training=self._bnPhase, trainable=self._trainable)
        hiddenC = self._activation(hiddenC)
        print hiddenC.shape
        return hiddenC
    def __call__(self, inputs, bnPhase=True):
        print 'decoder - '+self._scopeName
        print inputs.shape
        self._bnPhase = bnPhase
        linearOutputDim = 4*np.prod(inputs.get_shape().as_list()[1:])
        convInputImgDimWOChannel = self._outputImgDim[:-1]/np.prod(self._stridesList)
        convInputChannel = int(linearOutputDim/np.prod(convInputImgDimWOChannel))
        linearOutputDim = np.prod(convInputImgDimWOChannel) * convInputChannel
        convInputDim = np.concatenate([[-1],convInputImgDimWOChannel,[convInputChannel]])
        hidden = self._linearTransform(inputs=inputs, outputDim=linearOutputDim)
        print hidden.shape
        hidden = tf.reshape(hidden, shape=convInputDim)
        print hidden.shape
        totalDepth = len(self._filterNumList)
        with tf.variable_scope(self._scopeName, reuse=self._reuse):
            for depth in range(totalDepth-1):
                filterNum = self._filterNumList[depth]
                kernelSize = self._kernelSizeList[depth]
                strides = self._stridesList[depth]
                hidden = self._convDec(hidden, filters=filterNum, kernelSize=kernelSize, strides=strides)
            hidden = self._convTrans(hidden, self._filterNumList[totalDepth-1], self._kernelSizeList[totalDepth-1], self._stridesList[totalDepth-1], padding='same', activation=None, trainable=self._trainable, use_bias=False)
            print hidden.shape
            if self._lastLayerAct != None:
                hidden = self._lastLayerAct(hidden)
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._scopeName)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._scopeName)
        self.saver = tf.train.Saver(var_list=self.variables)
        return hidden

# inputs = tf.zeros((10000,64,64,64,1))
# enc = encoder(architecture=encoderStructure, encoderName='encoder')
# dec = decoder(architecture=decoderStructure, decoderName='decoder')
# latent = enc(inputs)
# output = dec(latent)
