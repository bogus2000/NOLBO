import numpy as np
import tensorflow as tf
import cv2
import time
import os
import sys


# def leaky_relu(x, alpha=0.2):
#     return tf.maximum(x, alpha * x)

class darknet19_core(object):
    def __init__(self, nameScope='dartnet19_core',
                 trainable=True, bnPhase=True, reuse=False, activation = tf.nn.elu):
        self._reuse = reuse
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._nameScope = nameScope
        self._activation = activation
        self.variables = None
        self.update_ops = None
        self.saver = None
    def _conv(self, inputs, filters, kernel_size):
        hiddenC = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=1, padding='same', activation=None, trainable=self._trainable, use_bias=False)
        hiddenC = tf.layers.batch_normalization(inputs=hiddenC, training=self._bnPhase, trainable=self._trainable)
        hiddenC = self._activation(hiddenC)
        print hiddenC.shape
        return hiddenC
    def _maxPool(self, inputs, pool_size=(2,2), strides=2, padding='same'):
        hiddenP = tf.layers.max_pooling2d(inputs, pool_size=pool_size, strides=strides, padding=padding)
        print hiddenP.shape
        return hiddenP
    def __call__(self, inputImg):
        print self._nameScope
        with tf.variable_scope(self._nameScope, reuse=self._reuse):
            # hiddenNormalized = tf.layers.batch_normalization(inputs=inputImg, training=self._bnPhase, trainable=self._trainable)
            hiddenC1 = self._conv(inputs=inputImg, filters=32, kernel_size=3)
            hiddenP1 = self._maxPool(inputs=hiddenC1)
            
            hiddenC2 = self._conv(inputs=hiddenP1, filters=64, kernel_size=3)
            hiddenP2 = self._maxPool(inputs=hiddenC2)
            
            hiddenC31 = self._conv(inputs=hiddenP2, filters=128, kernel_size=3)
            hiddenC32 = self._conv(inputs=hiddenC31, filters=64, kernel_size=1)
            hiddenC33 = self._conv(inputs=hiddenC32, filters=128, kernel_size=3)
            hiddenP3 = self._maxPool(inputs=hiddenC33)
            
            hiddenC41 = self._conv(inputs=hiddenP3, filters=256, kernel_size=3)
            hiddenC42 = self._conv(inputs=hiddenC41, filters=128, kernel_size=1)
            hiddenC43 = self._conv(inputs=hiddenC42, filters=256, kernel_size=3)
            hiddenP4 = self._maxPool(inputs=hiddenC43)
            
            hiddenC51 = self._conv(inputs=hiddenP4, filters=512, kernel_size=3)
            hiddenC52 = self._conv(inputs=hiddenC51, filters=256, kernel_size=1)
            hiddenC53 = self._conv(inputs=hiddenC52, filters=512, kernel_size=3)
            hiddenC54 = self._conv(inputs=hiddenC53, filters=256, kernel_size=1)
            hiddenC55 = self._conv(inputs=hiddenC54, filters=512, kernel_size=3)
            hiddenP5 = self._maxPool(inputs=hiddenC55)

            # #dummy
            # hiddenC61 = self._conv(inputs=hiddenP5, filters=1024, kernel_size=3)
            # hiddenC62 = self._conv(inputs=hiddenC61, filters=512, kernel_size=1)
            # hiddenC63 = self._conv(inputs=hiddenC62, filters=1024, kernel_size=3)
            # hiddenC64 = self._conv(inputs=hiddenC63, filters=512, kernel_size=1)
            # hiddenP5 = self._conv(inputs=hiddenC64, filters=1024, kernel_size=3)
            
            hiddenC61 = self._conv(inputs=hiddenP5, filters=1024, kernel_size=3)
            hiddenC62 = self._conv(inputs=hiddenC61, filters=512, kernel_size=1)
            hiddenC63 = self._conv(inputs=hiddenC62, filters=1024, kernel_size=3)
            hiddenC64 = self._conv(inputs=hiddenC63, filters=512, kernel_size=1)
            hiddenC65 = self._conv(inputs=hiddenC64, filters=1024, kernel_size=3)
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope)
        self.saver = tf.train.Saver(var_list=self.variables)
        outputs = hiddenC65
        return outputs

class encoder(object):
    def __init__(self, outputVectorDim,
                 nameScope='encoder_singleVectorOutput',
                 trainable=True, bnPhase=True, reuse=False,
                 coreActivation=tf.nn.leaky_relu,
                 lastLayerActivation=None,
                 lastLayerPooling='None',
                 ):
        self._outputVectorDim = outputVectorDim
        self._nameScope = nameScope
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._reuse = reuse
        self._coreAct = coreActivation
        self._lastAct = lastLayerActivation
        self._lastPool = lastLayerPooling
        self.variables = None
        self.update_ops = None
        self.saver = None
        self._darknetCore = None
    def __call__(self, inputImg):
        print "encoder_singleVectorOutput - "+self._nameScope
        self._darknetCore = darknet19_core(activation=self._coreAct,
                                           bnPhase=self._bnPhase,
                                           nameScope=self._nameScope+"_DKCore",
                                           reuse=self._reuse,trainable=self._trainable)
        hidden = self._darknetCore(inputImg)
        with tf.variable_scope(self._nameScope+'_LastConv', reuse=self._reuse):
            print "encoder_singleVectorOutput_lastLayer - "+self._nameScope
            hidden = tf.layers.conv2d(inputs=hidden, filters=self._outputVectorDim, kernel_size=1,
                                      strides=1, padding='same', activation=None,
                                      trainable=self._trainable, use_bias=False)
            if self._lastPool == 'max':
                hidden = tf.reduce_max(hidden, axis=[1,2])
            elif self._lastPool == 'average':
                hidden = tf.reduce_mean(hidden, axis=[1,2])
            elif self._lastPool == 'None':
                print 'last layer pooling : None'
                pass
            print hidden.shape
            if self._lastAct!=None:
                print "last layer activation is", self._lastAct
                hidden = self._lastAct(hidden)
            print hidden.shape
        self._reuse=True
        self.variables = [self._darknetCore.variables, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+'_LastConv')]
        self.update_ops = [self._darknetCore.update_ops, tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope+'_LastConv')]
        self.allVariables = self._darknetCore.variables + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+'_LastConv')
        self.allUpdate_ops = self._darknetCore.update_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope+'_LastConv')
        self.coreVariables = self._darknetCore.variables
        self.lastLayerVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+'_LastConv')
        self.coreSaver = tf.train.Saver(var_list=self.coreVariables)
        self.lastLayerSaver = tf.train.Saver(var_list=self.lastLayerVariables)
        outputs = hidden
        return outputs
    
class encoder_gridOutput(object):
    def __init__(self, lastLayserChannelNum,
                 nameScope='encoder_gridOutput',
                 trainable=True, bnPhase=True, reuse=False, coreActivation=tf.nn.leaky_relu, lastLayerActivation=None):
        self._lastLCN = lastLayserChannelNum
        self._nameScope = nameScope
        self._trainable = trainable
        self._bnPhase = bnPhase
        self._reuse = reuse
        self._coreAct = coreActivation
        self._lastAct = lastLayerActivation
        self._darknetCore = None
    def __call__(self, inputImg):
        print "encoder_gridOutput - " + self._nameScope
        self._darknetCore = darknet19_core(activation=self._coreAct, bnPhase=self._bnPhase, nameScope=self._nameScope+"DKCore",reuse=self._reuse,trainable=self._trainable)
        hidden = self._darknetCore(inputImg)
        with tf.variable_scope(self._nameScope+"LastConv", reuse=self._reuse):
            print "encoder_gridOutput_lastLayer - " + self._nameScope
            hidden = tf.layers.conv2d(
                inputs=hidden, filters=self._lastLCN,
                activation=self._lastAct, kernel_size=1, strides=1, padding='same', use_bias=True)
            print hidden.shape
        self._reuse=True
        self.variables = [self._darknetCore.variables,tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+"LastConv")]
        self.update_ops = [self._darknetCore.update_ops,tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope+"LastConv")]
        self.allVariables = self._darknetCore.variables + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+"LastConv")
        self.allUpdate_ops = self._darknetCore.update_ops + tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._nameScope+"LastConv")
        self.coreVariables = self._darknetCore.variables
        self.lastLayerVariables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._nameScope+"LastConv")
        self.coreSaver = tf.train.Saver(var_list=self.coreVariables)
        self.lastLayerSaver = tf.train.Saver(var_list=self.lastLayerVariables)
        outputs = hidden
        return outputs
