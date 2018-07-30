# import numpy as np
# import tensorflow as tf
# import cv2
# import time
# import os, random, re, pickle
# import sys
#
# # nolbo_multiObjectConfig = {
# #     'inputImgDim':[448,448,1],
# #     'maxPoolNum':5,
# #     'predictorNumPerGrid':11,
# #     'bboxDim':5,
# #     'class':True, 'zClassDim':64, 'classDim':24,
# #     'inst':True, 'zInstDim':64, 'instDim':1000,
# #     'rot':True, 'zRotDim':3, 'rotDim':3,
# #     'trainable':True,
# #     'decoderStructure':{
# #         'outputImgDim':[64,64,64,1],
# #         'trainable':True,
# #         'filterNumList':[512,256,128,64,1],
# #         'kernelSizeList':[4,4,4,4,4],
# #         'stridesList':[1,2,2,2,2],
# #         'activation':tf.nn.leaky_relu,
# #         'lastLayerActivation':tf.nn.sigmoid
# #     }
# # }
#
# def categorical_crossentropy(gt, pred):
#     loss = tf.reduce_mean(-tf.reduce_sum(gt*tf.log(pred+1e-9), reduction_indices=1))
#     return loss
#
# def leaky_relu(x, alpha=0.2):
#     return tf.maximum(x, alpha * x)
#
# def sampling(mu, logVar):
#     epsilon = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
#     samples = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(logVar)), epsilon))
#     return samples
#
# def regulizer_loss(z_mean, z_logVar, dist_in_z_space, class_input = None):
#     dim_z = tf.shape(z_mean)[-1]
#     batch_size = tf.shape(z_mean)[0]
#     z_m_repeat = tf.reshape(z_mean, tf.stack([batch_size, 1, dim_z]))
#     z_m_repeat_tr = tf.reshape(z_mean, tf.stack([1, batch_size, dim_z]))
#     z_logVar_repeat = tf.reshape(z_logVar, tf.stack([batch_size, 1, dim_z]))
#     z_m_repeat = tf.tile(z_m_repeat, tf.stack([1, batch_size, 1]))
#     z_m_repeat_tr = tf.tile(z_m_repeat_tr, tf.stack([batch_size, 1, 1]))
#     z_logVar_repeat = tf.tile(z_logVar_repeat, tf.stack([1, batch_size, 1]))
#
#     diff = tf.abs(z_m_repeat - z_m_repeat_tr) / tf.exp(0.5 * z_logVar_repeat)
#     diff = tf.reduce_sum(diff, axis=-1)
#
#     diff_in_z = diff - dist_in_z_space * tf.cast(dim_z, tf.float32) * tf.ones_like(diff)
#     diff_in_z = tf.where(
#         tf.greater(diff_in_z, tf.zeros_like(diff_in_z)), tf.zeros_like(diff_in_z), tf.square(diff_in_z))
#
#     dot_cos = tf.reduce_sum(z_m_repeat*z_m_repeat_tr,axis=-1)/(
#         tf.norm(z_m_repeat,axis=-1)*tf.norm(z_m_repeat_tr, axis=-1))
#     dot_cos_abs = tf.abs(dot_cos)
#     loss_reg = diff_in_z + diff_in_z * dot_cos_abs
#
#     if class_input != None:
#         c_i_repeat = tf.reshape(
#             class_input, tf.stack([batch_size, 1, class_input.get_shape().as_list()[-1]]))
#         c_i_repeat_tr = tf.reshape(
#             class_input, tf.stack([1, batch_size, class_input.get_shape().as_list()[-1]]))
#         c_i_repeat = tf.tile(c_i_repeat, tf.stack([1, batch_size, 1]))
#         c_i_repeat_tr = tf.tile(c_i_repeat_tr, tf.stack([batch_size, 1, 1]))
#         c_i_diff_abs = tf.abs(c_i_repeat - c_i_repeat_tr)
#         c_i_diff_sum = tf.reduce_sum(c_i_diff_abs, axis=-1)
#         # if categories are the same, get 1
#         # else, get zero
#         c_i_diff = tf.where(tf.greater(c_i_diff_sum, 0.0), tf.zeros_like(c_i_diff_sum), tf.ones_like(c_i_diff_sum))
#         loss_reg = loss_reg * c_i_diff
#
#     # loss_reg = tf.reduce_mean(loss_reg)
#     return loss_reg
#
# def binary_loss(xPred, xTarget, epsilon = 1e-7, gamma=0.5, b_range=False):
#     b_range = float(b_range)
#     voxelDimTotal = np.prod(xPred.get_shape().as_list()[1:])
#     yTarget = -b_range + (2.0 * b_range + 1.0) * tf.reshape(xTarget, (-1, voxelDimTotal))
#     yPred = tf.clip_by_value(tf.reshape(xPred, (-1, voxelDimTotal)), clip_value_min=epsilon, clip_value_max=1.0 - epsilon)
#     bce_loss = - tf.reduce_sum(gamma * yTarget * tf.log(yPred) + (1.0 - gamma) * (1.0 - yTarget) * tf.log(1.0 - yPred),
#                                axis=-1)
#     return bce_loss
#
# def nlb_loss(mean, logVar, mean_target, logVar_target):
#     vectorDimTotal = np.prod(mean.get_shape().as_list()[1:])
#     m = tf.reshape(mean, (-1, vectorDimTotal))
#     lV = tf.reshape(logVar, (-1, vectorDimTotal))
#     m_t = tf.reshape(mean_target, (-1, vectorDimTotal))
#     lV_t = tf.reshape(logVar_target, (-1, vectorDimTotal))
#     loss = tf.reduce_sum(0.5 * (lV_t - lV) + tf.div((tf.exp(lV) + tf.square(m - m_t)),(2.0 * tf.exp(lV_t))) - 0.5,
#                          axis=-1)
#     return loss
#
# def create_evaluation(xTarget, xPred, prob=0.5):
#     yTarget = tf.cast(tf.reshape(xTarget, (-1, np.prod(xTarget.get_shape().as_list()[1:]))), tf.float32)
#     yPred = tf.cast(tf.greater_equal(tf.reshape(xPred, (-1, np.prod(xPred.get_shape().as_list()[1:]))), prob), tf.float32)
#     TP = tf.reduce_mean(tf.reduce_sum(yTarget * yPred, axis=-1))
#     FP = tf.reduce_mean(tf.reduce_sum((1.0 - yTarget) * yPred, axis=-1))
#     FN = tf.reduce_mean(tf.reduce_sum(yTarget * (1.0 - yPred), axis=-1))
#     p = TP / (TP + FP + 1e-10)
#     r = TP / (TP + FN + 1e-10)
#     return p,r