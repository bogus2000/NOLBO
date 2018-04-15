import numpy as np
import tensorflow as tf
import cv2
import time
import os, random, re, pickle

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
def noisy(image, noise_typ):
    if noise_typ == "gaussian":
        mean = 0
        var = 0.01
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "salt&pepper":
        s_vs_p = 0.5
        amount = 0.05
        out = np.copy(image)
        
        # print image.size
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, np.max((i-1,1)), int(num_salt)) for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, np.max((i-1,1)), int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(np.abs(image) * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        gauss = np.random.randn(*image.shape)
        gauss = gauss.reshape(image.shape)        
        noisy = image + image * 0.1* gauss
        return noisy
    else:
        return image
