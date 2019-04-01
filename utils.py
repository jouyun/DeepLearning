import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from scipy.optimize import curve_fit
from PIL import Image
import os.path

def read_tiff(path, n_images):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(n_images):
        try:
            img.seek(i)
            slice_ = np.zeros((img.height, img.width))
            for j in range(slice_.shape[0]):
                for k in range(slice_.shape[1]):
                    slice_[j,k] = img.getpixel((j, k))

            images.append(slice_)

        except EOFError:
            # Not enough frames in img
            break

    return np.array(images)

def leaky_relu(inp, alpha=0.2):
    return tf.maximum(inp * alpha, inp)

def plot_2x2(g1, g2, g3, g4):
    f,axes=plt.subplots(2,2, figsize=(8,8))
    axes[0,0].imshow(g1)
    axes[0,1].imshow(g2)
    axes[1,0].imshow(g3)
    axes[1,1].imshow(g4)

def plot_3x1(g1, g2, g3):
    f,axes=plt.subplots(1,3, figsize=(15,15))
    axes[0].imshow(g1)
    axes[1].imshow(g2)
    axes[2].imshow(g3)
    
def plot_4x1(g1, g2, g3, g4):
    f,axes=plt.subplots(1,4, figsize=(15,10))
    axes[0].imshow(g1)
    axes[1].imshow(g2)
    axes[2].imshow(g3)
    axes[3].imshow(g4)

def plot_5x1(g1, g2, g3, g4, g5):
    f,axes=plt.subplots(1,5, figsize=(15,10))
    axes[0].imshow(g1)
    axes[1].imshow(g2)
    axes[2].imshow(g3)
    axes[3].imshow(g4)    
    axes[4].imshow(g5)
    
def plot_6x1(g1, g2, g3, g4, g5, g6):
    f,axes=plt.subplots(1,6, figsize=(15,10))
    axes[0].imshow(g1)
    axes[1].imshow(g2)
    axes[2].imshow(g3)
    axes[3].imshow(g4)
    axes[4].imshow(g5)
    axes[5].imshow(g6)

def get_image(fname, shp, force_reload=False):   
    rfname=fname[0:-4]+'.npy'
    if ((os.path.isfile(rfname)) and (force_reload!=True)):
        rtn=np.load(rfname)
    else:
        if (len(shp)>3):
            rtn=read_tiff(fname,shp[0]*shp[1])
        else:
            rtn=read_tiff(fname, shp[0])
        rtn=rtn.reshape(shp)
        np.save(rfname, rtn)
    return rtn

#[num_images, slices, width, height, channels]
def get_multichannel_image(fname, shp, force_reload=False):   
    rfname=fname[0:-4]+'.npy'
    if ((os.path.isfile(rfname)) and (force_reload!=True)):
        rtn=np.load(rfname)
    else:
        rtn=read_tiff(fname, shp[0]*shp[1]*shp[4])
        rtn=rtn.reshape(shp[0],shp[1],shp[4],shp[2],shp[3])
        rtn=np.swapaxes(np.swapaxes(rtn,2,3),3,4)
        #Now is in proper order, need to get rid of singleton dimensions
        if (shp[1]==1):
            rtn=rtn.reshape(shp[0],shp[2],shp[3],shp[4])
        if (shp[4]==1):
            rtn=rtn.reshape(shp[0],shp[1],shp[2],shp[3])
        if (shp[1]==1 and shp[4]==1):
            rtn=rtn.reshape(shp[0],shp[2],shp[3])
        np.save(rfname, rtn)
    return rtn

def format_raw_image(img, shp):
    rtn=img
    slices=shp[0]
    width=shp[1]
    height=shp[2]
    channels=shp[3]
    frames=rtn.shape[0]/slices/width/height/channels
    frames=int(frames)
    rtn=rtn.reshape([frames, slices, channels, width, height])

    rtn=np.swapaxes(np.swapaxes(rtn,2,3),3,4)
    if (slices==1):
        rtn=rtn.reshape(frames, width, height, channels)
    if (channels==1):
        rtn=rtn.reshape(frames, slices, width, height)
    if (slices==1 and channels==1):
        rtn=rtn.reshape(frames, width, height)
    return rtn

#shp=[slices, width, height, channels]
def get_raw_float_image(fname, shp):
    rtn=np.fromfile(fname, np.float32)
    rtn=rtn.byteswap()
    rtn=format_raw_image(rtn,shp)
    return rtn