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
