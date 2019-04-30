import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import skimage
import random
import shutil
import utils
import sys
import glob

data=sys.argv[1]
print(data)

############OPEN CONFIG FILE#########################
file=open(data+'Network.txt')
base_scaler=int(file.readline())
baseline_noise=float(file.readline())
x_dim=int(file.readline())
mean=float(file.readline())
std=float(file.readline())
model=int(file.readline())
channels=int(file.readline())
y_dim=x_dim
file.close()

##############SETUP NETWORK###########################

tf.reset_default_graph()
#Input and output
x=tf.placeholder(dtype=tf.float32, shape=[None, x_dim,y_dim,channels], name='x')
y=tf.placeholder(dtype=tf.float32, shape=[None, x_dim,y_dim,2], name='y')
lr=tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
dr=tf.placeholder(dtype=tf.float32, shape=[], name='dropout_rate')

xr=x
yr=y

#base_scaler=32

#Going down
A1=tf.layers.conv2d(xr, base_scaler, [5,5], padding='SAME', activation=utils.leaky_relu)
A2=tf.layers.conv2d(A1, base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

B0=tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
B1=tf.layers.conv2d(B0, 2*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
B2=tf.layers.conv2d(B1, 2*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

C0=tf.nn.max_pool(B2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
C1=tf.layers.conv2d(C0, 4*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
C2=tf.layers.conv2d(C1, 4*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

D0=tf.nn.max_pool(C2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
D1=tf.layers.conv2d(D0, 8*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
D2=tf.layers.conv2d(D1, 8*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

E0=tf.nn.max_pool(D2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
E1=tf.layers.conv2d(E0, 16*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
E2=tf.layers.conv2d(E1, 16*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

#Coming up
DD0=tf.layers.conv2d_transpose(E2, 8*base_scaler, kernel_size=[3,3], strides=[2, 2], padding='SAME')
DD1=tf.concat(axis=3, values=[DD0,D2])
DD2=tf.layers.conv2d(DD1, 8*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
DD3=tf.layers.conv2d(DD2, 8*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

CC0=tf.layers.conv2d_transpose(DD3, 4*base_scaler, kernel_size=[3,3], strides=[2, 2], padding='SAME')
CC1=tf.concat(axis=3, values=[CC0,C2])
CC2=tf.layers.conv2d(CC1, 4*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
CC3=tf.layers.conv2d(CC2, 4*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

BB0=tf.layers.conv2d_transpose(CC3, 2*base_scaler, kernel_size=[3,3], strides=[2, 2], padding='SAME')
#BB0=tf.contrib.layers.conv2d_transpose(C2, 2*base_scaler, kernel_size=[3,3], stride=[2, 2], padding='SAME')
BB1=tf.concat(axis=3, values=[BB0,B2])
BB2=tf.layers.conv2d(BB1, 2*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
BB3=tf.layers.conv2d(BB2, 2*base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

AA0=tf.layers.conv2d_transpose(BB3, base_scaler, kernel_size=[3,3], strides=[2, 2], padding='SAME')
AA1=tf.concat(axis=3, values=[AA0,A2])
AA2=tf.layers.conv2d(AA1, base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)
AA3=tf.layers.conv2d(AA2, base_scaler, [3,3], padding='SAME', activation=utils.leaky_relu)

logits=tf.layers.conv2d(AA3, 2, [1,1], padding='SAME', activation=utils.leaky_relu)
probs=tf.tanh(logits, name='probabilities')

diff=tf.subtract(probs, yr)
LSQ=tf.multiply(diff,diff)
#Added this to make the outlines more potent in error function
OutError, MaskError=tf.split(LSQ, [1,1], 3)
loss=1*tf.reduce_mean(OutError)+0.1*tf.reduce_mean(MaskError)
loss=tf.reduce_mean(LSQ, name='error')
l2_loss = tf.losses.get_regularization_loss()
#loss=loss+l2_loss/1000

train_op=tf.train.AdamOptimizer(learning_rate=lr, name='trainer').minimize(loss)

tf.summary.scalar('loss', loss)
tf.summary.image('input', tf.reshape(x[:,:,:,0], [-1, x_dim, y_dim,1]), 3)
tf.summary.image('standard', tf.reshape(y[:,:,:,0], [-1, x_dim, y_dim,1]) , 3)
tf.summary.image('outline', tf.reshape(probs[:,:,:,0], [-1, x_dim, y_dim,1]), 3)
tf.summary.image('mask', tf.reshape(probs[:,:,:,1], [-1, x_dim, y_dim,1]), 3)


merge = tf.summary.merge_all()

########################LOAD BEST NETWORK##########################

saver = tf.train.Saver()
sess=tf.Session()
sess.run(tf.global_variables_initializer())

data_model=data+'NewModels/Model'+str(model)
saver.restore(sess,data_model)

#####################PROCESSING FUNCTION#########################

def process_file(sess, file_path, save_dir):
    file_data=skimage.external.tifffile.imread(file_path)
    if (len(file_data.shape)==3):
        file_data=np.reshape(file_data, [file_data.shape[0], x_dim, x_dim, 1])
    else:
        file_data=np.swapaxes(np.swapaxes(file_data,1,2), 2,3)
    inference_data=file_data/655350.0
    
    #true_test2_data=true_test2_data/np.std(true_test2_data)*0.02
    #true_test2_data=true_test2_data-np.mean(true_test2_data)+0.55
    #v1true_test2_data=(true_test2_data-.578)/.138+0.5
    inference_data=(inference_data-mean)/std+0.5

    channels=inference_data.shape[3]
    num_images=inference_data.shape[0]
    output=np.zeros([num_images,x_dim,x_dim,channels+2])
    process_batch_size=20
    print('Starting')
    for t in range(0,num_images,process_batch_size):
        endrng=np.min((t+process_batch_size,num_images))
        inference_batch_data=inference_data[t:(t+process_batch_size), :,:,:]
        results=sess.run(probs, feed_dict={x:inference_batch_data})
        output[t:(t+process_batch_size),:,:,0:channels]=inference_batch_data[:,:,:,0:channels]
        output[t:(t+process_batch_size),:,:,channels]=results[:,:,:,0]
        output[t:(t+process_batch_size),:,:,channels+1]=results[:,:,:,1]
    print('Done')
    inference_data=0
    output[:,:,:,0:channels]=file_data[:,:,:,0:channels]
    output=np.swapaxes(np.swapaxes(output,3,2),2,1)
    np.place(output, output<0, 0)
    file_name=file_path.split('/')[-1]
    print(output.dtype)
    skimage.external.tifffile.imsave(save_dir+file_name, output.astype('float32'), imagej=True)
    print('Written')
    return output


##################PROCESS INPUT FOLDER ##############################
for f in glob.glob(data+'Data_input/*'):
    output=process_file(sess, f, data+'Data_output/')
