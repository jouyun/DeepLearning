import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import skimage
import random
import shutil
import sys
import utils

data=sys.argv[1]
print(data)

both=skimage.external.tifffile.imread(data+'Training_RotShift.tif')
both=np.swapaxes(np.swapaxes(both,1,2), 2,3)

x_dim=both.shape[1]
y_dim=both.shape[2]
channels=both.shape[3]-2

train_data=both[:,:,:,0:-2]/655350.0
train_truth=both[:,:,:,[-2,-1]]

non_zeros=np.where(train_data!=0)
mean=np.mean(train_data[non_zeros])
std=np.std(train_data[non_zeros])


train_data=(train_data-mean)/(std*1)+0.5
train_truth[np.where(train_truth>0.1)]=1

validation=skimage.external.tifffile.imread(data+'Validation_annotated.tif')
validation=np.swapaxes(np.swapaxes(validation,1,2), 2,3)

validation_data=validation[:,:,:,0:-2]/655350.0
validation_data=(validation_data-mean)/(std*1)+0.5

validation_truth=validation[:,:,:,[-2,-1]]
validation_truth[np.where(validation_truth>0.1)]=1


file=open(data+'Network.txt')
base_scaler=int(file.readline())
baseline_noise=float(file.readline())
file.close()
print([base_scaler, baseline_noise])


##############################NETWORK BLOCK######################################
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


#######TRAINING BLOCK################
try:
    shutil.rmtree('/scratch/c2015/DeepFiji/logs/')
except:
    print("log file doesn't exist")

tf.set_random_seed(123456)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=50)
test_writer = tf.summary.FileWriter('/scratch/c2015/DeepFiji/logs/170/test')
current_best=[0,1000000000]


learning_rates=[0.000195, 0.0001, 0.00002, 0.00001]
learning_rate_steps=[500,1500, 2200, 2200]
current_step=0
for lrate, lrs in zip(learning_rates, learning_rate_steps):
    for i in range(current_step, lrs):
        idx=np.random.choice(train_data.shape[0], replace=False, size=[25])
        cur_train=train_data[idx,:,:,:]+np.random.uniform(-baseline_noise, baseline_noise, 1)
        cur_truth=train_truth[idx,:,:]
        _, results, losses=sess.run([train_op,  probs, loss], feed_dict={x:cur_train, y:cur_truth, lr:lrate})
        #train_writer.add_summary(summary, i)
        if (i%100==0):
            print(i)
            print("Training loss: ",losses)
            #idx=np.random.choice(validation_data.shape[0], replace=False, size=[50])
            idx=range(0,3, 1)
            sub_validation_data=validation_data[idx, :,:,:]
            sub_validation_truth=validation_truth[idx, :,:]
            summary, results, losses=sess.run([merge, probs, loss], feed_dict={x:sub_validation_data, y:sub_validation_truth})
            test_writer.add_summary(summary, i)
            print(results.shape)
            print("Validation loss: ",losses)
            if (losses<current_best[1]):
                current_best=[i, losses]
                file=open(data+'Network.txt', 'w')
                file.write(str(base_scaler)+'\n')
                file.write(str(baseline_noise)+'\n')
                file.write(str(x_dim)+'\n')
                file.write(str(mean)+'\n')
                file.write(str(std)+'\n')
                file.write(str(i)+'\n')
                file.write(str(channels)+'\n')
                file.close()
            #for ti in range (0,3):
             #   utils.plot_3x1(sub_validation_data[ti,:,:,0], results[ti,:,:,0], results[ti,:,:,1])
             #   plt.show()
            saver.save(sess, data+'NewModels/Model'+str(i))
    current_step=lrs
    
    
