#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:59:05 2018

@author: jayhan
"""
import mxnet as mx
from mxnet import nd,gluon,autograd
import numpy as np
import matplotlib.pyplot as plt
mx.random.seed(1)

sample_y_linear = nd.random_normal(shape=(2,10))

print(sample_y_linear)
print(sample_y_linear-nd.max(sample_y_linear,axis=1).reshape((-1,1)))

def softmax(y_linear):
    ##利用广播之前，需要保证某一维度长度相等
    exp = nd.exp(y_linear-nd.max(y_linear,axis=1).reshape((-1,1)))
    norm =nd.sum(exp,axis=1).reshape((-1,1))
    return exp/norm
    
print(softmax(sample_y_linear))

def transform(data,label):
    return data.astype(np.float32)/255,label.astype(np.float32)
mnist_train = gluon.data.vision.MNIST(train=True,transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False,transform=transform)
batch_size = 64
dataloader_train = gluon.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
dataloader_test = gluon.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=True)
ctx_data = mx.cpu()
ctx_model = mx.cpu()

image,label = mnist_train[0]
print(image.shape,label)
plt.imshow(image.reshape((28,28)).asnumpy())

## define 
input_size = 28*28
output_size = 10
W = nd.random_normal(shape=(input_size,output_size),ctx=ctx_data)
b = nd.random_normal(shape=(1,output_size),ctx=ctx_data)
params = [W,b]
for param in params:
    param.attach_grad()
## net or model
def net(data):
    y0 = nd.dot(data,W)+b
    yhat =softmax(y0)
    return  yhat
## loss
def cross_entropy(yhat,y):
    return -nd.sum(y*nd.log(yhat+1e-6))
## opimizer
def sgd(params,lr):
    for param in params:
        param[:] = param-lr*param.grad

epoch = 5
loss_progress =[]
progress =[]
data_size = len(mnist_train)
for e in range(epoch):
    loss_total = 0
    for i,(data,label) in enumerate(dataloader_train):
        data = data.as_in_context(ctx_model).reshape((-1,784))
        label = label.as_in_context(ctx_model)
        y = nd.one_hot(label,10)
        with autograd.record():
            yhat = net(data)
            loss = cross_entropy(yhat,y)
        loss.backward()
        sgd(params,0.01)
        loss_total += nd.sum(loss).asscalar()
    loss_avg = loss_total/data_size
    loss_progress.append(loss_avg)
    progress.append(e)
    print("loss average:",loss_avg)
plt.figure("trainning progress")
plt.plot(progress,loss_progress)
plt.show()

correct_times = 0
numerator,denorminator = 0,0
for i,(data,label) in enumerate(dataloader_test):
    data = data.as_in_context(ctx_model).reshape((-1,784))
    label = label.as_in_context(ctx_model)
    y = nd.one_hot(label,10)
    output = net(data)
    prediction = nd.argmax(output,axis=1)
    numerator += int(nd.sum(prediction==label).asscalar())
    denorminator += data.shape[0] ## 0:sample number, 1:row size, 2:column size
    print("Accuracy is {0:.2f}%,{1:d}/{2:d}".format(100*numerator/denorminator,numerator,denorminator))
    
        
    
        