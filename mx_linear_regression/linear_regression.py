#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:40:11 2018

@author: jayhan
"""

from mxnet import nd,gluon,autograd
import mxnet as mx

sample_num = 10000
input_size = 2
output_size = 1
data_ctx = mx.cpu()
model_ctx = mx.cpu()

def real_func(x):
    y = 2 * x[:, 0] - 3.4 * x[:, 1] + 4.2
    return y

X = nd.random_normal(shape=(sample_num,input_size),ctx=data_ctx)
noise = 0.1*nd.random_normal(shape=(sample_num,),ctx=data_ctx)
Y = real_func(X)+noise

print(X[0])
print(Y[0])

import matplotlib.pyplot as plt
plt.scatter(X[:,0].asnumpy(),Y.asnumpy())
plt.show()

batch_size = 4
data_iterator = gluon.data.DataLoader(gluon.data.ArrayDataset(X,Y,),batch_size=batch_size,shuffle=True)

for data,label in data_iterator:
    print(data)
    print(label)
    break

##build net
w = nd.random_normal(shape=(input_size,output_size))
b = nd.random_normal(shape=(output_size))
params = [w,b]
for param in params:
    param.attach_grad()    
def net(X):
    return nd.dot(X,w)+b

##define loss function
def square_loss(yhat,y):
    return nd.mean((yhat-y)**2)
##define optimization function
def SGD(params,lr):
    for param in params:
        param[:] = param-lr*param.grad

epoch = 10
learning_rate = 0.001
batch_num = sample_num/batch_size
for i in range(epoch):
    loss_sum = 0
    for data,label in data_iterator:
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1,1))
        ##forward
        with autograd.record():
            output = net(data)
            loss = square_loss(output,label)
        ##backw    
        loss.backward()
        
        SGD(params,learning_rate)
        loss_sum += loss.asscalar()
    print(loss_sum/batch_num)
print(params)
        
        
        
        
        
        