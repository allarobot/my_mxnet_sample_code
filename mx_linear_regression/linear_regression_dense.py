#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 21:37:22 2018

@author: jayhan
"""

import mxnet as mx
from mxnet import nd,gluon,autograd
sample_num = 10000
input_size = 2
output_size = 1

X = nd.random_normal(shape=(sample_num,input_size))
noise = 0.1*nd.random_normal(shape=(sample_num,))

def real_func(x):
    return 2 * x[:, 0] - 3.4 * x[:, 1] + 4.2

Y = real_func(X)+noise
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X,Y),batch_size=batch_size,shuffle=True)

##build net
net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(),ctx=mx.cpu())

#create loss function
loss_l2 = gluon.loss.L2Loss()
#create optimizer
optimizer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.001})

epoch = 10
run_per_epoch = sample_num/batch_size
for i in range(epoch):
    loss_cum = 0
    for data,label in train_data:
        data = data.as_in_context(mx.cpu())
        label = label.as_in_context(mx.cpu()).reshape((-1,1))
        #forward
        with autograd.record():
            output = net(data)
            loss  = loss_l2(output,label) 
        #backward
        loss.backward()
        #param optimization
        optimizer.step(batch_size)
        loss_cum += nd.mean(loss).asscalar()
    print("loss:",loss_cum/run_per_epoch)
 
#param output       
for param in net.collect_params().values():
    print(param.name,param.data())