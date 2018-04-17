#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:18:01 2018

@author: jayhan
"""
import mxnet as mx
from mxnet import nd,autograd,gluon
import matplotlib.pyplot as plt
ctx_data = mx.cpu()
ctx_model = mx.cpu()

def logistic(z):
    return 1.0/(1.0+nd.exp(-z))

#x = nd.arange(-5,5,.1)
#y = logistic(x)
#plt.plot(x.asnumpy(),y.asnumpy()) ## display as curve with line connection between points
#plt.show()

def read_adult(filename,ctx):
    with open(filename) as f:
        lines = f.readlines()
    sample_num = len(lines)
    label,data =nd.zeros((sample_num,1),ctx=ctx),nd.zeros((sample_num,123),ctx=ctx)    
    for i,line in enumerate(lines):
        line = line.split()
        label[i]=((int(line[0])+1)/2)
        for taken in line[1:]:
            index = int(taken[:-2])-1
            data[i,index] = 1
    return data,label

data_train,label_train = read_adult('data/adult/a1a.txt', ctx_data)
data_test,label_test = read_adult('data/adult/a1a.t', ctx_data)
print(data_train[0,:],label_train[0,:])
print(data_test[0,:],label_test[0,:])

batch_size = 64
dataset_train = gluon.data.DataLoader(gluon.data.ArrayDataset(data_train,label_train),batch_size=batch_size,shuffle=True)
dataset_test = gluon.data.DataLoader(gluon.data.ArrayDataset(data_test,label_test),batch_size=batch_size,shuffle=True)
##net
net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(),ctx=ctx_model)
##loss
def log_loss(output,y):
    yhat = logistic(output)
    return -nd.nansum(y*nd.log(yhat)+(1-y)*nd.log(1-yhat))
##optimizer
optimizer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.01})

##operator process
epochs = 30
loss_total =0
sample_num = len(label_train)
loss_seq=[]
times =[]
for e in range(epochs):
    loss_total =0
    for data,label in dataset_train:
        data = data.as_in_context(ctx_model)
        label = label.as_in_context(ctx_model).reshape((-1,1))
        with autograd.record():
            output = net(data)
            loss = log_loss(output,label)
        loss.backward()
        optimizer.step(batch_size)
        loss_total += nd.sum(loss).asscalar()
    loss_mean = loss_total/sample_num
    loss_seq.append(loss_mean)
    times.append(e)
    print("loss:",loss_mean)
plt.plot(times,loss_seq)
plt.show()
    
num_correct =0
num_total = len(label_test)
for i,(data,label) in enumerate(dataset_test):
    data = data.as_in_context(ctx_model)
    label = label.as_in_context(ctx_model)
    prediction = (nd.sign(net(data))+1)/2
    num_correct += nd.sum(prediction==label).asscalar()
    
print("accuracy: {:.2f}%".format(100*num_correct/num_total))
    



