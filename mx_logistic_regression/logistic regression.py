#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:18:01 2018

@author: jayhan
"""
import mxnet as mx
from mxnet import nd,autograd,gluon
import matplotlib.pyplot as plt

def logistic(z):
    return 1./(1.+nd.exp(-z))

x = nd.arange(-5,5,.1)
y = logistic(x)

plt.plot(x.asnumpy(),y.asnumpy()) ## display as curve with line connection between points
plt.show()
