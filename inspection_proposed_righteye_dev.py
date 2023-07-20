#!/usr/bin/env python
"""Example code of learning a large scale 'convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""

from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing
import random
import sys
import threading
import time
from PIL import Image
import six
import cPickle as pickle
from six.moves import queue
import chainer
import math
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from chainer import serializers
from chainer import cuda
import nin
import numpy as np
import proposed_nin_ini_righteye_dev as ni

def inspection(path):

	all_start = time.time()

	sys.argv =[""]
	parser = argparse.ArgumentParser(description='Image inspection using chainer')
	parser.add_argument('--model','-m',default='proposed_righteye_model')
	parser.add_argument('--mean', default='proposed_righteye_mean.npy')
	args = parser.parse_args()
	xp = cuda.cupy
	
	def read_image(path, center=False, flip=False):

		image = Image.open(path)
		image = np.asarray(image).transpose(2, 0, 1)
		if center:
			top = left = cropwidth / 2
		else:
			top = random.randint(0, cropwidth - 1)
			left = random.randint(0, cropwidth - 1)
		bottom = model.insize + top
		right = model.insize + left
		image = image[:, top:bottom, left:right].astype(np.float32)
		image -= mean_image[:, top:bottom, left:right]
		image /= 255
		if flip and random.randint(0, 1) == 0:
			return image[:, :, ::-1]
		else:
			return image

	mean_image = pickle.load(open(args.mean, 'rb'))
	model = ni.model
	cropwidth = 256 - model.insize
	model.to_gpu()
	#model.to_cpu()

	def predict(net, x):
		h = F.max_pooling_2d(F.relu(net.mlpconv1(x)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(net.mlpconv2(h)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(net.mlpconv3(h)), 3, stride=2)
		h = net.mlpconv4(F.dropout(h, train=net.train))
		h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 7))
		return F.softmax(h)

	img = read_image(path)
	x = np.ndarray((1, 3, model.insize, model.insize), dtype=np.float32)
	x[0]=img
	x_data = xp.asarray(x)
	x = chainer.Variable(x_data, volatile='on')
	#x = chainer.Variable(np.asarray(x), volatile='on')
	score = predict(model,x)
	categories = np.loadtxt("labels.txt", str, delimiter="\t")
	top_k = 7
	prediction = zip(score.data[0].tolist(), categories)
	prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

	listscore = [15,15,14,14,14,14,14]	

	print('--------------------------------------------------------')
	print(path)
	for rank, (score, name) in enumerate(prediction[:top_k], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
		listscore[rank-1] = score*100		
	print('--------------------------------------------------------')

	all_elapsed_time = time.time() - all_start
	print('all')	
	print(all_elapsed_time)

	return 0

	'''
	if(listscore[0] >= listscore[1]):
		print(listscore[0])
		return "success"
	elif(listscore[0] < listscore[1]):
		print(listscore[1])	
		return "fail"
	'''

