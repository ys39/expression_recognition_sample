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
import proposed_nin_ini_all_dev as ni

def inspection(face_path,righteye_path,lefteye_path,mouth_path):

	# initial execute
	# ---------------------------------------------------------------------------

	sys.argv =[""]
	parser = argparse.ArgumentParser(description='Image inspection using chainer')
	parser.add_argument('--model','-m',default='')
	parser.add_argument('--mean', default='')
	args = parser.parse_args()
	xp = cuda.cupy

	cropwidth = 29
	insize = 227

	# read image	
	def read_image(path, center=False, flip=False):

		image = Image.open(path)
		image = np.asarray(image).transpose(2, 0, 1)
		if center:
			top = left = cropwidth / 2
		else:
			top = random.randint(0, cropwidth - 1)
			left = random.randint(0, cropwidth - 1)
		bottom = insize + top
		right = insize + left
		image = image[:, top:bottom, left:right].astype(np.float32)
		image -= mean_image[:, top:bottom, left:right]
		image /= 255
		if flip and random.randint(0, 1) == 0:
			return image[:, :, ::-1]
		else:
			return image

	# predict facial expression
	def predict(net, x):
		h = F.max_pooling_2d(F.relu(net.mlpconv1(x)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(net.mlpconv2(h)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(net.mlpconv3(h)), 3, stride=2)
		h = net.mlpconv4(F.dropout(h, train=net.train))
		h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 7))
		return F.softmax(h)


	all_start = time.time()

	# face
	# ---------------------------------------------------------------------------
	mean_image = pickle.load(open('proposed_face_mean.npy', 'rb'))
	face_model = ni.face_model
	face_model.to_gpu()

	img = read_image(face_path)
	x = np.ndarray((1, 3, insize, insize), dtype=np.float32)
	x[0]=img
	x_data = xp.asarray(x)
	x = chainer.Variable(x_data, volatile='on')

	score = predict(face_model,x)
	categories = np.loadtxt("labels.txt", str, delimiter="\t")
	top_k = 7
	prediction = zip(score.data[0].tolist(), categories)
	#prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

	face_ls = [15,15,14,14,14,14,14]	

	print('face----------------------------------------------------')
	print(face_path)

	for rank, (score, name) in enumerate(prediction[:top_k], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
		face_ls[rank-1] = score*100		
	print('--------------------------------------------------------')

	# righteye
	# ---------------------------------------------------------------------------
	mean_image = pickle.load(open('proposed_righteye_mean.npy', 'rb'))
	righteye_model = ni.righteye_model
	righteye_model.to_gpu()

	img = read_image(righteye_path)
	x = np.ndarray((1, 3, insize, insize), dtype=np.float32)
	x[0]=img
	x_data = xp.asarray(x)
	x = chainer.Variable(x_data, volatile='on')

	score = predict(righteye_model,x)
	categories = np.loadtxt("labels.txt", str, delimiter="\t")
	top_k = 7
	prediction = zip(score.data[0].tolist(), categories)
	#prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

	righteye_ls = [15,15,14,14,14,14,14]	

	print('righteye------------------------------------------------')
	print(righteye_path)
	for rank, (score, name) in enumerate(prediction[:top_k], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
		righteye_ls[rank-1] = score*100		
	print('--------------------------------------------------------')

	# lefteye
	# ---------------------------------------------------------------------------
	mean_image = pickle.load(open('proposed_lefteye_mean.npy', 'rb'))
	lefteye_model = ni.lefteye_model
	lefteye_model.to_gpu()

	img = read_image(lefteye_path)
	x = np.ndarray((1, 3, insize, insize), dtype=np.float32)
	x[0]=img
	x_data = xp.asarray(x)
	x = chainer.Variable(x_data, volatile='on')

	score = predict(lefteye_model,x)
	categories = np.loadtxt("labels.txt", str, delimiter="\t")
	top_k = 7
	prediction = zip(score.data[0].tolist(), categories)
	#prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

	lefteye_ls = [15,15,14,14,14,14,14]	

	print('lefteye-------------------------------------------------')
	print(lefteye_path)
	for rank, (score, name) in enumerate(prediction[:top_k], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
		lefteye_ls[rank-1] = score*100		
	print('--------------------------------------------------------')

	# mouth
	# ---------------------------------------------------------------------------
	mean_image = pickle.load(open('proposed_mouth_mean.npy', 'rb'))
	mouth_model = ni.mouth_model
	mouth_model.to_gpu()

	img = read_image(mouth_path)
	x = np.ndarray((1, 3, insize, insize), dtype=np.float32)
	x[0]=img
	x_data = xp.asarray(x)
	x = chainer.Variable(x_data, volatile='on')

	score = predict(mouth_model,x)
	categories = np.loadtxt("labels.txt", str, delimiter="\t")
	top_k = 7
	prediction = zip(score.data[0].tolist(), categories)
	#prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

	mouth_ls = [15,15,14,14,14,14,14]	

	print('mouth---------------------------------------------------')
	print(mouth_path)
	for rank, (score, name) in enumerate(prediction[:top_k], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
		mouth_ls[rank-1] = score*100		
	print('--------------------------------------------------------')

	# ---------------------------------------------------------------------------

	all_elapsed_time = time.time() - all_start
	print(all_elapsed_time)

	expression = ['anger','disgust','fear','happiness','neutral','sadness','surprise']
	expression_total_score = [0,0,0,0,0,0,0]
		
	max_ = 0
	max_key = 0
	for var in range(0,7):
		expression_total_score[var] = int(face_ls[var]) + int(righteye_ls[var]) + int(lefteye_ls[var]) + int(mouth_ls[var])
		
		if(max_ < expression_total_score[var]):
			max_ = expression_total_score[var]
			max_key = var		

	print(max_)
	print(expression[max_key])

	return expression[max_key]
