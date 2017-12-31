#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 10:19:31 2017

@author: vgupta
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np

import scipy as scipy

from numpy import linalg as LA

import pickle

global newWordList
newWordList = []
global allWords
TotalWordsKernel = None
allWords = None
global words
words = []
    
    
def build_dataset(words, n_words):
  """Process raw inputs into a dataset."""
  print("Building dataset")
  count = []
  count.extend(collections.Counter(words).most_common(n_words))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  for word in words:
    if word in dictionary:
      index = dictionary[word]
      data.append(index)
  
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary
a=0
if not os.path.exists("data/kernel.dat"):
    fname = 'corpuses/dbArticles/englisharticlesProcessed.txt'
    with open(fname) as f:
        for line in f:
            a=a+1
            print(a)
            data = line.split()
            for word in data:
                if (len(word)>3):
                    words.append(word)

    print("dataset size",len(words))
    vocabulary_size = 15000
    data, count, dictionary, reverse_dictionary = build_dataset(words,vocabulary_size)
    w = list(reverse_dictionary.values())
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    sizeW = len(w)
    print("vocab size",sizeW)


if not os.path.exists("data/kernel.dat"):
  with open('data/data.txt', 'wb') as fp1:
    pickle.dump(data, fp1)
  with open('data/count.txt', 'wb') as fp2:
    pickle.dump(count, fp2)
  with open('data/dictionary.txt', 'wb') as fp3:
    pickle.dump(dictionary, fp3, protocol=pickle.HIGHEST_PROTOCOL)
  with open('data/reverse_dictionary.txt', 'wb') as fp4:
    pickle.dump(reverse_dictionary, fp4, protocol=pickle.HIGHEST_PROTOCOL)
  with open('data/w.txt', 'wb') as fp:
    pickle.dump(w, fp)
  print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
else:
  with open ('data/data.txt', 'rb') as fp:
    data = pickle.load(fp)

  with open ('data/count.txt', 'rb') as fp:
    count = pickle.load(fp)

  with open ('data/dictionary.txt', 'rb') as fp:
    dictionary = pickle.load(fp)

  with open ('data/reverse_dictionary.txt', 'rb') as fp:
    reverse_dictionary = pickle.load(fp)

  with open ('data/w.txt', 'rb') as fp:
    w = pickle.load(fp)

sizeW = len(w)
print(sizeW)
def nGrams(text, n):
    return map(''.join, zip(*[text[i:] for i in range(n)]))
def Kernel(X, n):
    m = len(X)
    D = np.zeros((m,m))
    for i in range(m):
        print(i)
        ngi = set(nGrams(X[i], n)) #nnumber of grams
        lngi = len(ngi)
        for j in range(m):
            ngj = set(nGrams(X[j], n))
            lngj = len(ngj)
            lintersect = len(set.intersection(ngi,ngj))    
            d = 1. - ((2. * lintersect) / (lngi + lngj))
            s = 0.75
            #gaussian kernel
            K = np.float32(scipy.exp(d / (2*s * s)))
            D[i,j] = K
            D[j,i] = K   
    print("done with kernel")   
            
    return D
if not os.path.exists("data/kernel.dat"):
    D = Kernel(w,n =4)
    print(D.shape)
    fp = np.memmap("data/kernel.dat", dtype='float32', mode='w+', shape=(len(w),len(w)))
    fp[:] = D[:]
else:
    fp = np.memmap("data/kernel.dat", dtype='float32', mode="r", shape=(len(w),len(w)))
    D = fp
print(D.shape)

w, v = LA.eigh(D)
vec =  np.divide(v[:,0],w[0])
for dimen in range(1,300):
    vec2 = np.divide(v[:,dimen],w[dimen])
    vec = np.vstack((vec,vec2))
if not os.path.exists("data/projectionMatrix.dat"):
    fp = np.memmap("data/projectionMatrix.dat", dtype='float32', mode='w+', shape=vec.shape)
    fp[:] = vec[:]
morphvec = np.dot(vec,D)# N*V Matrix
morphvec = morphvec.transpose()# V*N Matrix
if not os.path.exists("data/morphvec.dat"):
    fp = np.memmap("data/morphvec.dat", dtype='float32', mode='w+', shape=morphvec.shape)
    fp[:] = morphvec[:]
