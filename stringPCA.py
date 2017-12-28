#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:36:49 2017

@author: vgupta
"""


# coding: utf-8

# In[8]:

#!/usr/bin/env python
import numpy as np
import scipy as scipy

from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt





# In[2]:
X = np.array(['abe simpson','apu nahasapeemapetilon','barney gumbel', 
              'bart simpson','carl carlson','charles montgomery burns','clancy wiggum',
              'comic book guy','disco stu','dr. julius hibbert','dr. nick riveria','edna krabappel',
              'fat tony','gary chalmers','groundskeeper willie','hans moleman','homer simpson','kent brockman',
              'maggie simpson','marge simpson','martin prince','mayor quimby','milhouse van houten','moe syslak',
              'ned flanders','nelson muntz','otto mann','patty bouvier','prof. john frink','ralph wiggum',
              'reverend lovejoy','rod flanders','selma bouvier','seymour skinner','sideshow bob','snake jailbird',
              'todd flanders','waylon smithers'])
#X = np.array(['abe simpson','bart simpson', 'marge simpson','homer simpson','lisa simpson','maggie simpson'])
def nGrams(text, n):
	return map(''.join, zip(*[text[i:] for i in range(n)]))

nGrams('homer simpson', n=3)


# In[3]:

def Distance(similarity):
    pairwise_dists = squareform(pdist(similarity, 'euclidean'))
   


# In[4]:

def nGramDistMatrix(X, n):
    m = len(X)
    D = np.zeros((m,m))
    D_c = np.zeros((m,m))
    similarity = np.zeros((m,m))
    #print D.shape
    for i in range(m):
        ngi = set(nGrams(X[i], n)) #nnumber of grams
        lngi = len(ngi)
        #print"<<<<<<<>>>>>>>>"
        for j in range(i+1,m):
            ngj = set(nGrams(X[j], n))
            lngj = len(ngj)
            lintersect = len(set.intersection(ngi,ngj))        
            d = 1. - 2. * lintersect / (lngi + lngj)
            s = 0.75
            K = scipy.exp(d / s ** 2)
            D[i,j] = K
            D[j,i] = K

    #print K
    ###centering kernel###
    column_sum =np.sum(D, axis=1)
    print("<<<<<>>>>>")
    rows_sum = np.sum(D, axis=0)    
    matrix_sum = np.sum(D)    
    for i in range(m):
        for j in range(i+1,m):
            current = D[i][j] - ((2/m)*column_sum[i]) + (1/(m**2)*matrix_sum)          
            D_c[i,j] = current
            D_c[j,i] = current           
    return D_c

def kernel (X,x,n):
    ngi = set(nGrams(x, n)) #nnumber of grams
    lngi = len(ngi)
    m = len(X)
    d_ = np.zeros(m)
    print(m)
    for j in range(m):
        ngj = set(nGrams(X[j], n))
        lngj = len(ngj)
        lintersect = len(set.intersection(ngi,ngj))        
        d = 1. - 2. * lintersect / (lngi + lngj)
        s = 0.75
        K = scipy.exp(d / s ** 2)
        d_[j] = K
    return d_
               
# In[5]:

D_matrix = nGramDistMatrix(X,n =4)
w, v = LA.eigh(D_matrix)
#w is eigen values
#v is eigen vectors
vec1 =  v[:,0]
vec2 = v[:,1]
vec = np.vstack((vec1,vec2))
result = np.dot(vec,D_matrix)

print(vec.shape)

newWord = "marge simpson"
newWordKernel = kernel(X, newWord,n=4)
print("newWordKernel",newWordKernel.shape)
r = np.dot(vec,newWordKernel)
print(r.shape)
# In[6]:

x = result[0,:]
y = result[1,:]

x_new = r[0]
y_new = r[1]
print(x_new)
print(y_new)
print(X[19])
print(x[19])
print(y[19])
# In[7]:

#plt.scatter(x, y)
fig, ax = plt.subplots()
ax.scatter(x, y)
#ax.annotate(newWord, (x_new,y_new))
plt.plot(x_new,y_new,'ro') 
for i, txt in enumerate(X):
    ax.annotate(txt, (x[i],y[i]))
plt.show()


# In[ ]:




