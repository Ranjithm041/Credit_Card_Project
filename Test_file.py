# -*- coding: utf-8 -*-
"""
Created on Sat May 14 10:01:13 2022

@author: USER
"""

from numpy import number
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X = enc.fit_transform([[0, 0, 3,1],[5,1,0,3],[7,2,0,3],[7,2,6,9],[7,5,4,8],[7,9,8,2],[1,0,2,2]]).toarray()
print(X)
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)
print(X_pca)
print(X_pca[0])
print(X_pca[0][1])
number = "+916385481845"