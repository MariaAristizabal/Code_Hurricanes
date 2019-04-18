#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:10:21 2019

@author: aristizabal
"""

#%%

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');

#%%

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

#%%
print(pca.components_)
print(pca.explained_variance_)
print(pca.get_covariance())

#%%
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    
    v = vector * 3 * np.sqrt(length)
    print(length,vector,v)
    draw_vector(pca.mean_, pca.mean_ + v)
    print(pca.mean_,pca.mean_+v)
plt.axis('equal');

#%%
xx = pca.components_[0,0] * np.sqrt(pca.explained_variance_[0])
yy = pca.components_[1,0] * np.sqrt(pca.explained_variance_[0])
angle_posit_xaxis = np.arctan(yy/xx)
angle_rotation = np.arctan(xx/yy)
print(np.degrees(angle_rotation))

#%% Rotate data by angle of rotation
alpha = angle_rotation
xp = np.cos(alpha)*X[:,0] - np.sin(alpha) * X[:,1]
yp = np.sin(alpha)*X[:,0] + np.cos(alpha) * X[:,1]

#%%

plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(xp,yp,alpha=0.2)
