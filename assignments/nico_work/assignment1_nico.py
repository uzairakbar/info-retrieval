# -*- coding: utf-8 -*-
"""
Information Retrieval in High Dimensional Data
Assignment 1

Created on Wed Apr 25 12:12:48 2018

@author: Nico Hertel
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#==============================================================================
#       Task 2
#==============================================================================
def angle_between(x1, x2):
    ''' Returns the angle between vectors x1 and x2'''
    x1_norm = x1/np.linalg.norm(x1)
    x2_norm = x2/np.linalg.norm(x2)

    return np.arccos(np.clip(np.dot(x1_norm, x2_norm), -1.0 1.0))


def calculate_angles(d=2, n=100):
    '''
    Creates n uniformly distributed vectors in the range [-1,1]^d
    with d=2
    Determines the minimal angle of each vector to each other and computes the
    average minimal angle
    '''
    X = np.random.rand(n,d)*2 - 1 # uniformly distributed between -1 and 1
    min_angles = np.zeros(X.shape[0])
    for i, x1 in enumerate(X):
        X_dropped = np.delete(X, i, axis=0) # remove current vector form X
        angles = np.zeros(X_dropped.shape[0])
        for j, x2 in enumerate(X_dropped):
            angles[j] = angle_between(x1, x2)
        min_angles[i] = min(angles)
    return min_angles

# create dimension-vector
P = np.array([1,2,3,4,5,6,7,8,9])
P10 = P*10
P100 = 100*P
dims = np.concatenate((P,P10,P100,[1000]))

# Create minimal angles for n=50
min_angles_50 = np.zeros(dims.shape)
n = 50
for i, dim in enumerate(dims):
    print('Dimension = ' + str(dim))
    min_angles_50[i] = np.mean(calculate_angles(d=dim, n=n))

# Create minimal angles for n=100
min_angles_100 = np.zeros(dims.shape)
n = 100
for i, dim in enumerate(dims):
    print('Dimension = ' + str(dim))
    min_angles_100[i] = np.mean(calculate_angles(d=dim, n=n))

# Create minimal angles for n=200
min_angles_200 = np.zeros(dims.shape)
n = 200
for i, dim in enumerate(dims):
    print('Dimension = ' + str(dim))
    min_angles_200[i] = np.mean(calculate_angles(d=dim, n=n))

# Plotting
plt.figure(figsize=(10,10))
plt.semilogx(dims, min_angles_50, label='Samplesize: 50')
plt.semilogx(dims, min_angles_100, label='Samplesize: 100')
plt.semilogx(dims, min_angles_200, label='Samplesize: 200')

# Formatting Plot
plt.legend()
plt.xlabel('Dimension d')
plt.ylabel('Average Minimal Angle (rad)')
plt.title('Minimal Angle with Increasing Dimensionality')