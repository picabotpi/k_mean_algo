from sklearn.datasets import make_blobs
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import copy
import math

def kmean():
    name = raw_input("Enter the file name: ")
    file_name = pd.read_csv("name")
    num_itr = raw_input("Enter Number of Iterations: ")
    X, _ = make_blobs(n_samples=f, centers=5, n_features=2, cluster_std=2.2)
    # Store data in a pandas dataframe.
    k = 3
    n = 75
    idx = np.random.randint(n, size=k)
    centers = X[idx,:]
    distances = np.zeros((n,k))#Intilize the dist
    clusters = np.zeros(n)#Intslize the clust
    old_C = np.zeros(centers.shape) #Return a new array of given shape and type, filled with zeros.
    new_C = deepcopy(centers)# Save the new centrs using deepcopy(recrsive func)
    for j in range(num_itr):
        for i in range(k):
            distances[:,i] = np.linalg.norm(X - centers[i], axis=1)
        clusters = np.argmin(distances, axis = 1) # Assign all data to closest center "smallest val"
        #Find the mean of clusters update the centers
        for i in range(k):
            new_C[i] = np.mean(X[clusters == i], axis = 0)
    plt.scatter(new_C[:, 0], new_C[:, 1])
    for i in range(k):
        plt.scatter(X[clusters == i][:, 0], X[clusters == i][:, 1])
    plt.show()


