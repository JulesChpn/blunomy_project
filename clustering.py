### Version : Python 3.10.12
### Coding : UTF-8

### We import packages

# To manage dataframes
import numpy as np
import pandas as pd

# To do the clustering
from sklearn.cluster import KMeans

# To realize graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


### We import our python files

import preprocessing

### We import our dataframes and lists

lidar_easy = preprocessing.lidar_easy
lidar_medium = preprocessing.lidar_medium
lidar_hard = preprocessing.lidar_hard
lidar_extrahard = preprocessing.lidar_extrahard

list_df = preprocessing.list_df
list_difficulty = preprocessing.list_difficulty


### Clustering
# We use k-means to perform the clustering
# We know that for easy, hard and extrahard files, there are only 3 clusters
# For medium file, there are 7 clusters

var = ["x_rotated"]
i = 0
# We define the number of clusters
for df in list_df:
    if i == 1:
        nb_clusters = 7
    else:
        nb_clusters = 3
    # We fit the model of k-means
    model = KMeans(n_clusters=nb_clusters, init="k-means++", n_init=10, random_state=42)
    df["cluster"] = model.fit(df[var]).labels_
    i += 1
