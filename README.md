# Jules CHAPON - Blunomy project

## Introduction

This repository contains my files for the Blunomy project that I realized in August 2023.

## Objectives

I was given LIDAR point cloud datasets and was tasked to generate best fit catenary model for each wire within the dataset.

## How to go into the project

There are two types of files :
- **file_construction.ipynb** : these are Jupyter Notebooks that I used to build my python files. In those Notebooks, you will find all my steps, including data visualization. Those files are more complete for since they contain graphs that may not appear in .py files.
- **file.py** : these are the Python files I created to build my catenary models. There are more efficient since they do not include all graphs.

For each type of file, you will find 3 steps :
- **preprocessing** : data cleaning and creation of new variables
- **clustering** : creation of clusters for each wire
- **fitting** : creation of the best catenary model

There is also a **path_perso.py** file that contains my personal path for the datasets.

*Be careful* : some graphs have been made with plotly and may not appear in the github interface. I recommend that you download the files and run them.

## My method

### Preprocessing

As the datasets were already good, I had almost no data cleaning to do. I just reindexed the dataframes. Then, I did some data visualization to see how the points were located and how to tackle clustering.
I decided to look at the problem from the above, only looking at the x and y coordinates. I saw that by rotating the axes, it would be easier to cluster the points by wire since they would be more widely separated.
Hence, I created new variables *x_rotated* and *y_rotated* to make clustering easier.
Graphs can be seen in **preprocessing_construction.ipynb**.

### Clustering

Thanks to my new coordinates, it is now easy to create clusters for each wire, only by using the new *x_rotated* coordinate and the k-means method.
The number of clusters for each file could be guessed by looking at the graphs from **preprocessing**. We could have used an elbow method but the different clusters are really distinguishable when looking at the graphs.
Graphs representing the different clusters can be seen in **clustering_construction.ipynb**.

### Fitting

Now, for each cluster within each dataset, I had to find the best catenary model that fits the cloud points.
First, I decided to run a regression of *y* on *x* to find the best plane, for *z* coordinates are all in the same vertical plane within the same cluster.
Hence, I chose the best model between a linear and a quadratic regression regarding RMSE.
Then, given the catenary equation, I could fit it to the *z* coordinates to find the best parameter *c* for each wire.
Finally, I represented my results with 3D graphs.
Final graphs are present in **fitting.py**. To find all other graphs, you can use **fitting_construction.py**.

## Areas for improvement

There are some areas for improvement :
- to create rotated coordinates, I tried many configurations by hand and chose the best fit. Nevertheless, it would be better to make this automatic with a function.
- I did not take the width of the wire into account. To do so, I think that it would be possible to focus on very close points and to take the median of the *z* coordinates.
- Fits are not perfect for they use the lowest point, but the lowest point is likely to be from the extremity of the wire, and not from the inside, so it creates a small mismatch.
- The management of the project and the files could have been better by creating different folders, but I forgot to to do it in the beginning and ran out of time to correct it.
