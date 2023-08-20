### Version : Python 3.10.12
### coding : UTF-8

### We import packages

# To manage dataframes
import numpy as np
import pandas as pd

# To do the preprocessing
from sklearn import preprocessing

### We import our python files

import path_perso

### We import our datasets

lidar_easy = pd.read_parquet(path_perso.lidar_easy_file)
lidar_medium = pd.read_parquet(path_perso.lidar_medium_file)
lidar_hard = pd.read_parquet(path_perso.lidar_hard_file)
lidar_extrahard = pd.read_parquet(path_perso.lidar_extrahard_file)


### We reorder our datasets

list_difficulty = ["easy", "medium", "hard", "extrahard"]
list_df = [lidar_easy, lidar_medium, lidar_hard, lidar_extrahard]
dict_df = dict(zip(list_difficulty, list_df))

for df in list_df:
    df.reset_index(drop=False, inplace=True)
    df.sort_values(["index"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)


### We rotate our coordonates
# We rotate our x and y axes
# So our x and y coordonates are in a new axis
# And they will be easier to cluster


# We define the angle of rotation
angle_degrees = -28.65
angle = np.radians(angle_degrees)
# Matrix of rotation
rotation_matrix = np.array(
    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
)
# We create a new dataframe with rotated coordonates and we concatenate the dataframes
for df in list_df:
    rotated_points = np.dot(rotation_matrix, np.array([df["x"], df["y"]]))
    rotated_df = pd.DataFrame(
        {"x_rotated": rotated_points[0], "y_rotated": rotated_points[1]}
    )
    df[["x_rotated", "y_rotated"]] = rotated_df
