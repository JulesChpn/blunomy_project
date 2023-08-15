### Version : Python 3.10.12

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

# We check what the datasets look like
lidar_easy.head(n=5)

### We reorder our datasets

list_difficulty = ["easy", "medium", "hard", "extrahard"]
list_df = [lidar_easy, lidar_medium, lidar_hard, lidar_extrahard]
dict_df = dict(zip(list_difficulty, list_df))

for df in list_df:
    df.reset_index(drop=False, inplace=True)
    df.sort_values(["index"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

# We check result
lidar_easy

### Descriptive stats
# We check the completeness and the coherency of our datasets

for difficulty in dict_df.keys():
    print(f"{difficulty} :\n")
    print(f"values : {dict_df[difficulty].shape}\n")
    print(f"missing values : {dict_df[difficulty].isnull().sum()}\n")
    print(f"types : {dict_df[difficulty].dtypes}\n")
    print(f"stats : {dict_df[difficulty].describe()}\n")
    print("\n")


# We can see that there is no missing value in the four datasets
# They have between 601 and 2803 data points, and all data are float, so there is nothing to change
# However, some variables have widest ranges than others
# Hence, there weight will be more important than others
# We must normalize data

### We normalize our data

scaler = preprocessing.MinMaxScaler()
for df in list_df:
    X1 = pd.DataFrame(scaler.fit_transform(df.drop(columns=["index"])))
    X1 = pd.DataFrame(
        scaler.fit_transform(df.drop(columns=["index"])),
        columns=["x_norm", "y_norm", "z_norm"],
    )
    df[["x_norm", "y_norm", "z_norm"]] = X1

# We check result
lidar_easy
