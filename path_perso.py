### Version : Python 3.10.12

### We import packages

import os

### We define paths for our datasets

# Folder
folder = "C:\\Users\\jchapon\\Documents\\Blunomy\\Data science LiDAR case study"
# File names
lidar_easy_name = "lidar_cable_points_easy.parquet"
lidar_medium_name = "lidar_cable_points_medium.parquet"
lidar_hard_name = "lidar_cable_points_hard.parquet"
lidar_extrahard_name = "lidar_cable_points_extrahard.parquet"
# File paths
lidar_easy_file = os.path.join(folder, lidar_easy_name)
lidar_medium_file = os.path.join(folder, lidar_medium_name)
lidar_hard_file = os.path.join(folder, lidar_hard_name)
lidar_extrahard_file = os.path.join(folder, lidar_extrahard_name)
