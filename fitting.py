### Version : Python 3.10.12
### Coding : UTF-8

### We import packages

# To manage dataframes
import numpy as np
import pandas as pd

# To realize graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go

# To perform linear and polynomial regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

quadratic_transformation = PolynomialFeatures(2, include_bias=False).fit_transform
import sklearn.metrics
from sklearn.model_selection import train_test_split

# To fit the catenary curve
from scipy.optimize import curve_fit
import math

### We import our python files

import clustering

### We import our dataframes

lidar_easy = clustering.lidar_easy
lidar_medium = clustering.lidar_medium
lidar_hard = clustering.lidar_hard
lidar_extrahard = clustering.lidar_extrahard

list_df = clustering.list_df
list_difficulty = clustering.list_difficulty

### We define our colors (for future graphs)

green = "#53EEBA"
blue = "#0050DB"
orange = "#FF9F85"


### We find the plane of best fit for each cluster
# To do so, we compare a linear and a quadratic regression of y on x
# We keep the best model (regarding root mean squared errors) in a list

list_model = []
i = 0
for df in list_df:
    df_models = []
    list_cluster = sorted(list(set(df["cluster"])))
    for cluster in list_cluster:
        # Data
        x = df[df["cluster"] == cluster]["x"].values.reshape(-1, 1)
        y = df[df["cluster"] == cluster]["y"].values
        # We split data between training and testing sets to compare the two models
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        # Linear model
        linear_model = LinearRegression().fit(X_train, y_train)
        y_linear_pred = linear_model.predict(X_test)
        rmse_linear = sklearn.metrics.mean_squared_error(
            y_test, y_linear_pred, squared=False
        )
        # Quadratic model
        quadratic_model = LinearRegression().fit(
            quadratic_transformation(X_train), y_train
        )
        y_quadratic_pred = quadratic_model.predict(quadratic_transformation(X_test))
        rmse_quadratic = sklearn.metrics.mean_squared_error(
            y_test, y_quadratic_pred, squared=False
        )
        # We keep the best model regarding rmse
        if rmse_linear < rmse_quadratic:
            model = "linear"
        else:
            model = "quadratic"

        df_models.append(model)
    list_model.append(df_models)
    i += 1

list_model


### We now fit our catenary functions and represent them in 3D graphs

i = 0
for df in list_df:
    j = 0
    list_cluster = sorted(list(set(df["cluster"])))
    for cluster in list_cluster:
        # Data
        x = df[df["cluster"] == cluster]["x"].values
        y = df[df["cluster"] == cluster]["y"].values
        z = df[df["cluster"] == cluster]["z"].values
        xs = np.linspace(min(x), max(x), 500)
        # Model selection
        if list_model[i][j] == "linear":
            model = LinearRegression().fit(x.reshape(-1, 1), y)
            y_pred = model.predict(x.reshape(-1, 1))
            ys = model.predict(xs.reshape(-1, 1))
        else:
            model = LinearRegression().fit(
                quadratic_transformation(x.reshape(-1, 1)), y
            )
            y_pred = model.predict(quadratic_transformation(x.reshape(-1, 1)))
            ys = model.predict(quadratic_transformation(xs.reshape(-1, 1)))
        # We get the index of the min of the z coordinates
        index_min_z = np.argmin(z)
        y0 = y_pred[index_min_z]
        z0 = z[index_min_z]

        # We define our catenary function
        def function_catenary(y: float, c: float) -> float:
            return z0 + c * (np.cosh((y - y0) / c) - 1)

        # We fit the function to the points
        param_estim, _ = curve_fit(function_catenary, y_pred, z)
        # We fit the catenary curve with the points
        zs = function_catenary(ys, param_estim[0])
        # We show the graphs
        fig = go.Figure()
        trace_scatter = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=2, color=green),
            name="Points",
        )
        fig.add_trace(trace_scatter)
        trace_curve = go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", line=dict(color=blue, width=10), name="Fit"
        )
        fig.add_trace(trace_curve)
        fig.update_layout(
            title=f"{list_difficulty[i]} - wire {cluster}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
            ),
        )
        fig.show()

        j += 1

    i += 1


### We create a dataframe containing all wires data

i = 0
df_concat = []
for df in list_df:
    j = 0
    list_cluster = sorted(list(set(df["cluster"])))
    for cluster in list_cluster:
        # Data
        x = df[df["cluster"] == cluster]["x"].values
        y = df[df["cluster"] == cluster]["y"].values
        z = df[df["cluster"] == cluster]["z"].values
        xs = np.linspace(min(x), max(x), 500)
        # Model selection
        if list_model[i][j] == "linear":
            model = LinearRegression().fit(x.reshape(-1, 1), y)
            y_pred = model.predict(x.reshape(-1, 1))
            ys = model.predict(xs.reshape(-1, 1))
        else:
            model = LinearRegression().fit(
                quadratic_transformation(x.reshape(-1, 1)), y
            )
            y_pred = model.predict(quadratic_transformation(x.reshape(-1, 1)))
            ys = model.predict(quadratic_transformation(xs.reshape(-1, 1)))
        # We get the index of the min of the z coordinates
        index_min_z = np.argmin(z)
        y0 = y_pred[index_min_z]
        z0 = z[index_min_z]

        # We define our catenary function
        def function_catenary(y: float, c: float) -> float:
            return z0 + c * (np.cosh((y - y0) / c) - 1)

        # We fit the function to the points
        param_estim, _ = curve_fit(function_catenary, y_pred, z)
        # We fit the catenary curve with the points
        zs = function_catenary(ys, param_estim[0])
        # We save data in a dataframe
        df_fit = pd.DataFrame()
        df_fit["xs"] = xs
        df_fit["ys"] = ys
        df_fit["zs"] = zs
        df_fit["file"] = list_difficulty[i]
        df_fit["cluster"] = cluster
        df_concat.append(df_fit)
        j += 1

    i += 1

df_wires = pd.concat(df_concat, axis=0)


### We can now visualize wires for each file

for difficulty in list_difficulty:
    df = df_wires[df_wires["file"] == difficulty]
    x = df["xs"]
    y = df["ys"]
    z = df["zs"]
    color = df["cluster"]
    fig = go.Figure()
    scatter_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=4, color=color, colorscale="Viridis"),
    )
    fig.add_trace(scatter_trace)
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title=f"{difficulty} file",
    )
    fig.show()
