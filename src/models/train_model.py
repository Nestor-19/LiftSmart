import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle('../../data/interim/data_features.pkl')

df_train = df.drop(["athlete", "category", "set"], axis=1)

x = df_train.drop("exercise", axis=1)
y = df_train["exercise"]

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

basic_features = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
square_features = ['acc_r', 'gyr_r']
pca_features = ['pca_1', 'pca_2', 'pca_3']
frequency_features = [f for f in df_train.columns if (('_freq' in f) or ('_pse' in f))]
time_features = [f for f in df_train.columns if "_temp_" in f]
cluster_features = ['cluster']

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + square_features + pca_features))
feature_set_3 = list(set(feature_set_2 + time_features))
feature_set_4 = list(set(feature_set_3 + frequency_features + cluster_features))