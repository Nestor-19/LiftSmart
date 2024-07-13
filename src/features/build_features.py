import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis

# Modify plot settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2

df = pd.read_pickle('../../data/interim/outliers_removed_data.pkl')

predictor_columns = list(df.columns[:6])

# Manage the NaN values in dataframe by performing linear interpolation
for col in predictor_columns:
    df[col] = df[col].interpolate()
    
# Determine the duration of each set in seconds
for s in df['set'].unique():
    set_duration = df[df['set'] == s].index[-1] - df[df['set'] == s].index[0]
    df.loc[(df['set'] == s), 'duration'] = set_duration.seconds

mean_duration = df.groupby(['category'])['duration'].mean()

# Use the Butterworth lowpass filter to remove noise from the data
df_lowpass = df.copy()

sampling_freq = 1000/200
cutoff_freq = 1.2

for col in predictor_columns:
    df_lowpass = LowPassFilter().low_pass_filter(df_lowpass, col, sampling_freq, cutoff_freq, order=5)
    df_lowpass[col] = df_lowpass[col + '_lowpass']
    del df_lowpass[col + '_lowpass']
    
# Use Principal component analysis (PCA) to reduce the complexity of the data
df_pca  = df_lowpass.copy()

pc_values = PrincipalComponentAnalysis().determine_pc_explained_variance(df_pca, predictor_columns)

# Plot the variance captured against the component number to determine the optimal component number
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel('Component Number')
plt.ylabel('Explained Variance')
plt.show()

# From the plot, and using the elbow technique, we deduce that the optimal component number = 3
df_pca = PrincipalComponentAnalysis().apply_pca(df_pca, predictor_columns, 3)

# Convert the accelerometer and gyroscope data into scalars by using the sum of squares
df_squared = df_pca.copy()

acc_r = (df_squared['acc_x'] ** 2) + (df_squared['acc_y'] ** 2) + (df_squared['acc_z'] ** 2)
gyr_r = (df_squared['gyr_x'] ** 2) + (df_squared['gyr_y'] ** 2) + (df_squared['gyr_z'] ** 2)

df_squared['acc_r'] = np.sqrt(acc_r)
df_squared['gyr_r'] = np.sqrt(gyr_r)

