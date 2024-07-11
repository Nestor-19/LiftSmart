import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter

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
    




