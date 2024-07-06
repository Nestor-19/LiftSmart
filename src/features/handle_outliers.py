import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

# Modify plot settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100

df = pd.read_pickle('../../data/interim/processed_data.pkl')

outlier_columns = list(df.columns[:6])

def find_outliers_chauvenet(dataset, col, C=2):
    dataset = dataset.copy()
    
    # Compute mean and standard deviation
    mean = dataset[col].mean()
    std_dev = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points
    deviation = abs(dataset[col] - mean) / std_dev

    # Compute the upper and lower bounds
    lower_bound = -deviation / math.sqrt(C)
    upper_bound = deviation / math.sqrt(C)
    prob = []
    outlier_mask = []

    # Traverse through all rows in the dataset
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(upper_bound.iloc[i]) - scipy.special.erf(lower_bound.iloc[i]))
        )
        # Mark the point as an outlier when the probability is below the criterion
        outlier_mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = outlier_mask
    return dataset

removed_outliers_df = df.copy()

for col in outlier_columns:
    for exercise in df['exercise'].unique():
        dataset =  find_outliers_chauvenet(df[df['exercise'] == exercise], col)
        dataset.loc[dataset[col + '_outlier'], col] = np.nan
        
        removed_outliers_df.loc[(removed_outliers_df['exercise'] == exercise), col] = dataset[col] 
        
        outlier_count = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {outlier_count} from {col} for {exercise}")
        
# Export the cleaned dataframe
removed_outliers_df.to_pickle('../../data/interim/outliers_removed_data.pkl')