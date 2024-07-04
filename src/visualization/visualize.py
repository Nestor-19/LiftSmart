import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

df = pd.read_pickle('../../data/interim/processed_data.pkl')

# Plot a graph for each exercise
for exercise in df['exercise'].unique():
    subset = df[df['exercise'] == exercise]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True), label=exercise)
    plt.legend()
    plt.show()

