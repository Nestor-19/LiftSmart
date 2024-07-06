import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Modify plot settings
mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# Read the dataframe stored in data/interim/
df = pd.read_pickle('../../data/interim/processed_data.pkl')

# Plot a graph for accelerometer and gyroscope data for all athletes
athletes = df['athlete'].unique()
exercises = df['exercise'].unique()

for exercise in exercises:
    for athlete in athletes:
        combined_plot_df = (df.query(f"exercise == '{exercise}'")
                       .query(f"athlete == '{athlete}'")
                       .reset_index())
        
        if len(combined_plot_df) > 0:
            fig, ax  = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax = ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax = ax[1])
            
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            
            ax[0].set_title(f"Exercise: {exercise.title()}, Athlete: {athlete}", loc='left', pad=20)
            ax[1].set_xlabel("samples")
            
            plt.savefig(f"../../reports/figures/{exercise.title()} ({athlete}).png")
            plt.show()
