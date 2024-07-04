import pandas as pd
from glob import glob

files = glob("../../data/raw/MotionData/*.csv")
data_path = "../../data/raw/MotionData/"  

def read_data(files, data_path):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    
    acc_set = 1
    gyr_set = 1
    
    # Looping over, parse and build Accelerometer and Gyroscope dataframes
    for f in files:
        athlete = f.split("-")[0].replace(data_path, "")
        exercise = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
    
        temp_df = pd.read_csv(f)

        temp_df["athlete"] = athlete
        temp_df["exercise"] = exercise
        temp_df["category"] = category
    
        if 'Accelerometer' in f:
            temp_df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, temp_df])
        
        if 'Gyroscope' in f:
            temp_df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, temp_df])
            
    # Convert epoch into datetime object and set it as an index for the dataframe
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
    
    # Remove unnecessary time columns
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df


acc_df, gyr_df = read_data(files, data_path)

# Merge accelerometer and gyroscope dataframes
merged_df = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)

merged_df.columns = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "athlete", "exercise", "category", "set"]

sampling = {
    'acc_x': "mean",
    'acc_y': "mean",
    'acc_z': "mean",
    'gyr_x': "mean",
    'gyr_y': "mean",
    'gyr_z': "mean",
    'athlete': "last",
    'exercise': "last",
    'category': "last",
    'set': "last"
}

days = [g for n, g in merged_df.groupby(pd.Grouper(freq='D'))]

resampled_data = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

resampled_data['set'] = resampled_data['set'].astype('int')


