import pandas as pd
from glob import glob

files = glob("../../data/raw/MotionData/*.csv")

data_path = "../../data/raw/MotionData/"    

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

    

        

    
    

    



