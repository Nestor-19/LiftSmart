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

    

        

    
    

    



