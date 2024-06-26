import pandas as pd
from glob import glob

# Read a CSV file
acc_file = pd.read_csv("../../data/raw/MotionData/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

gyr_file = pd.read_csv("../../data/raw/MotionData/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")


