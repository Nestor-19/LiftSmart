import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_pickle('../../data/interim/processed_data.pkl')