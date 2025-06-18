import data_sanity_check as dsc
import pandas as pd
import numpy as np
from glob import glob
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

dsc

# Function to remove ground points, adjust, and normalize distance intensity
def preprocess_frame(df):
    df = df[df['Z'] > -0.5].copy()
    df.loc[:, 'DISTANCE_NORM'] = StandardScaler().fit_transform(df[['DISTANCE']])
    df.loc[:, 'INTENSITY_NORM'] = StandardScaler().fit_transform(df[['INTENSITY']])
    return df[['X', 'Y', 'Z', 'DISTANCE_NORM', 'INTENSITY_NORM']]

