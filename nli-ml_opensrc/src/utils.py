"""
NLI ML Utilities  : Computation of Non Linear Interference using Neural Networks
Deep Learning framework : pytorch
version 1.0
@author: aneog
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split

def GetTrianTestDataFrames(csvfilepath):
    df_orig = read_csv(csvfilepath)
    df = df_orig.sample(frac=1).reset_index(drop=True)
    train, cv = train_test_split(df, test_size=0.3)
    return train, cv

def GetDF(csvfilepath):
    df_orig = read_csv(csvfilepath)
    return df_orig

