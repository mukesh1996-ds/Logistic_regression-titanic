import pandas as pd
import numpy as np

def load_data_csv(data):
    data = pd.read_csv(data)
    return data

# Testing 

df = load_data_csv('G:\\Kaggle_compitation\\Logistic Regression\\Dataset\\train.csv')
print(df.head())