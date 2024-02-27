import pandas as pd
import  os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

basePath :str = os.path.dirname(__file__) + '\\Models\\'
modelName = 'housing.csv'

