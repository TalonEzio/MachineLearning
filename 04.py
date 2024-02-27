import pandas as pd
import  os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

basePath :str = os.path.dirname(__file__) + '\\Models\\'
modelName = 'USA_Housing.csv'

def learn(df: pd.DataFrame, testSizePercent: int, columnX, columnY):
    X = df[columnX]
    y = df[columnY]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSizePercent / 100, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    r2_score = regressor.score(X_test, y_test)

    print(f"Độ chính xác của hồi quy tuyến tính (train {testSizePercent}%) : ", r2_score * 100, '%')

if __name__ == '__main__':
    df = pd.read_csv(basePath + modelName)
    columnX = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
           'Area Population']
    columnY = ['Price']

    for i in range(3,10):
        learn(df, i * 10, columnX, columnY)