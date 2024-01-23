import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Sử dụng metric phù hợp cho hồi quy
import matplotlib.pyplot as plt

basePath: str = os.path.dirname(__file__) + '\\Models\\'
modelName = 'taxi-fare.csv'


def learn(df: pd.DataFrame, testSizePercent: int, columnX, columnY):

    X = df[columnX]
    Y = df[columnY]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testSizePercent / 100, random_state=0)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("R-squared:", r2)
    print("Mean Absolute Error:", mae)

    plt.figure(figsize=(8, 6))
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Giá cước thực tế")
    plt.ylabel("Giá cước dự đoán")
    plt.show()


# Đọc csv
df = pd.read_csv(basePath + modelName)

df["payment_type_num"] = LabelEncoder().fit_transform(df["payment_type"])

columnX = ["passenger_count", "trip_time_in_secs", "trip_distance", "rate_code", "payment_type_num"]
columnY = "fare_amount"

# for i in range(5,10):
#     learn(df, i  * 10, columnX, columnY)
learn(df, 8  * 10, columnX, columnY)