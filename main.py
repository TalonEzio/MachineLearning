import pandas as pd
import  os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

basePath :str = os.path.dirname(__file__) + '\\Models\\'
modelName = 'customers.csv'


def learn(df:pd.DataFrame,testSizePercent:int):
    # phân chia dữ liệu học
    X= df[['Gender_number', 'Age', 'Annual Income (k$)']]
    y= df['Spending Score (1-100)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSizePercent / 100, random_state = 0)

    #học
    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(mse)
    print("R-squared:", r2)
    print("Mean squared error:", mse)


    #biểu diễn dữ liệu
    plt.scatter(y_test, y_pred)
    plt.xlabel("Điểm chi tiêu thực tế")
    plt.ylabel("Điểm chi tiêu dự đoán")
    plt.title(f"Biểu đồ phân tán chi tiêu thực tế và điểm chi tiêu dự đoán tỉ lệ {testSizePercent / 100}")
    plt.plot([0, 100], [0, 100], color='red', linestyle='--')
    plt.show()



# đọc csv
df = pd.read_csv(basePath + modelName)

#chuyển đổi dữ liệu chữ sang số
df['Gender_number'] = LabelEncoder().fit_transform(df['Gender'])


for i in range(5,10):
    learn(df, i * 10)

