import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

basePath: str = os.path.dirname(__file__) + '\\Models\\'
modelName = 'Iris.csv'


def learn(df: pd.DataFrame, testSizePercent: int, columnX, columnY):
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columnX])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, df[columnY], test_size=testSizePercent / 100, random_state=0)

    # Tiêu chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tạo mô hình phân loại
    model = LogisticRegression()

    # Huấn luyện mô hình
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

# Vẽ biểu đồ phân tán
    plt.figure(figsize=(8, 6))
    plt.scatter(df[columnX[0]], df[columnX[1]], c=df[columnY])
    plt.xlabel('Chiều dài đài hoa')
    plt.ylabel('Chiều rộng đài hoa')
    plt.title(f"Iris (test_size = {testSizePercent}%)")

    plt.show()

# đọc csv
df = pd.read_csv(basePath + modelName)

# chuyển đổi dữ liệu chữ sang số
df["Species_num"] = LabelEncoder().fit_transform(df["Species"])


columnX = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
columnY = 'Species_num'

learn(df, 80,columnX,columnY)
