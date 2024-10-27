import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('bike.csv')
data = data.drop(columns=['id'])
data = data[data['city'] == 0].drop(columns=['city'])
data['hour'] = np.where((data['hour'] >= 6) & (data['hour'] <= 18), 1, 0)
y = data['y'].values
data = data.drop(columns=['y'])
X = data.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'模型的均方根误差 (RMSE): {rmse}')
