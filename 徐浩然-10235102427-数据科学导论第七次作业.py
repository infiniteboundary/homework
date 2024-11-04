import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Step 1: 读取数据
df = pd.read_csv('fraudulent.csv')

# Step 2: 处理缺失值
missing_values = df.isnull().sum() / len(df) * 100

# 剔除缺失值超过50%的列
cols_to_drop = missing_values[missing_values > 50].index
df.drop(columns=cols_to_drop, inplace=True)

# 使用众数填充其余的缺失值
imputer = SimpleImputer(strategy='most_frequent')
df[df.columns] = imputer.fit_transform(df)

X = df.drop(columns=['y']) 
y = df['y']                 

# 将数据分为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 4: 建立决策树模型并训练
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)

# Step 5: 模型预测并计算F1分数
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"Decision Tree F1 Score: {f1:.4f}")
