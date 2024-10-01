import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

iris = load_iris()

# print(iris.DESCR)

X = pd.DataFrame(iris.data)
Y = iris.target

X.head()

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_train_n = X_train.copy()
X_test_n = X_test.copy()

norm = MinMaxScaler().fit(X_train_n)

X_train_norm = norm.transform(X_train_n)
X_test_norm = norm.transform(X_test_n)

X_train_norm_df = pd.DataFrame(X_train_norm)

X_train_norm_df.columns = iris.feature_names
X_train_norm_df.describe()
