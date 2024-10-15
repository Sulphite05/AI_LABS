import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import keras

url = "heart.csv"
df = pd.read_csv(url)

numerical_features = ['age', 'chol', 'trestbps', 'thalach', 'oldpeak']
categorical_features = ['gender', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal', 'slope']


target = 'target'

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(), ['cp', 'thal', 'restecg']),
    ('ordinal', OrdinalEncoder(), ['slope'])
], remainder='passthrough')

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])
X_train_processed = model_pipeline.fit_transform(X_train)
X_test_processed = model_pipeline.transform(X_test)

X_train_df = pd.DataFrame(X_train_processed)
X_test_df = pd.DataFrame(X_test_processed)

input_shape = X_train_processed.shape[1]

model = keras.Sequential([
    keras.layers.InputLayer(shape=(input_shape,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_processed, y_train, epochs=20, batch_size=32, validation_data=(X_test_processed, y_test))
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")
new_data = np.array([X_test_processed[0]])
prediction = model.predict(new_data)
print(f"Predicted class for the new data: {'Heart Disease' if prediction > 0.5 else 'No Heart Disease'}")