# weather_prediction.py
import subprocess

subprocess.run("pip install pandas scikit-learn flask joblib", shell=True) 

subprocess.run("mkdir ./model", shell=True) 
subprocess.run("mkdir ./content", shell=True) 
subprocess.run("mkdir ./content/dataset", shell=True) 
subprocess.run("curl -L -o ./content/dataset/weather-prediction.zip https://www.kaggle.com/api/v1/datasets/download/ananthr1/weather-prediction", shell=True) 
subprocess.run("unzip ./content/dataset/weather-prediction.zip -d ./content/dataset", shell=True) 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

#Import Dataset
file_path = './content/dataset/seattle-weather.csv'
df = pd.read_csv(file_path)

df = df.drop('date', axis=1)

X = df.drop('weather', axis=1)
y = df['weather']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 100),
    max_iter=1000,
    activation='relu',
    solver='adam',
    random_state=42
)

#Training Model
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

#Evaluation
print("Model Evaluation:")

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

conf_matrix = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_))

#Export model
joblib.dump(mlp, 'model/mlp_model.joblib')
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(label_encoder, 'model/label_encoder.joblib')