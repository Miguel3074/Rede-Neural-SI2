import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

try:
    df_train = pd.read_csv('treino_sinais_vitais_com_label.csv', sep=',', skiprows=1, names=['id', 's1', 's2', 'qPA', 'pulso', 'resp', 'gravidade', 'classe'])
except FileNotFoundError:
    print("Arquivo treino_sinais_vitais_com_label.csv não encontrado.")
    exit()

features = ['qPA', 'pulso', 'resp']
target_regression = 'gravidade'
target_classification = 'classe'

X = df_train[features]
y_regression = df_train[target_regression]
y_classification = df_train[target_classification]

X_train, X_val, y_train_reg, y_val_reg, y_train_clf, y_val_clf = train_test_split(
    X, y_regression, y_classification, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_clf)
y_val_encoded = label_encoder.transform(y_val_clf)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))
y_val_onehot = onehot_encoder.transform(y_val_encoded.reshape(-1, 1))

print("Dados carregados e pré-processados.")
print("Shape de X_train_scaled:", X_train_scaled.shape)
print("Shape de y_train_reg:", y_train_reg.shape)
print("Shape de y_train_onehot:", y_train_onehot.shape)