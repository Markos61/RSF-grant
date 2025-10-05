##
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
import glob
import os
from keras.callbacks import EarlyStopping
from keras.src.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from tqdm import tqdm
from SciRusTiny import get_embedding  # Испорт функции для эмбеддингов
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import to_categorical
from keras.models import *
from scikeras.wrappers import KerasRegressor

data = pd.read_csv(r"E:\Обучение\Датасет_бинарный.csv", encoding='cp1251')
class_dir = []
class_osob = []
class_soc = []
class_eco = []
class_neolib = []

class_dir = np.array(data['class_dir'].astype(int).tolist())
class_osob = np.array(data['class_osob'].astype(int).tolist())
class_soc = np.array(data['class_soc'].astype(int).tolist())
class_eco = np.array(data['class_eco'].astype(int).tolist())
class_neolib = np.array(data['class_neolib'].astype(int).tolist())


def process_embedding_string(s):
    s = s.strip('[]')

    try:
        return [float(x) for x in s.split()]
    except ValueError as e:
        print(f"Ошибка в строке: {e}")
        return []


data['embedding'] = data['embedding'].apply(process_embedding_string)
# print(data)

binary_models = {}  # Словарь для моделей первого слоя
train_features_dict = {}
test_features_dict = {}

embeddings_binary_np = np.array(data['embedding'].tolist())

labels_binary = {
    'Дирижизм': np.array(data['class_dir'].astype(int).tolist()),
    'Особый путь': np.array(data['class_osob'].astype(int).tolist()),
    'Социализм': np.array(data['class_soc'].astype(int).tolist()),
    'Экологизм': np.array(data['class_eco'].astype(int).tolist()),
    'Неолиберализм': np.array(data['class_neolib'].astype(int).tolist())
}

input_shape = (312, 1)
for label_name, y_np in labels_binary.items():
    X_train, X_test, y_train, y_test = train_test_split(embeddings_binary_np, y_np,
                                                        test_size=0.2, random_state=42, shuffle=False)
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    X_train_reshaped = np.reshape(X_train, (-1, 1, X_train.shape[1]))
    X_test_reshaped = np.reshape(X_test, (-1, 1, X_test.shape[1]))
    model_binary = Sequential([
        Bidirectional(LSTM(64, input_shape=input_shape, return_sequences=True)),
        Flatten(),
        Dense(32),
        Dense(2, activation='softmax')
    ])
    model_binary.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001),
                         metrics=['accuracy'])  # 'categorical_accuracy' , 'Precision', 'Recall'

    model_binary.fit(X_train_reshaped, y_train_hot, epochs=30, batch_size=32, verbose=0)
    binary_models[label_name] = model_binary

    y_pred = model_binary.predict(X_test_reshaped, verbose=0)
    # print(y_test)
    # print(y_pred.argmax(axis=1))
    accuracy = accuracy_score(y_test, y_pred.argmax(axis=1))
    recall = recall_score(y_test, y_pred.argmax(axis=1))
    f1 = f1_score(y_test, y_pred.argmax(axis=1))
    precision = precision_score(y_test, y_pred.argmax(axis=1))
    print(f' {label_name}: {accuracy * 100:.2f}% верных ответов, {precision * 100:.2f}% точность, {recall * 100:.2f}% полнота, {f1 * 100:.2f}% f-1 score')
    # print('-' * 50)
    # print(classification_report(y_test, y_pred.argmax(axis=1)))
    # print('-' * 50)
    # print('Матрица ошибок: ')
    # print(confusion_matrix(y_test, y_pred.argmax(axis=1)))
