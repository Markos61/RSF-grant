##
from statistics import mean
import numpy as np
import pandas as pd
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Activation
from tensorflow.keras.layers import Attention, Permute, Multiply, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention
from keras.layers import *
import glob
import os
from keras.callbacks import EarlyStopping
from keras.src.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score, \
    precision_score
from tqdm import tqdm
from SciRusTiny import get_embedding  # Испорт функции для эмбеддингов
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, StratifiedKFold
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import to_categorical
from keras.models import *
from scikeras.wrappers import KerasRegressor
path_name = f'Обучение ГосДума'
data = pd.read_csv(fr"E:\Грант\{path_name}\Датасет_aug_new.csv", encoding='cp1251')
emb = data['embedding']
file_names, classes, embeddings = [], [], []
for emb in data['file']:
    file_names.append(emb)
for emb in data['embedding']:
    numbers_str = emb.strip("[]").split()
    numbers_float = [float(num) for num in numbers_str]
    numpy_array = list(numbers_float)
    embeddings.append(numpy_array)
X = np.array(embeddings)
Y = np.array(data['class'].astype(int).tolist())

X, Y = shuffle(X, Y, random_state=42)

input_shape = (1, 312)
n_classes = 5
n_splits = 10
kFold = StratifiedKFold(n_splits=n_splits)
list_scores, list_precision, list_f1, list_recall = [], [], [], []
list_losses = []
count = 0

target_classes = ['Неолиберализм', 'Социализм', 'Дирижизм', 'Особый путь', 'Экологизм']
matrixes, reports = [], []

for train, test in kFold.split(X, Y):
    count += 1
    # Для LSTM
    X_train = np.reshape(X[train], (-1, 1, X[train].shape[1]))
    X_test = np.reshape(X[test], (-1, 1, X[test].shape[1]))
    # X_train = np.reshape(X[train], (-1, X[train].shape[1], 1))
    # X_test = np.reshape(X[test], (-1, X[test].shape[1], 1))
    Y_train = to_categorical(Y[train])
    y_test_one_hot = to_categorical(Y[test])

    # Входные размеры
    input_shape = (1, X_train.shape[2])  # (timesteps, features)

    # Построение модели (BiLSTM + Attention)
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(32, return_sequences=True))(inputs)
    x = GlobalAveragePooling1D()(x)
    # Полносвязные слои
    # x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model1 = load_model(r'E:\PycharmProjects\PARsing\model_LSTM.h5')
    model2 = Sequential([
        Bidirectional(LSTM(32, input_shape=input_shape,
                           return_sequences=True)),
        # BatchNormalization(),
        # Dropout(0.25),
        Bidirectional(LSTM(16, input_shape=input_shape,
                           return_sequences=True)),
        Dropout(0.1),
        Flatten(),
        # BatchNormalization(),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001),  # 0.001
                  metrics=['accuracy', 'Precision', 'Recall'])

    history = model.fit(X_train, Y_train, epochs=110, batch_size=100,
                        validation_data=(X_test, y_test_one_hot), verbose=0)
    scores = model.evaluate(X_test, y_test_one_hot, verbose=1)
    test_probabilities = model.predict(X_test, verbose=1)
    accuracy = accuracy_score(Y[test], test_probabilities.argmax(axis=1))
    recall = recall_score(Y[test], test_probabilities.argmax(axis=1), average='macro')
    f1 = f1_score(Y[test], test_probabilities.argmax(axis=1), average='macro')
    precision = precision_score(Y[test], test_probabilities.argmax(axis=1), average='macro')
    #
    matrix = confusion_matrix(Y[test], test_probabilities.argmax(axis=1))
    matrixes.append(matrix)
    report = classification_report(Y[test], test_probabilities.argmax(axis=1), target_names=target_classes)
    reports.append(report)
    #
    list_scores.append(accuracy)
    list_recall.append(recall)
    list_f1.append(f1)
    list_precision.append(precision)
    list_losses.append(scores[0])
    print('-' * 50)
    print('-' * 50)
    print(f'Кросс-валидация № {count}/{n_splits}')
    print(f'Ошибка: {round(scores[0], 2)}   % Верных ответов: {round(accuracy * 100, 2)}%')
    print(f'Точность: {round(precision * 100, 2)}%   Полнота: {round(recall * 100, 2)}%   f1: {round(f1 * 100, 2)}%')
    print('-' * 50)
    print(f'Средняя ошибка: {round(mean(list_losses), 2)}   Средний % Верных ответов: {round(mean(list_scores) * 100, 2)}%')
    print(f'Средняя точность: {round(mean(list_precision) * 100, 2)}%   Средняя полнота: {round(mean(list_recall) * 100, 2)}%   Средняя f1: {round(mean(list_f1) * 100, 2)}%')
##
import seaborn as sns
import matplotlib.pyplot as plt
global_matrix = sum(matrixes)
print(global_matrix)

cm_df = pd.DataFrame(global_matrix, range(5), range(5))
column_sums = cm_df.sum(axis=0)  # axis=1
cm_df_percent = cm_df.div(column_sums, axis=1)  # Считаем проценты по горизонтали axis=0
color = sns.color_palette("coolwarm", as_cmap=True)
cmap_new = sns.light_palette("blue", as_cmap=True)  # gray

# Визуализация матрицы
plt.figure(figsize=(8, 8), dpi=130)
sns.heatmap(cm_df_percent, annot=True, fmt='.2%', vmin=0, cmap=cmap_new, vmax=1, center=0.5, cbar=True,
            xticklabels=target_classes, yticklabels=target_classes)
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
# plt.title('Матрица ошибок (Точность, precision)')
plt.show()

##
import numpy as np

cm = np.array([
    [47, 29, 6, 4, 8],
    [10, 72, 5, 8, 4],
    [4, 23, 17, 8, 5],
    [4, 34, 6, 19, 4],
    [9, 14, 5, 5, 62]
])

# Precision и Recall по каждому классу
precision = []
recall = []

for i in range(len(cm)):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp

    precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0

    precision.append(precision_i)
    recall.append(recall_i)

print("Precision:", precision)
print("Recall:", recall)

# Средние значения
print("Macro Precision:", np.mean(precision))
print("Macro Recall:", np.mean(recall))
