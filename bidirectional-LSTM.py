##
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, Activation
from tensorflow.keras.layers import Attention, Permute, Multiply, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import MultiHeadAttention
from statistics import mean
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
import glob
import os
from keras.callbacks import EarlyStopping
from keras.src.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, \
    f1_score
from tqdm import tqdm
from SciRusTiny import get_embedding  # Испорт функции для эмбеддингов
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.utils import to_categorical
from keras.models import *
import shutil
from scikeras.wrappers import KerasRegressor
from SciRusTiny import get_embedding  # Испорт функции для эмбеддингов
# "E:\Грант\Обучение\Датасет_мульти.csv"
# E:\Грант\Обучение ГосДума\Датасет_aug_new.csv
data = pd.read_csv(r"E:\Грант\Обучение ГосДума\Датасет_aug_new.csv", encoding='cp1251')
file_names, classes, embeddings = [], [], []
for emb in data['file']:
    file_names.append(emb)
for emb in data['embedding']:
    numbers_str = emb.strip("[]").split()
    numbers_float = [float(num) for num in numbers_str]
    numpy_array = list(numbers_float)
    embeddings.append(numpy_array)
embeddings_np = np.array(embeddings)
classes_np = np.array(data['class'].astype(int).tolist())

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(embeddings_np, classes_np,
                                                    test_size=0.2, shuffle=True, random_state=42)  # random_state=42,

input_shape = (312, 1)
model2 = Sequential([
    Bidirectional(LSTM(32, input_shape=input_shape, return_sequences=True)),
    Bidirectional(LSTM(16, input_shape=input_shape, return_sequences=True)),
    Dropout(0.1),
    Flatten(),
    Dense(5, activation='softmax')
])

# Входные размеры
input_shape = (1, X_train.shape[1])  # (timesteps, features)
n_classes = len(np.unique(y_train))

# ======================
# Построение модели (BiLSTM + Attention)
# ======================
inputs = Input(shape=input_shape)

x = Bidirectional(LSTM(32, return_sequences=True))(inputs)
# x = Bidirectional(LSTM(16, return_sequences=True))(x)

# Attention блок
att_out = MultiHeadAttention(num_heads=3, key_dim=32)(x, x)  # 3 и 32 лучший self-attention
gap = tf.reduce_mean(att_out, axis=1)
gmp = tf.reduce_max(att_out, axis=1)
x = Concatenate()([gap, gmp])
# x = tf.reduce_mean(att_out, axis=1)
# x = GlobalAveragePooling1D()(att_out)
# attention = Dense(256, activation='tanh')(x)  # обучаемый скоринг
# attention = Softmax(axis=1)(attention)  # нормировка по timestep'ам
# x = Multiply()([x, attention])  # взвешивание скрытых состояний
# x = Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)  # суммирование по временной оси
# x = tf.reduce_sum(x, axis=1)
# Полносвязные слои
x = Dense(32, activation='relu')(x)
# x = BatchNormalization()(x)
x = Dropout(0.1)(x)
outputs = Dense(n_classes, activation='softmax')(x)

model = Model(inputs, outputs)

model1 = load_model(r'E:\PycharmProjects\PARsing\model_LSTM.h5')  # r'E:\PycharmProjects\PARsing\model_LSTM_gos_duma_new_wave.h5'
# model = load_model(r'E:\PycharmProjects\PARsing\model_LSTM_gos_duma_new_wave.h5')
X_train_reshaped = np.reshape(X_train, (-1, 1, X_train.shape[1]))
print(X_train_reshaped.shape)
X_test_reshaped = np.reshape(X_test, (-1, 1, X_test.shape[1]))
print(X_test_reshaped.shape)
# model.add(Activation('softmax'))  Dense(64),
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
# metrics=['accuracy', 'Precision', 'Recall'])  # 'categorical_accuracy' , 'Precision', 'Recall'

# model.summary()
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
# Обучение

early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(
    X_train_reshaped,
    y_train_one_hot,
    # validation_split=0.05,
    epochs=120,
    batch_size=35)  # callbacks=[reduce_lr]

print('На тестовой выборке: ')
scores = model.evaluate(X_test_reshaped, y_test_one_hot, verbose=1)
test_probabilities = model.predict(X_test_reshaped)
accuracy = accuracy_score(y_test, test_probabilities.argmax(axis=1))  # test_predictions.argmax(axis=1)
recall = recall_score(y_test, test_probabilities.argmax(axis=1), average='macro')
f1 = f1_score(y_test, test_probabilities.argmax(axis=1), average='macro')
precision = precision_score(y_test, test_probabilities.argmax(axis=1), average='macro')
print(f' {precision * 100:.2f}% точность, {recall * 100:.2f}% полнота, {f1 * 100:.2f}% f-1 score')

print(f' {accuracy * 100:.2f}% верных ответов')
print('-' * 50)
print('Истинные метки: ')
print(y_test)
print('Предсказания: ')
print(test_probabilities.argmax(axis=1))
print("Отчёт о классификации : ")
target_classes = ['Неолиберализм', 'Социализм', 'Дирижизм', 'Особый путь', 'Экологизм']
print(classification_report(y_test, test_probabilities.argmax(axis=1), target_names=target_classes))
print('-' * 50)
print('Матрица ошибок: ')
matrix = confusion_matrix(y_test, test_probabilities.argmax(axis=1))
print(matrix)
##
model_path = r'E:\PycharmProjects\PARsing\model_LSTM_gos_duma_GOOD.h5'
model.save(f'{model_path}')
##
cm_df = pd.DataFrame(matrix, range(5), range(5))
column_sums = cm_df.sum(axis=1)  # 1 - recall, 0 - presicion
cm_df_percent = cm_df.div(column_sums, axis=0)
color = sns.color_palette("Greys", as_cmap=True)  # "gray"  coolwarm
##
# Визуализация матрицы
plt.figure(figsize=(16, 9), dpi=150)
sns.heatmap(cm_df_percent, annot=True, fmt='.2%', vmin=0, cmap=color, vmax=1, center=0.5, cbar=True,
            xticklabels=target_classes, yticklabels=target_classes)
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.title('Матрица ошибок (recall)')
plt.show()

##
model_path = r'E:\PycharmProjects\PARsing\model_LSTM_gos_duma_summer_2025_new.h5'
model = load_model(fr'{model_path}')


# Для предсказания журналов
def files_in_directory(path, select):
    all_pdf = []
    for root, dirs, files in os.walk(path):
        if select in os.path.basename(root):
            for file in files:
                # Получение полного пути к файлу
                file_path = os.path.join(root, file)
                if 'txt' in file_path and '.' in file_path:
                    all_pdf.append(file_path)
    all_pdf = random.sample(all_pdf, len(all_pdf))
    return all_pdf


def create_dirs(directory_path, journal_):
    full_path1 = os.path.join(directory_path, journal_)
    os.makedirs(full_path1, exist_ok=True)
    for ideology in target_classes:
        full_path = os.path.join(full_path1, ideology)
        os.makedirs(full_path, exist_ok=True)
    return full_path1


classes_dict = {0: 'Неолиберализм', 1: 'Социализм',
                2: 'Дирижизм', 3: 'Особый путь', 4: 'Экологизм'}
journal = 'JIS'
destination = fr'E:\Предсказания'
paths = files_in_directory(fr"E:\Перекодированные данные\{journal}", '')
journal_destination = create_dirs(destination, journal)
true_scores, predict_scores, acc = [], [], []
count = 0
certitude = [[], [], [], [], []]
for file in tqdm(paths):
    count += 1
    article = get_embedding(file)
    art = np.array([article])
    art_reshaped = np.reshape(art, (-1, 1, art.shape[1]))
    art_probability = model.predict(art_reshaped, verbose=0)
    index = art_probability.argmax(axis=1)
    certitude[index[0]].append(art_probability[0][index][0] * 100)
    #
    predict_ideology = classes_dict.get(index[0])
    full_destination = os.path.join(journal_destination, predict_ideology)
    shutil.copy(file, full_destination)
    #
    if 'Социализм' in file:
        true_scores.append(1)
    elif 'Дирижизм' in file:
        true_scores.append(2)
    elif 'Особый путь' in file:
        true_scores.append(3)
    elif 'Экологизм' in file:
        true_scores.append(4)
    else:
        true_scores.append(0)
    predict_scores.append(index[0])

for ind, value in enumerate(true_scores):
    if value == predict_scores[ind]:
        acc.append(1)
mean_accuracy = (len(acc) / len(true_scores)) * 100
# print(f'Средняя точность: {round(mean_accuracy, 2)}%')
print(f'Средняя уверенность, класс 0: {round(mean(certitude[0]), 1)}%')
print(f'Средняя уверенность, класс 1: {round(mean(certitude[1]), 1)}%')
print(f'Средняя уверенность, класс 2: {round(mean(certitude[2]), 1)}%')
print(f'Средняя уверенность, класс 3: {round(mean(certitude[3]), 1)}%')
print(f'Средняя уверенность, класс 4: {round(mean(certitude[4]), 1)}%')
##
list_count = [[], [], [], [], []]
for i in predict_scores:
    if i == 0:
        list_count[i].append(0)
    elif i == 1:
        list_count[i].append(0)
    elif i == 2:
        list_count[i].append(0)
    elif i == 3:
        list_count[i].append(0)
    else:
        list_count[i].append(0)

freq = [len(list_count[0]), len(list_count[1]), len(list_count[2]),
        len(list_count[3]), len(list_count[4])]
percents = [len(list_count[0]) / (freq[0] + freq[1] + freq[2] + freq[3] + freq[4]),
            len(list_count[1]) / (freq[0] + freq[1] + freq[2] + freq[3] + freq[4]),
            len(list_count[2]) / (freq[0] + freq[1] + freq[2] + freq[3] + freq[4]),
            len(list_count[3]) / (freq[0] + freq[1] + freq[2] + freq[3] + freq[4]),
            len(list_count[4]) / (freq[0] + freq[1] + freq[2] + freq[3] + freq[4])]
plt.figure(figsize=(12, 10))
palette = ['black', 'r', 'orange', 'b', 'g']
ax = sns.barplot(x=target_classes, y=freq, width=0.5, palette=palette)
# for i in range(5):
#     ax.bar_label(ax.containers[i], label_type='edge', color=palette[i],
#                  rotation=360, fontsize=15, padding=0.01,
#                  fmt=lambda x: int(x) if x > 0 else '')
for i, p in enumerate(ax.patches):
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{round(percents[i] * 100, 1)}%', (x + width / 2, y + height + 0.3),
                ha='center', color=palette[i], fontsize=17,
                fontstyle='italic', fontweight='bold', fontfamily='Arial')
    ax.annotate(f'{freq[i]}', (x + width / 2, y + height * 0.5),
                ha='center', color='white', fontsize=25,
                fontstyle='italic', fontweight='bold', fontfamily='Arial')

plt.grid(axis='y', linestyle='--', alpha=0.7)
# Настройка осей
font_dict = {'family': 'Arial',
             'style': 'italic',
             'size': 16,
             'weight': 'bold'}
plt.xticks(fontsize=14, fontstyle='italic', fontweight='bold', fontfamily='Arial')
plt.yticks(fontsize=14, fontstyle='italic', fontweight='bold', fontfamily='Arial')
plt.xlabel('Классы', fontdict=font_dict)
plt.ylabel('Количество статей', fontdict=font_dict)
plt.title(f'Распределение статей журнала : {journal}', fontdict=font_dict)
# Отображение графика
# Улучшение читаемости
# Сохранение графика
# plt.savefig(f"{journal}_confidence_levels.png", bbox_inches='tight', dpi=300, pad_inches=0.1)
plt.tight_layout(pad=2.5)
plt.show()
to_save = journal_destination + '/Распределение.png'
plt.savefig(to_save, bbox_inches='tight', dpi=300, pad_inches=0.1)
##
# Уверенность классификации

import matplotlib.pyplot as plt
import seaborn as sns
import random

i = 4
# plt.figure(figsize=(16, 9), dpi=150)
palette = ['gray', 'r', 'orange', 'b', 'g']
ax = sns.displot(certitude[i], color=palette[i], kde=True, bins=15)  # height=6, aspect=2
mean_point = mean(certitude[i])
# Get the underlying axes
axes = ax.axes.flat
# Iterate over the axes (which contain the patches)
for j, ax in enumerate(axes):
    # Assuming you want to work with the first plot in case of multiple subplots
    if j == 0:
        mean_point = mean(certitude[i])

        # Now you can access the patches
        for num, p in enumerate(ax.patches):
            if num != i:
                continue
            width = p.get_width()
            height = p.get_height()
            print(height)
            x, y = p.get_xy()
            ax.axvline(mean_point, color=palette[i], linewidth=1, linestyle='--')
            ax.annotate(f'{round(mean_point, 1)}%', (mean_point, height),
                        ha='center', color=palette[num], fontsize=17,
                        fontstyle='italic', fontweight='bold', fontfamily='Arial')

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# Настройка осей
font_dict = {'family': 'Arial',
             'style': 'italic',
             'size': 16,
             'weight': 'bold'}
plt.xticks(fontsize=14, fontstyle='italic', fontweight='bold', fontfamily='Arial')
plt.yticks(fontsize=14, fontstyle='italic', fontweight='bold', fontfamily='Arial')
plt.xlabel('Вероятность, %', fontdict=font_dict)
plt.ylabel('Количество статей', fontdict=font_dict)
plt.title(f'Уверенность классификации журнала : {journal}\n {target_classes[i]}', fontdict=font_dict)
plt.tight_layout(pad=2.5)
plt.show()
##
# Создаем фигуру и оси
plt.figure(figsize=(16, 9), dpi=150)

# Создаем box plot
ax = sns.boxplot(data=certitude, palette=palette)

# Настройка осей и заголовков
plt.title(f'{journal}', fontdict=font_dict)
plt.xlabel('Классы', fontdict=font_dict)
plt.ylabel('Уверенность, %', fontdict=font_dict)
ax.set_xticklabels(target_classes)
# Добавляем границы для улучшения читаемости
plt.grid(axis='y', linestyle='--', alpha=0.7)
to_save1 = journal_destination + '/Уверенность классификации.png'
plt.savefig(to_save1, bbox_inches='tight', dpi=300, pad_inches=0.1)
# Отображаем график
plt.show()
