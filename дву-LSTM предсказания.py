##
import tensorflow as tf
import shutil
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from keras.models import *
from torch.utils.data import DataLoader, Dataset


def predictions(model1, names, embs, classes):
    for index_, emb1 in enumerate(embs):
        emb1 = tf.convert_to_tensor(emb1, dtype=tf.float32)
        art_reshaped = np.reshape(emb1, (-1, 1, emb1.shape[0]))
        art_probability = model1.predict(art_reshaped, verbose=0)
        index = art_probability.argmax(axis=1)
        certitude = art_probability[0][index][0] * 100
        if certitude >= 50:
            ideology = classes.get(index[0])
        else:
            ideology = 'Неопределено'
        full_destination = names[index_].replace('Экономические стенограммы', f'Распределение стенограмм (новая модель)/{ideology}')
        dir_to_create = os.path.dirname(full_destination)
        os.makedirs(dir_to_create, exist_ok=True)
        shutil.copy2(names[index_], full_destination)
        #
        dictionary1 = {names[index_]: [[index[0]], [ideology], [certitude]]}
        for key, value in dictionary1.items():
            try:
                # Создаем DataFrame из собранных данных
                dataframe1 = pd.DataFrame({
                    'file': key,
                    'class_index': value[0],
                    'class_name': value[1],
                    'certitude': value[2]
                })
                dataframe1.to_csv(fr'{full_destination[:-3]}csv', index=False, encoding='cp1251')
            except UnicodeEncodeError:
                continue
        #


def files_in_directory(path1, select):
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        if select in os.path.basename(root):
            for file in files1:
                # Получение полного пути к файлу
                file_path1 = os.path.join(root, file)
                if '.csv' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


target_classes = ['Неолиберализм', 'Социализм', 'Дирижизм',
                  'Особый путь', 'Экологизм']


class CustomDataset(Dataset):
    def __init__(self, file_names_, embeddings_):
        self.file_names = file_names_
        self.embeddings = embeddings_

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return self.file_names[idx], self.embeddings[idx]


model = load_model(r'E:\PycharmProjects\PARsing\model_LSTM_gos_duma_GOOD.h5')
classes_dict = {0: 'Неолиберализм', 1: 'Социализм',
                2: 'Дирижизм', 3: 'Особый путь', 4: 'Экологизм'}

data = pd.read_csv(r"E:\Грант\Экономические_Стенограммы_Датасет.csv", encoding='cp1251')
file_names, embeddings = [], []
for emb in data['file']:
    file_names.append(emb)
for emb in data['embedding']:
    numbers_str = emb.strip("[]").split()
    numbers_float = [float(num.replace(',', '')) for num in numbers_str]
    numpy_array = list(numbers_float)
    embeddings.append(numpy_array)
embeddings_np = np.array(embeddings)

dataset = CustomDataset(file_names, embeddings_np)
dataloader = DataLoader(dataset, batch_size=2000, shuffle=False)
for name, emb in tqdm(dataloader):
    predictions(model, name, emb, classes_dict)
