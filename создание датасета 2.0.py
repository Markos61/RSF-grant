##
import pandas as pd
import os
from tqdm import tqdm
from urllib3.exceptions import ReadTimeoutError
from requests.exceptions import ReadTimeout
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import random
from aug import augment


# Функции взяты из документации разработчиков
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embedding(title, abstract, model, tokenizer, max_length=None):
    # Tokenize sentences
    sentence = '</s>'.join([title, abstract])
    encoded_input = tokenizer(
        [sentence], padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(model.device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().detach().numpy()[0]


def get_embedding(all_texts, max_length=1500):
    """ Функция для получения среднего эмбеддинга для всей статьи,
    так как у модели существует ограничение контекста """
    tokenizer = AutoTokenizer.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny")
    model = AutoModel.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny")
    # model = torch.compile(model)
    model.cuda()
    averages = []
    for text in all_texts:
        # Разбиваем текст на части по max_length токенов
        parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        embeddings = []
        for part in parts:
            # Вычисляем эмбендинг для каждой части
            embedding = get_sentence_embedding(part, '', model, tokenizer, max_length)
            embeddings.append(embedding)
        avg_emb = np.mean(embeddings, axis=0)  # Вычисляем средний эмбендинг для всего текста
        averages.append(avg_emb)
    return averages


def files_in_directory(path1, select):
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        if select in os.path.basename(root):
            for file in files1:
                # Получение полного пути к файлу
                file_path1 = os.path.join(root, file)
                if '.txt' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


class CustomDataset(Dataset):
    def __init__(self, files_texts):
        self.files_texts = files_texts

    def __len__(self):
        return len(self.files_texts)

    def __getitem__(self, idx):
        return self.files_texts[idx]


def find_class(x: str):
    if 'Дирижизм' in x:
        class_label = 2
    elif 'Особый путь' in x:
        class_label = 3
    elif 'Социализм' in x:
        class_label = 1
    elif 'Экологизм' in x:
        class_label = 4
    else:
        class_label = 0

    return class_label


path_name = 'Обучение ГосДума'  # Обучение
files = files_in_directory(fr'E:\Грант\{path_name}', '')

print('  Начало векторизации текстов...')
dataset = CustomDataset(files)
dataloader = DataLoader(dataset, batch_size=500, shuffle=False)

file_names, classes, embeddings = [], [], []

for file_paths in tqdm(dataloader):
    texts = []
    for path in file_paths:
        # file_names.append(f'{path}')
        with open(path, 'r', encoding='cp1251') as f:
            text1 = f.read()
            text1 = text1.encode("cp1251", errors="replace").decode("cp1251")
            texts.append(text1)
            # Определяем метку класса
            classes.append(find_class(path))
            file_names.append(f'{path}')
            # Аугментация
            p = random.random()
            if p > -1:
                aug_text, aug_class, aug_name = augment(path, text1, files.copy())
                texts.append(aug_text)
                classes.append(aug_class)
                file_names.append(aug_name)

    try:

        batch_embeddings = get_embedding(texts)
        embeddings += batch_embeddings

    except TimeoutError:
        continue
    except ReadTimeout:
        continue
    except ReadTimeoutError:
        continue
    except UnicodeEncodeError:
        continue

# Создаем DataFrame из собранных данных
dataframe = pd.DataFrame({
    'file': file_names,
    'embedding': embeddings,
    'class': classes
})

dataframe.to_csv(fr'E:\Грант\{path_name}\Датасет_aug_new.csv', index=False, encoding='cp1251')
print('  Векторизация прошла успешно!')
