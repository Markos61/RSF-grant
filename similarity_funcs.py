# -*- coding: cp1251 -*-
##
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch


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


def make_example_tensor(example_triad: list):
    """
    Функция для создания тензора для образцового набора нарративных признаков
    :param example_triad: список строк [актор1, действие1, объект1, актор2, действие2, объект2, ...]
    :return: список torch тензоров
    """
    examples_vec_torch = []
    example_vec = get_embedding(example_triad)
    for idx in range(0, len(example_vec), 3):
        role_example_vec = example_vec[idx + 1] + example_vec[idx + 2] - example_vec[idx]
        role_example_vec = role_example_vec / np.linalg.norm(role_example_vec)
        example_vec_torch = torch.tensor(role_example_vec)
        examples_vec_torch.append(example_vec_torch)
    return examples_vec_torch


def similarity_economic_meaning(embeddings: list, sample_emb_torch: torch.tensor):
    """
    Функция определяет сходство "экономического смысла" между списком
    нарративных элементов и идеальным образцом
    :param sample_emb_torch: torch тензор для сравнения (образцовый);
    :param embeddings: список векторов, в котором указаны актор, действие и объект действия;
    :return: значение от 0 до 1, где 1 означает идеальное сходство
    """
    if len(embeddings) % 3 != 0:
        raise ValueError("Количество элементов должно быть кратно 3 (актор, действие, объект)")

    # Получаем вектора для списка [актор, действие, объект]
    # embs = get_embedding(narrative_triad)
    similarities = []
    for idx in range(0, len(embeddings), 3):
        # Формируем "ролевое" представление
        nar_vec = embeddings[idx + 1] + embeddings[idx + 2] - embeddings[idx]  # Действие + Объект - Актор
        # Нормализуем вектор
        nar_vec = nar_vec / np.linalg.norm(nar_vec)
        # Преобразуем в тензор
        nar_vec_torch = torch.tensor(nar_vec)
        # Считаем сходство
        similarity = util.cos_sim(nar_vec_torch, sample_emb_torch)
        similarities.append(similarity.item())
    return similarities
