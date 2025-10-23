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


triad1 = ["доля"]
triad2 = ["проект"]

v1 = get_embedding(triad1)[0]
v2 = get_embedding(triad2)[0]

v1_t = torch.tensor(v1)
v2_t = torch.tensor(v2)
similarity = util.cos_sim(v1_t, v2_t)

print(similarity.item())

