# -*- coding: cp1251 -*-
##
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from tqdm import tqdm


# ������� ����� �� ������������ �������������
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


def get_embedding(all_texts, max_length=1500, verbose=True):
    """ ������� ��� ��������� �������� ���������� ��� ���� ������,
    ��� ��� � ������ ���������� ����������� ��������� """
    tokenizer = AutoTokenizer.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny")
    model = AutoModel.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny")
    # model = torch.compile(model)
    model.cuda()
    averages = []
    if verbose:
        for text in tqdm(all_texts, '��������� �����������'):
            # ��������� ����� �� ����� �� max_length �������
            parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            embeddings = []
            for part in parts:
                # ��������� ��������� ��� ������ �����
                embedding = get_sentence_embedding(part, '', model, tokenizer, max_length)
                embeddings.append(embedding)
            avg_emb = np.mean(embeddings, axis=0)  # ��������� ������� ��������� ��� ����� ������
            averages.append(avg_emb)
        return averages
    else:
        for text in all_texts:
            # ��������� ����� �� ����� �� max_length �������
            parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            embeddings = []
            for part in parts:
                # ��������� ��������� ��� ������ �����
                embedding = get_sentence_embedding(part, '', model, tokenizer, max_length)
                embeddings.append(embedding)
            avg_emb = np.mean(embeddings, axis=0)  # ��������� ������� ��������� ��� ����� ������
            averages.append(avg_emb)
        return averages


def get_example_narratives():
    narratives = [
        "�����������", "���������", "�������� ���������",
        "�����������", "���������", "��������� ����",
        "�����������", "������������", "���������",
        "�����������", "������������ ������������", "��������� ����������� ���������������� �������",
        "�����������", "�� �����������", "�����",
        "�����", "������������ ������������", "��������� ����",
        "������� �����������", "�����������", "������������� �������������",
        "�����������", "������������ ������������", "�����������",
        "�����������", "��������", "����� �������������",
        "�����������", "������������", "������� �������",
        "�����������", "�������", "������� �� ��������������� ����������, �������, ��������������",
        "�����������", "��������", "������� �� ���������������, ����������� � ��������������",
        "�����������", "���������", "��������� ����",
        "�����������", "�������", "���������� ��������� ������",
        "�����", "������������ ������������", "���������� ������",
        "�����������", "�� ����������� � ������������", "���������� ������",
        "�����������", "������������ �� ������ ������", "�������� ���� ������������ ������",
        "�����������", "������������ ��������", "�������� ���� ������������ ������",
        "�����������", "�������� �����������", "������� � ������",
        "�����������", "�������� �����������", "����������� ����������",
        "�����������", "�������������", "��������������� �����������",
        "�����������", "�������� �����������", "�����������",
        "�����������", "��������", "����� �������������",
        "�����������", "�������������", "��������������� �����������",
        "����������� ����", "������������ �������� ������� � ���������� ������ ������", "�������� �����",
        "�����", "����������", "������",
        "�����������", "�� ����������� � ����� ������� ������������ �������", "�����",
        "�����������", "��������� ��� �������������� ��������", "��������������� �����������",
        "�����������", "������������� ������� �� ������������ ������������", "����� � ��������������� �����������",
        "�����������", "���������", "������������ ������� ��� �������������� �����",
        "�����������", "������������", "������������� �����������",
        "�����������", "�����������", "������������� �������",
        "�����������", "�������� �����", "����������� ���������",
        "�����������", "������������", "������������� �������������",
        "�����������", "���������", "���������",
        "�����������", "��������� ��� ������ �����", "���������",
        "�����������", "������ �����������", "���������",
        "�����������", "�� ����������� � �����������", "���������",
        "�����������", "���������", "���������� ����������"
    ]
    return narratives


def make_example_tensor(example_triad: list):
    """
    ������� ��� �������� ������� ��� ����������� ������ ����������� ���������
    :param example_triad: ������ ����� [�����1, ��������1, ������1, �����2, ��������2, ������2, ...]
    :return: ������ torch ��������
    """

    example_vec = get_embedding(example_triad)
    role_vecs = []

    for idx in range(0, len(example_vec), 3):
        role_example_vec = example_vec[idx + 1] + example_vec[idx + 2] - example_vec[idx]
        role_example_vec = role_example_vec / np.linalg.norm(role_example_vec)
        role_vecs.append(role_example_vec)

    # ����������� ������ numpy-�������� � ���� torch-������
    example_tensor = torch.tensor(np.stack(role_vecs), dtype=torch.float32)
    return example_tensor


def similarity_economic_meaning(embeddings: list, sample_emb_torch: torch.tensor):
    """
    ������� ���������� �������� "�������������� ������" ����� �������
    ����������� ��������� � ���������� ���������
    :param sample_emb_torch: torch ������� ��� ��������� (����������);
    :param embeddings: ������ ��������, � ������� ������� �����, �������� � ������ ��������;
    :return: �������� �� 0 �� 1, ��� 1 �������� ��������� ��������
    """
    if len(embeddings) % 3 != 0:
        raise ValueError("���������� ��������� ������ ���� ������ 3 (�����, ��������, ������)")

    similarities = []
    for idx in range(0, len(embeddings), 3):
        # ��������� "�������" �������������
        nar_vec = embeddings[idx + 1] + embeddings[idx + 2] - embeddings[idx]  # �������� + ������ - �����
        # ����������� ������
        nar_vec = nar_vec / np.linalg.norm(nar_vec)
        # ����������� � ������
        nar_vec_torch = torch.tensor(nar_vec)
        # ������� ��������
        similarity = util.cos_sim(nar_vec_torch, sample_emb_torch)
        # similarities.append(similarity.item())
        similarities.append(similarity.squeeze().tolist())

    return similarities
