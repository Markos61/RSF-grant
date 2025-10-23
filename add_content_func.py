# -*- coding: cp1251 -*-
##
import sys
sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from tonality_and_mod import *


def find_adj(actors: list, sent):
    """ ������� ��������� ������ �������������� ������ (��������������)
    ����� ��������;
    :param actors - ������ �������;
    :param sent - ������� ����������� """

    actor_tokens = []
    used_tokens = []
    actor_tokens_with_adj = []
    new_actors = []
    for actor in actors:
        for token in sent.tokens:  # �������� ������-������
            if token.text == actor and token.text not in used_tokens:
                actor_tokens.append(token)
                actor_tokens_with_adj.append(token)  # ��������� �������� ����� ������
                used_tokens.append(token.text)
                continue

    dop_tokens, dop_tokens1, dop_tokens2, dop_tokens3 = [], [], [], []
    for token in sent.tokens:
        for actor_token in actor_tokens:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # �������������� ��������������
                    dop_tokens.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # �������������� ��������������
                    dop_tokens1.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens1:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # �������������� ��������������
                    dop_tokens2.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens2:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # �������������� ��������������
                    dop_tokens3.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens3:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # �������������� ��������������

    actor_tokens_with_adj.sort(key=lambda t: tuple(map(int, t.id.split('_'))))
    for act_token in actor_tokens_with_adj:
        new_actors.append(act_token.text)
    return new_actors


def add_context(actors, actions, sent, tokenizer, model):
    STOP_token_texts = [',', '.']
    # ���������� ���������
    # sent.tokens - ��� ������ �����������, �� ����� ����������
    objects, action_descriptions, modality, tonality = [], [], [], []
    objects_tokens = []
    actors_with_adj = []
    # ��������� �����������
    if not actions:
        tonality = analyze_tonality(sent.text, tokenizer, model)
    else:
        actions_str = ''
        for act in actions:
            actions_str = actions_str + ' ' + act
        tonality = analyze_tonality(sent.text, tokenizer, model)  # actions_str

    # ��������� �����������
    modality = analyze_modality(sent.text)

    used_tokens, action_tokens = [], []

    underling_tokens_first_level, underling_tokens_second_level = [], []
    underling_tokens_third_level = []

    for token in sent.tokens:  # �������� ������ ���������
        for action in actions:
            if token.text == action and token.id not in used_tokens:
                action_tokens.append(token)
                used_tokens.append(token.id)

    # �������� ��������� �� ���������� ������
    for token in sent.tokens:
        for action_token in action_tokens:
            if token.head_id == action_token.id:
                if token.text not in STOP_token_texts:
                    # ����� ��������� �� action �����, ������� ��������� ��� ������
                    if token.rel == 'obj' or token.rel == 'iobj' or token.rel == 'obl' or token.rel == 'nmod':
                        if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':
                            objects_tokens.append(token)
                        # ��� ����������� ������ ���������� ���� �������� � �.�.
                        underling_tokens_first_level.append(token)
    # ��������
    for token in sent.tokens:  # �������� ��������� �� ������� ���������� (������ �������)
        for first_level_token in underling_tokens_first_level:
            if token.head_id == first_level_token.id:
                if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':  # 'ADP' - �������, 'PRON' - ������������ �����������
                    if token.text not in STOP_token_texts:
                        objects_tokens.append(token)
                        underling_tokens_second_level.append(token)

    for token in sent.tokens:  # �������� ��������� �� ������� ���������� (������ �������)
        for second_level_token in underling_tokens_second_level:
            if token.head_id == second_level_token.id:
                if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':
                    if token.text not in STOP_token_texts:
                        objects_tokens.append(token)
                        underling_tokens_third_level.append(token)

    for token in sent.tokens:  # �������� ��������� �� ������� ���������� (������ �������)
        for third_level_token in underling_tokens_third_level:
            if token.head_id == third_level_token.id:
                if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':
                    if token.text not in STOP_token_texts:
                        objects_tokens.append(token)
    # ...
    objects_tokens.sort(key=lambda t: tuple(map(int, t.id.split('_'))))  # ���������� ������� �� ����� � �����������
    for obj_token in objects_tokens:
        objects.append(obj_token.text)

    # ���������� ��������������� (���. �����) � ������
    actors_with_adj = find_adj(actors, sent)

    return objects, action_descriptions, modality, tonality, actors_with_adj
