# -*- coding: cp1251 -*-
##
import sys
sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from tonality_and_mod import *


def find_adj1(actors: list, sent):
    """
    Функция дополняет акторов прилагательными, числительными и зависимыми существительными.

    :param actors: список акторов (строк);
    :param sent: предложение с токенами (объекты с атрибутами .text, .id, .head_id, .pos)
    :return: список акторов с добавленными определениями
    """
    # Множество допустимых POS
    ALLOWED_POS = ["ADJ", "NUM", "NOUN"]

    # Ищем токены-акторы
    actor_tokens = [
        t for t in sent.tokens
        if t.text in actors
    ]

    # Сохраняем найденные токены (чтобы избежать дублей)
    visited = []
    result_tokens = []

    def expand_dependents(token):
        """Рекурсивно добавляем зависимые токены подходящих POS."""
        for child in sent.tokens:
            if child.head_id == token.id and child.pos in ALLOWED_POS and child not in visited:
                visited.append(child)
                result_tokens.append(child)
                expand_dependents(child)

    # Рекурсивно расширяем для каждого актора
    for token in actor_tokens:
        expand_dependents(token)

    # Сортируем токены по порядку появления (id вида '1_2' ? [1,2])
    result_tokens.sort(key=lambda t: tuple(map(int, t.id.split('_'))))

    # Возвращаем список текстов
    return [t.text for t in result_tokens]


def find_adj(actors: list, sent):
    """ Функция дополняет актора дополнительным словом (прилагательным)
    нужно дописать;
    :param actors - список акторов;
    :param sent - текущее предложение """

    actor_tokens = []
    used_tokens = []
    actor_tokens_with_adj = []
    new_actors = []
    for actor in actors:
        for token in sent.tokens:  # Выявляем токены-акторы
            if token.text == actor and token.text not in used_tokens:
                actor_tokens.append(token)
                actor_tokens_with_adj.append(token)  # Добавляем исходный токен актора
                used_tokens.append(token.text)
                continue

    dop_tokens, dop_tokens1, dop_tokens2, dop_tokens3 = [], [], [], []
    for token in sent.tokens:
        for actor_token in actor_tokens:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # Дополнительное прилагательное
                    dop_tokens.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # Дополнительное прилагательное
                    dop_tokens1.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens1:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # Дополнительное прилагательное
                    dop_tokens2.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens2:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # Дополнительное прилагательное
                    dop_tokens3.append(token)

    for token in sent.tokens:
        for actor_token in dop_tokens3:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM") or (
                    token.head_id == actor_token.id and token.pos == "NOUN"):
                if token not in actor_tokens_with_adj:
                    actor_tokens_with_adj.append(token)  # Дополнительное прилагательное

    actor_tokens_with_adj.sort(key=lambda t: tuple(map(int, t.id.split('_'))))
    for act_token in actor_tokens_with_adj:
        new_actors.append(act_token.text)
    return new_actors


def add_context(actors, actions, sent, tokenizer, model):
    STOP_token_texts = [',', '.']
    # Добавление контекста
    # sent.tokens - все токены предложения, со всеми признаками
    objects, modality, tonality = [], [], []
    objects_tokens = []
    # получение тональности
    tonality = analyze_tonality(sent.text, tokenizer, model)
    # получение модальности
    modality = analyze_modality(sent.text)

    used_tokens, action_tokens = [], []

    underling_tokens_first_level, underling_tokens_second_level = [], []
    underling_tokens_third_level = []

    for token in sent.tokens:  # Выявляем токены сказуемые
        for action in actions:
            if token.text == action and token.id not in used_tokens:
                action_tokens.append(token)
                used_tokens.append(token.id)

    # Выявляем зависимые от сказуемого токены
    for token in sent.tokens:
        for action_token in action_tokens:
            if token.head_id == action_token.id:
                if token.text not in STOP_token_texts:
                    # Нашли зависимый от action токен, перебор вариантов для записи
                    if token.rel == 'obj' or token.rel == 'iobj' or token.rel == 'obl' or token.rel == 'nmod':
                        if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':
                            objects_tokens.append(token)
                        # Для дальнейшего поиска записываем даже предлоги и т.д.
                        underling_tokens_first_level.append(token)
    # Уточняем
    for token in sent.tokens:  # Выявляем зависимые от токенов сказуемого (первый уровень)
        for first_level_token in underling_tokens_first_level:
            if token.head_id == first_level_token.id:
                if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':  # 'ADP' - предлог, 'PRON' - указательное местоимение
                    if token.text not in STOP_token_texts:
                        objects_tokens.append(token)
                        underling_tokens_second_level.append(token)

    for token in sent.tokens:  # Выявляем зависимые от токенов сказуемого (второй уровень)
        for second_level_token in underling_tokens_second_level:
            if token.head_id == second_level_token.id:
                if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':
                    if token.text not in STOP_token_texts:
                        objects_tokens.append(token)
                        underling_tokens_third_level.append(token)

    for token in sent.tokens:  # Выявляем зависимые от токенов сказуемого (третий уровень)
        for third_level_token in underling_tokens_third_level:
            if token.head_id == third_level_token.id:
                if token.pos != 'ADP' and token.pos != 'PRON' and token.pos != 'VERB':
                    if token.text not in STOP_token_texts:
                        objects_tokens.append(token)
    # ...
    objects_tokens.sort(key=lambda t: tuple(map(int, t.id.split('_'))))  # Сортировка токенов по месту в предложении
    for obj_token in objects_tokens:
        objects.append(obj_token.text)

    # Добавление прилагательного (доп. слова) к актору
    actors_with_adj = find_adj(actors, sent)

    return objects, modality, tonality, actors_with_adj
