##
import os
import re
from tqdm import tqdm
import random
import pandas as pd
from natasha import MorphVocab
from natasha import NewsNERTagger
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

MODEL = "cointegrated/rubert-tiny-sentiment-balanced"
# MODEL = "mxlcw/rubert-tiny2-russian-financial-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def get_previous_part(filename: str) -> str:
    """
    Возвращает имя файла с предыдущей частью.
    Если номер части = 1, то возвращает None.
    """
    filename = filename.replace(r'E:\Грант\Обучение ГосДума\Неолиберализм', '')
    # print(filename)
    # Ищем в конце шаблон "_частьN"
    name, dirty_number = filename.split('часть')
    number, trash = dirty_number.split('.')
    try:
        new_number = int(number) - 1
    except ValueError:
        try:
            er = number.split(' ')
            new_number = int(er[0]) - 1
        except ValueError:
            return None

    if new_number <= 1:
        return None  # предыдущей части нет
    # print(f"{name[1:]}часть{new_number}.txt")
    return f"{name[1:]}часть{new_number}.txt"


def extract_speaker(text: str, path: str, find_in_previous_doc=False):
    """
    Извлекает ФИО в формате 'Фамилия И. О.' из текста.
    Возвращает список найденных совпадений (может быть несколько).
    """
    # Регулярка ищет:
    # - слово с заглавной буквы (Фамилия)
    # - пробел
    # - И. О. (инициалы с точками)
    if find_in_previous_doc:  # Ветка для поиска спикера в предыдущем документе
        speaker_is_founded = False
        prom_path = path  # Путь для хранения предыдущих документов в цикле
        while not speaker_is_founded:
            previous_doc = get_previous_part(prom_path)
            prom_path = previous_doc
            if previous_doc is None:
                return ''
            else:
                # print(previous_doc)
                old_files = files_in_directory(r'E:\Грант\Стенограммы структура оригиналы', previous_doc[1:])
                # print(old_files[0])
                old_texts = download_data(old_files)
                pattern = r'[А-ЯЁ][а-яё]+(?:\s[А-Я]\.\s?[А-Я]\.)'

                matches = re.findall(pattern, old_texts[0])
                # print(matches)
                if matches:
                    speaker_is_founded = True
                    return matches[len(matches) - 1]
                else:
                    if 'Председательствующий' in old_texts[0]:
                        return 'Председательствующий'
                    else:
                        return ''

    else:
        # Ветка для поиска спикера в данном предложении, если есть текущий спикер
        pattern = r'[А-ЯЁ][а-яё]+(?:\s[А-Я]\.\s?[А-Я]\.)'

        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
        else:
            if 'Председательствующий' in text:
                return 'Председательствующий'
            else:
                return ''


def analyze_tonality(text):
    global tokenizer, model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=1).numpy()[0]
    labels = ['negative', 'neutral', 'positive']
    sentiment_result = dict(zip(labels, probs.tolist()))
    predicted_sentiment = max(sentiment_result, key=sentiment_result.get)
    return predicted_sentiment


def analyze_modality(text):
    modality_dict = {
        "Постановка вопроса": ['?'],
        "Обязанность": [
            "нужно", "надо", "следует", "обязан", "обязаны", "должен", "должны", "должна", 'должно',
            "требуется", "необходимо", "надлежит", "полагается", "предписано",
            "следует рассматривать", "обязательство", "неотъемлемо", "следует учитывать",
            "необходимо учитывать", "обязано", "обязательность", "непременно",
            "обязаны соблюдать", "обязаны выполнять", "требуется выполнение",
            "должен быть", "должно быть", "следует выполнять", "необходимо выполнить",
            "надлежит сделать", "следует сделать", "обязано выполнить", "обязано сделать",
            "должно соблюдаться", "следует соблюдать", "обязаны соблюдать", 'будем', 'будет'
        ],
        "Возможность": [
            "можно", "возможно", "способен", "способны", "имеет возможность",
            "есть шанс", "допускается", "не исключено", "позволено", "разрешается",
            "вероятно", "скорее всего", "вряд ли", "может быть", "имеется возможность",
            "есть вероятность", "допустимо", "возможно", "с вероятностью",
            "возможность", "можно попробовать", "имеет шанс", "есть вероятность того",
            "разрешено", "допускается выполнение",
            "есть шанс того", "могло бы быть", "думаю"
        ],
        "Желательность": [
            "хочу", "хотелось бы", "хотели", "желательно", "мечтаю", "стоит", "следовало бы",
            "неплохо бы", "бы хорошо", "было бы здорово", "предпочтительно", "лучше бы",
            "рекомендую", "целесообразно", "оптимально", "предпочтительнее", "желали"
                                                                             "целесообразно сделать",
            "желательно сделать", "рекомендовано", "желательно учитывать",
            "лучше всего", "желательно соблюдать", "желаемо", "хотелось бы видеть",
            "следовало бы учитывать", "желательно выполнять", "полезно бы", "неплохо бы сделать",
            "хотелось бы иметь", "лучше", "предпочтительно выполнять", "желательно иметь",
            "желательно учитывать при", "стоило бы", "было бы неплохо"
        ]
    }

    modality_counts = {"Постановка вопроса": 0, "Обязанность": 0, "Возможность": 0, "Желательность": 0}
    for category, words in modality_dict.items():
        for w in words:
            if w in text.lower():
                modality_counts[category] += 1

    max_value = max(modality_counts.values())
    max_categories = [k for k, v in modality_counts.items() if v == max_value]

    if len(max_categories) == 3:
        return "Нейтральная"

    elif len(max_categories) == 2:
        return f"{max_categories[0]}/{max_categories[1]}"
    else:
        return max_categories[0]


def files_in_directory(path1, select):
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        for file in files1:
            # Получение полного пути к файлу
            file_path1 = os.path.join(root, file)
            if select in file_path1:
                if '.txt' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


def download_data(x):
    data = []
    for file in tqdm(x, desc="Загрузка данных"):
        with open(file, encoding='cp1251') as f:
            data.append(f.read())
    return data


def find_adj(actors: list, sent):
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

    for token in sent.tokens:
        for actor_token in actor_tokens:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM"):
                actor_tokens_with_adj.append(token)  # Дополнительное прилагательное

    actor_tokens_with_adj.sort(key=lambda t: tuple(map(int, t.id.split('_'))))
    for act_token in actor_tokens_with_adj:
        new_actors.append(act_token.text)
    return new_actors


def add_context(actors, actions, sent):
    STOP_token_texts = [',', '.']
    # Добавление контекста
    # sent.tokens - все токены предложения, со всеми признаками
    objects, action_descriptions, modality, tonality = [], [], [], []
    objects_tokens = []
    actors_with_adj = []
    # получение тональности
    if not actions:
        tonality = analyze_tonality(sent.text)
    else:
        actions_str = ''
        for act in actions:
            actions_str = actions_str + ' ' + act
        tonality = analyze_tonality(sent.text)  # actions_str

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

    # добавление прилагательного к актору
    # actors_with_adj = find_adj(actors, sent)

    actors_with_adj = actors.copy()
    return objects, action_descriptions, modality, tonality, actors_with_adj


def formalize_text(x: list, doc_paths: list):
    triads = []
    doc_count = -1
    for single_text in tqdm(x, 'Обработка текстов'):
        doc_count += 1
        segmenter = Segmenter()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
        syntax_parser = NewsSyntaxParser(emb)
        doc = Doc(single_text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        morph_vocab = MorphVocab()
        ner_tagger = NewsNERTagger(emb)
        doc.tag_ner(ner_tagger)

        # Спикер (актор)
        current_speaker = ''
        sent_count = 0  # Счётчик, чтобы отследить первое предложение для поиска спикера

        for sent in doc.sents:
            sent_count += 1
            sent_for_analysis = sent  # Предложение для анализа
            # Идея -- использовать данный цикл для нахождения отправных слов (индексов)
            # Списки для ключей и айди слов, текста слов

            # Словарь для записи триады нарратива
            triad = {'speakers': [], 'actors': [], 'actions': [], 'objects': [],
                     'action descriptions': [], 'modality': [], 'tonality': [],
                     'sentence': [],
                     'PER': [], 'LOC': [], 'ORG': []}

            # Список с исходными элементами поиска триады
            narrative_tokens = []
            for token in sent.tokens:

                # Добавление действия (идеи), начальная точка поиска
                if token.rel == "neg" or token.rel == "root" or (token.pos == 'VERB' and token.rel != 'csubj'):
                    # не добавляем управляемое сказуемое без подлежащего
                    if token.rel != 'xcomp':
                        narrative_tokens.append(token)

                elif token.rel == "parataxis" and token.pos != 'ADV':
                    # Если есть такой элемент, то предложение назывное
                    narrative_tokens.append(token)

                elif token.id == token.head_id and token.pos == 'NOUN':
                    narrative_tokens.append(token)

                else:
                    pass

            # Если в предложении нет сказуемого и parataxis, ищем подлежащее
            if not narrative_tokens:
                for token in sent.tokens:
                    if token.rel == "nsubj" or token.rel == "nsubj:pass":
                        narrative_tokens.append(token)

            # Конец поиска, у нас есть список с исходными токенами
            actors, actions = [[] for _ in range(len(narrative_tokens))], [[] for _ in range(len(narrative_tokens))]
            objects = [[] for _ in range(len(narrative_tokens))]
            action_descriptions = [[] for _ in range(len(narrative_tokens))]
            modality = [[] for _ in range(len(narrative_tokens))]
            tonality = [[] for _ in range(len(narrative_tokens))]

            # Ищем спикера, если нет в первом предложении, то открываем предыдущую стенограмму
            if sent_count == 1:
                current_speaker1 = extract_speaker(sent_for_analysis.text, doc_paths[doc_count])
                # print(current_speaker1)
                if current_speaker1 == '':
                    # print(current_speaker1)
                    current_speaker1 = extract_speaker(sent_for_analysis.text, doc_paths[doc_count],
                                                       find_in_previous_doc=True)
                    # print(current_speaker1)
            else:

                current_speaker1 = extract_speaker(sent_for_analysis.text, doc_paths[doc_count])
            # Проверяем, нашли ли мы действительно нового спикера
            if current_speaker1 != '':
                current_speaker = current_speaker1

            speakers = [current_speaker for _ in range(len(narrative_tokens))]
            # Собираем триаду перебором начальных элементов
            for narrative_num, n_token in enumerate(narrative_tokens):
                # Выбор пути
                if n_token.rel == "neg" or n_token.rel == "root" or n_token.pos == 'VERB':
                    # ПУТЬ №1 ГЛАГОЛА-СКАЗУЕМОГО
                    if (n_token.rel == 'root' and n_token.pos == 'NOUN') or (
                            n_token.rel == 'root' and n_token.pos == 'PROPN'):
                        # Если root'ом оказалось существительное
                        actions[narrative_num].append(n_token.text)
                        noun_id = n_token.id
                        # Ищем зависимые элементы
                        for p_pod in sent.tokens:
                            # Ищем соответствующее однородное (Государство - источник)
                            if p_pod.rel == "conj" or p_pod.rel == 'nsubj':
                                if p_pod.head_id == noun_id:
                                    actors[narrative_num].append(p_pod.text)

                    else:
                        # Если не root или root'ом оказалось не существительное
                        actions[narrative_num].append(n_token.text)
                        glag_id = n_token.id
                        # Ищем зависимые элементы
                        for p_pod in sent.tokens:
                            # Ищем соответствующие подлежащее
                            if p_pod.rel == "nsubj" or p_pod.rel == "nsubj:pass":
                                if p_pod.head_id == glag_id:
                                    actors[narrative_num].append(p_pod.text)

                            # Ищем Управляемое сказуемое (придёт РАБОТАТЬ)
                            if p_pod.rel == "xcomp" or p_pod.rel == 'csubj':
                                if p_pod.head_id == glag_id:
                                    actions[narrative_num].append(p_pod.text)

                            # Поиск частицы с негативной окраской
                            elif p_pod.rel == "advmod":
                                if p_pod.head_id == glag_id:
                                    try:
                                        if p_pod.feats['Polarity'] == "Neg":
                                            actions[narrative_num].insert(0, p_pod.text)
                                            # triad['actions'][-2:] = [" ".join(triad['actions'][-2:])]
                                    except KeyError:
                                        pass

                    objects[narrative_num], action_descriptions[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis)

                elif n_token.rel == "parataxis" and n_token.pos != 'ADV':
                    # ПУТЬ №2
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], action_descriptions[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis)

                elif n_token.rel == "nsubj" or n_token.rel == "nsubj:pass":
                    # ПУТЬ №3 БЕЗ ГЛАГОЛА
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], action_descriptions[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis)

                elif n_token.id == n_token.head_id and n_token.pos == 'NOUN':
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], action_descriptions[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis)

                else:
                    pass

            # Добавление NER
            sent_doc = Doc(sent.text)  # Новый документ только для предложения
            sent_doc.segment(segmenter)
            ner_tagger = NewsNERTagger(emb)
            sent_doc.tag_ner(ner_tagger)
            for span in sent_doc.spans:
                span.normalize(morph_vocab)
                triad[span.type].append(span.normal)

            triad['speakers'] = speakers
            triad['actors'] = actors
            triad['actions'] = actions
            triad['objects'] = objects
            triad['action descriptions'] = action_descriptions
            triad['modality'] = modality
            triad['tonality'] = tonality
            triad['sentence'] = sent.text

            triads.append(triad)

    return triads


files = files_in_directory(r'E:\Грант\Обучение ГосДума\Неолиберализм',
                           '')
texts1 = download_data(files)
results = formalize_text(texts1[20:30], files[20:30])

speakers_ind, actors_ind, actions_ind, objects_ind, action_descs, mode, ton, PER_ind, LOC_ind, ORG_ind, sent_ind = [], [], [], [], [], [], [], [], [], [], []
all_params = [speakers_ind, actors_ind, actions_ind, objects_ind, mode, ton, PER_ind, LOC_ind, ORG_ind,
              sent_ind]  # action_descs
names = ['speakers', 'actors', 'actions', 'objects', 'modality', 'tonality',
         'PER', "LOC", "ORG", 'sentence']  # 'action descriptions'

for triad in results:
    for num in range(len(triad['actors'])):
        for idx, _ in enumerate(names):
            try:
                if _ in ['PER', "LOC", "ORG"]:
                    all_params[idx].append(triad[_][0])
                elif _ == 'sentence':
                    all_params[idx].append(triad[_])
                else:
                    all_params[idx].append(triad[_][num])
            except IndexError:
                all_params[idx].append(['-'])

dataframe = pd.DataFrame({
    'speakers': all_params[0],
    'actors': all_params[1],
    'actions': all_params[2],
    'objects': all_params[3],
    # 'action descriptions': all_params[3],
    'modality': all_params[4],
    'tonality': all_params[5],
    'PER': all_params[6],
    "LOC": all_params[7],
    "ORG": all_params[8],
    'sentence': all_params[9]
})

dataframe.to_excel(fr'E:\Downloads\Триады_нарративов_new2.xlsx', index=False)
