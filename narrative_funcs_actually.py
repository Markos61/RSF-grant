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
    ���������� ��� ����� � ���������� ������.
    ���� ����� ����� = 1, �� ���������� None.
    """
    filename = filename.replace(r'E:\�����\�������� �������\�������������', '')
    # print(filename)
    # ���� � ����� ������ "_�����N"
    name, dirty_number = filename.split('�����')
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
        return None  # ���������� ����� ���
    # print(f"{name[1:]}�����{new_number}.txt")
    return f"{name[1:]}�����{new_number}.txt"


def extract_speaker(text: str, path: str, find_in_previous_doc=False):
    """
    ��������� ��� � ������� '������� �. �.' �� ������.
    ���������� ������ ��������� ���������� (����� ���� ���������).
    """
    # ��������� ����:
    # - ����� � ��������� ����� (�������)
    # - ������
    # - �. �. (�������� � �������)
    if find_in_previous_doc:  # ����� ��� ������ ������� � ���������� ���������
        speaker_is_founded = False
        prom_path = path  # ���� ��� �������� ���������� ���������� � �����
        while not speaker_is_founded:
            previous_doc = get_previous_part(prom_path)
            prom_path = previous_doc
            if previous_doc is None:
                return ''
            else:
                # print(previous_doc)
                old_files = files_in_directory(r'E:\�����\����������� ��������� ���������', previous_doc[1:])
                # print(old_files[0])
                old_texts = download_data(old_files)
                pattern = r'[�-ߨ][�-��]+(?:\s[�-�]\.\s?[�-�]\.)'

                matches = re.findall(pattern, old_texts[0])
                # print(matches)
                if matches:
                    speaker_is_founded = True
                    return matches[len(matches) - 1]
                else:
                    if '��������������������' in old_texts[0]:
                        return '��������������������'
                    else:
                        return ''

    else:
        # ����� ��� ������ ������� � ������ �����������, ���� ���� ������� ������
        pattern = r'[�-ߨ][�-��]+(?:\s[�-�]\.\s?[�-�]\.)'

        matches = re.findall(pattern, text)
        if matches:
            return matches[0]
        else:
            if '��������������������' in text:
                return '��������������������'
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
        "���������� �������": ['?'],
        "�����������": [
            "�����", "����", "�������", "������", "�������", "������", "������", "������", '������',
            "���������", "����������", "��������", "����������", "����������",
            "������� �������������", "�������������", "�����������", "������� ���������",
            "���������� ���������", "�������", "��������������", "����������",
            "������� ���������", "������� ���������", "��������� ����������",
            "������ ����", "������ ����", "������� ���������", "���������� ���������",
            "�������� �������", "������� �������", "������� ���������", "������� �������",
            "������ �����������", "������� ���������", "������� ���������", '�����', '�����'
        ],
        "�����������": [
            "�����", "��������", "��������", "��������", "����� �����������",
            "���� ����", "�����������", "�� ���������", "���������", "�����������",
            "��������", "������ �����", "���� ��", "����� ����", "������� �����������",
            "���� �����������", "���������", "��������", "� ������������",
            "�����������", "����� �����������", "����� ����", "���� ����������� ����",
            "���������", "����������� ����������",
            "���� ���� ����", "����� �� ����", "�����"
        ],
        "�������������": [
            "����", "�������� ��", "������", "����������", "������", "�����", "��������� ��",
            "������� ��", "�� ������", "���� �� �������", "���������������", "����� ��",
            "����������", "�������������", "����������", "����������������", "������"
                                                                             "������������� �������",
            "���������� �������", "�������������", "���������� ���������",
            "����� �����", "���������� ���������", "�������", "�������� �� ������",
            "��������� �� ���������", "���������� ���������", "������� ��", "������� �� �������",
            "�������� �� �����", "�����", "��������������� ���������", "���������� �����",
            "���������� ��������� ���", "������ ��", "���� �� �������"
        ]
    }

    modality_counts = {"���������� �������": 0, "�����������": 0, "�����������": 0, "�������������": 0}
    for category, words in modality_dict.items():
        for w in words:
            if w in text.lower():
                modality_counts[category] += 1

    max_value = max(modality_counts.values())
    max_categories = [k for k, v in modality_counts.items() if v == max_value]

    if len(max_categories) == 3:
        return "�����������"

    elif len(max_categories) == 2:
        return f"{max_categories[0]}/{max_categories[1]}"
    else:
        return max_categories[0]


def files_in_directory(path1, select):
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        for file in files1:
            # ��������� ������� ���� � �����
            file_path1 = os.path.join(root, file)
            if select in file_path1:
                if '.txt' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


def download_data(x):
    data = []
    for file in tqdm(x, desc="�������� ������"):
        with open(file, encoding='cp1251') as f:
            data.append(f.read())
    return data


def find_adj(actors: list, sent):
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

    for token in sent.tokens:
        for actor_token in actor_tokens:
            if (token.head_id == actor_token.id and token.pos == "ADJ") or (
                    token.head_id == actor_token.id and token.pos == "NUM"):
                actor_tokens_with_adj.append(token)  # �������������� ��������������

    actor_tokens_with_adj.sort(key=lambda t: tuple(map(int, t.id.split('_'))))
    for act_token in actor_tokens_with_adj:
        new_actors.append(act_token.text)
    return new_actors


def add_context(actors, actions, sent):
    STOP_token_texts = [',', '.']
    # ���������� ���������
    # sent.tokens - ��� ������ �����������, �� ����� ����������
    objects, action_descriptions, modality, tonality = [], [], [], []
    objects_tokens = []
    actors_with_adj = []
    # ��������� �����������
    if not actions:
        tonality = analyze_tonality(sent.text)
    else:
        actions_str = ''
        for act in actions:
            actions_str = actions_str + ' ' + act
        tonality = analyze_tonality(sent.text)  # actions_str

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

    # ���������� ��������������� � ������
    # actors_with_adj = find_adj(actors, sent)

    actors_with_adj = actors.copy()
    return objects, action_descriptions, modality, tonality, actors_with_adj


def formalize_text(x: list, doc_paths: list):
    triads = []
    doc_count = -1
    for single_text in tqdm(x, '��������� �������'):
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

        # ������ (�����)
        current_speaker = ''
        sent_count = 0  # �������, ����� ��������� ������ ����������� ��� ������ �������

        for sent in doc.sents:
            sent_count += 1
            sent_for_analysis = sent  # ����������� ��� �������
            # ���� -- ������������ ������ ���� ��� ���������� ��������� ���� (��������)
            # ������ ��� ������ � ���� ����, ������ ����

            # ������� ��� ������ ������ ���������
            triad = {'speakers': [], 'actors': [], 'actions': [], 'objects': [],
                     'action descriptions': [], 'modality': [], 'tonality': [],
                     'sentence': [],
                     'PER': [], 'LOC': [], 'ORG': []}

            # ������ � ��������� ���������� ������ ������
            narrative_tokens = []
            for token in sent.tokens:

                # ���������� �������� (����), ��������� ����� ������
                if token.rel == "neg" or token.rel == "root" or (token.pos == 'VERB' and token.rel != 'csubj'):
                    # �� ��������� ����������� ��������� ��� �����������
                    if token.rel != 'xcomp':
                        narrative_tokens.append(token)

                elif token.rel == "parataxis" and token.pos != 'ADV':
                    # ���� ���� ����� �������, �� ����������� ��������
                    narrative_tokens.append(token)

                elif token.id == token.head_id and token.pos == 'NOUN':
                    narrative_tokens.append(token)

                else:
                    pass

            # ���� � ����������� ��� ���������� � parataxis, ���� ����������
            if not narrative_tokens:
                for token in sent.tokens:
                    if token.rel == "nsubj" or token.rel == "nsubj:pass":
                        narrative_tokens.append(token)

            # ����� ������, � ��� ���� ������ � ��������� ��������
            actors, actions = [[] for _ in range(len(narrative_tokens))], [[] for _ in range(len(narrative_tokens))]
            objects = [[] for _ in range(len(narrative_tokens))]
            action_descriptions = [[] for _ in range(len(narrative_tokens))]
            modality = [[] for _ in range(len(narrative_tokens))]
            tonality = [[] for _ in range(len(narrative_tokens))]

            # ���� �������, ���� ��� � ������ �����������, �� ��������� ���������� �����������
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
            # ���������, ����� �� �� ������������� ������ �������
            if current_speaker1 != '':
                current_speaker = current_speaker1

            speakers = [current_speaker for _ in range(len(narrative_tokens))]
            # �������� ������ ��������� ��������� ���������
            for narrative_num, n_token in enumerate(narrative_tokens):
                # ����� ����
                if n_token.rel == "neg" or n_token.rel == "root" or n_token.pos == 'VERB':
                    # ���� �1 �������-����������
                    if (n_token.rel == 'root' and n_token.pos == 'NOUN') or (
                            n_token.rel == 'root' and n_token.pos == 'PROPN'):
                        # ���� root'�� ��������� ���������������
                        actions[narrative_num].append(n_token.text)
                        noun_id = n_token.id
                        # ���� ��������� ��������
                        for p_pod in sent.tokens:
                            # ���� ��������������� ���������� (����������� - ��������)
                            if p_pod.rel == "conj" or p_pod.rel == 'nsubj':
                                if p_pod.head_id == noun_id:
                                    actors[narrative_num].append(p_pod.text)

                    else:
                        # ���� �� root ��� root'�� ��������� �� ���������������
                        actions[narrative_num].append(n_token.text)
                        glag_id = n_token.id
                        # ���� ��������� ��������
                        for p_pod in sent.tokens:
                            # ���� ��������������� ����������
                            if p_pod.rel == "nsubj" or p_pod.rel == "nsubj:pass":
                                if p_pod.head_id == glag_id:
                                    actors[narrative_num].append(p_pod.text)

                            # ���� ����������� ��������� (����� ��������)
                            if p_pod.rel == "xcomp" or p_pod.rel == 'csubj':
                                if p_pod.head_id == glag_id:
                                    actions[narrative_num].append(p_pod.text)

                            # ����� ������� � ���������� ��������
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
                    # ���� �2
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], action_descriptions[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis)

                elif n_token.rel == "nsubj" or n_token.rel == "nsubj:pass":
                    # ���� �3 ��� �������
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

            # ���������� NER
            sent_doc = Doc(sent.text)  # ����� �������� ������ ��� �����������
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


files = files_in_directory(r'E:\�����\�������� �������\�������������',
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

dataframe.to_excel(fr'E:\Downloads\������_����������_new2.xlsx', index=False)
