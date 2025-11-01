# -*- coding: cp1251 -*-

import os
import re
from datetime import datetime
from multiprocessing import Pool
from tqdm import tqdm
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
from torch.utils.data import DataLoader
import sys

sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from custom_dataloading import NarrativeDataset
from writer_func import writer
from speaker_funcs import extract_speaker
from find_files import files_in_directory
from add_content_func import add_context


def formalize_text(x: list, doc_paths: list,
                   path_to_all_docs: str, path_to_files1: str,
                   tokenizer, model):
    """
    �������, ������� ���� ����������� ��������
    :param model: ������, ��� ��������� �����������;
    :param tokenizer: ����������;
    :param path_to_files1: ���� � ������;
    :param x: ������ ����� � �������;
    :param doc_paths: ������ ������ ����� � ����������;
    :param path_to_all_docs: ������, ���������� ���� � ���������� � �����������-�������� (����� �����������)
    """
    triads = []
    doc_count = -1
    for index, single_text in enumerate(x):
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

        # ����
        date = re.search(r'(\d{1,2})\s+([�-��-�]+)\s+(\d{4})', doc_paths[index])
        date = date.group(0)

        sent_count = 0  # �������, ����� ��������� ������ ����������� ��� ������ �������
        for sent in doc.sents:
            sent_count += 1
            sent_for_analysis = sent  # ����������� ��� �������
            # ���� -- ������������ ������ ���� ��� ���������� ��������� ���� (��������)

            # ������� ��� ������ ������ ���������
            triad = {'speakers': [], 'actors': [], 'actions': [], 'objects': [],
                     'modality': [], 'tonality': [], 'sentence': [], 'path': [],
                     'date': [], 'connected sentences': []}

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
            modality = [[] for _ in range(len(narrative_tokens))]
            tonality = [[] for _ in range(len(narrative_tokens))]
            date_list = [date for _ in range(len(narrative_tokens))]

            # ���� �������, ���� ��� � ������ �����������, �� ��������� ���������� �����������
            if sent_count == 1:
                current_speaker1 = extract_speaker(sent_for_analysis.text, doc_paths[doc_count], path_to_all_docs,
                                                   path_to_files1)
                if current_speaker1 == '':
                    current_speaker1 = extract_speaker(sent_for_analysis.text, doc_paths[doc_count],
                                                       path_to_all_docs, path_to_files1, find_in_previous_doc=True)
            # ����� ������� ����� ���� ��� ������ "���������"
            else:
                if '��������������� �����������' not in doc_paths[doc_count]:
                    current_speaker1 = extract_speaker(sent_for_analysis.text, doc_paths[doc_count], path_to_all_docs,
                                                       path_to_files1)
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

                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                elif n_token.rel == "parataxis" and n_token.pos != 'ADV':
                    # ���� �2
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                elif n_token.rel == "nsubj" or n_token.rel == "nsubj:pass":
                    # ���� �3 ��� �������
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                elif n_token.id == n_token.head_id and n_token.pos == 'NOUN':
                    actors[narrative_num].append(n_token.text)
                    objects[narrative_num], modality[narrative_num], tonality[
                        narrative_num], actors[narrative_num] = add_context(actors[narrative_num],
                                                                            actions[narrative_num], sent_for_analysis,
                                                                            tokenizer, model)

                else:
                    pass

            triad['speakers'] = speakers
            triad['actors'] = actors
            triad['actions'] = actions
            triad['objects'] = objects
            triad['modality'] = modality
            triad['tonality'] = tonality
            triad['sentence'] = sent.text
            triad['path'] = doc_paths[index]
            triad['date'] = date_list

            start = max(0, sent_count - 3)
            end = min(len(doc.sents) - 1, sent_count + 3)
            con_sents = ''
            for idx, con_sent in enumerate(doc.sents[start:end]):
                con_sents = con_sents + " " + con_sent.text
            triad['connected sentences'] = con_sents

            triads.append(triad)

    return triads


def process_batch(args):
    """������� ����������� � ��������� ��������"""
    doc_texts, doc_paths, path_to_all_files, path_to_files, tokenizer, model, output_dir = args

    try:
        # �������� ������
        res = formalize_text(doc_texts, doc_paths, path_to_all_files, path_to_files, tokenizer, model)

        # �������� ����������� ����� �����
        first_name = os.path.splitext(os.path.basename(doc_paths[0]))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out_name = f"���������_{first_name}_{timestamp}.xlsx"
        out_path = os.path.join(output_dir, out_name)

        # ���������� ����������
        writer(res, out_path)

        return f"������: {out_name}"

    except Exception as e:
        return f"������ ��� ��������� {doc_paths[0]}: {e}"


#
if __name__ == "__main__":
    # === ��������� ����� ===
    path_to_files = r'E:\�����\������������� �����������'
    path_to_all_files = r'E:\�����\����������� ��������� ���������'
    output_dir = r'E:\�����\������������'
    os.makedirs(output_dir, exist_ok=True)

    # === ������������� ������� ===
    # MODEL = "cointegrated/rubert-tiny-sentiment-balanced"
    # tokenizer_1 = AutoTokenizer.from_pretrained(MODEL)
    # model_1 = AutoModelForSequenceClassification.from_pretrained(MODEL)
    MODEL, tokenizer_1, model_1 = '', '', ''

    # === �������� ������ ===
    files = files_in_directory(path_to_files, '1997')
    dataset = NarrativeDataset(files)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

    # === ���������� ���������� ��� multiprocessing ===
    tasks = [
        (doc_texts, doc_paths, path_to_all_files, path_to_files,
         tokenizer_1, model_1, output_dir)
        for doc_texts, doc_paths in dataloader
    ]

    # === ������ multiprocessing ===
    num_workers = 10
    print(f"������������ {num_workers} ���������")

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_batch, tasks), total=len(tasks), desc="������������ �������"))

    # === ����� ������ ===
    for r in results:
        print(r)

    print("��� ������ ��������� �������.")
