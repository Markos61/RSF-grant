import os
import random
import re


def split_text_by_sentences(text):
    # ��������� ����� �� ������, �������� �����
    sentences = re.findall(r'[^.]+(?:\.)?', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # ������� ������ ������ � �������

    return sentences, len(sentences)


def files_in_directory(path1, select):
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        if select in os.path.basename(root):
            for file in files1:
                # ��������� ������� ���� � �����
                file_path1 = os.path.join(root, file)
                if '.txt' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


def find_class1(x: str):
    if '��������' in x:
        class_label = '��������'
        class_label1 = 2
    elif '������ ����' in x:
        class_label = '������ ����'
        class_label1 = 3
    elif '���������' in x:
        class_label = '���������'
        class_label1 = 1
    elif '���������' in x:
        class_label = '���������'
        class_label1 = 4
    else:
        class_label = '�������������'
        class_label1 = 0

    return class_label, class_label1


def augment(path_to_aug: str, first_doc_text: str, all_paths: list):
    class_paths = []  # ���� � ������ � ����� ������� ��� � ������� ���������
    all_paths.remove(path_to_aug)
    first_doc_class, class_num = find_class1(path_to_aug)  # ���� ����� ������� ���������
    for path in all_paths:
        if first_doc_class in path:
            class_paths.append(path)
    second_path = random.choice(class_paths)
    with open(second_path, 'r', encoding='cp1251') as f:
        second_doc_text = f.read()
    first_doc_sentences, len1 = split_text_by_sentences(first_doc_text)
    second_doc_sentences, len2 = split_text_by_sentences(second_doc_text)
    aug_doc_sentences = first_doc_sentences[: (len1 // 2)] + second_doc_sentences[(len2 // 2):]
    aug_doc_text = ' '.join(aug_doc_sentences)

    return aug_doc_text, class_num, fr'aug-{first_doc_class}'













