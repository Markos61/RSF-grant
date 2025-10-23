# -*- coding: cp1251 -*-
##
import re
import sys
sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from find_files import *


def get_previous_part(filename: str, path_to_files1: str) -> str:
    """
    ���������� ��� ����� � ���������� ������.
    ���� ����� ����� = 1, �� ���������� None.
    :param path_to_files1: ���� � ������.
    :param filename: ���� � ��������������� �����;
    :return ������, ���������� ���� � �����-������
    """
    filename = filename.replace(path_to_files1, '')
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


def extract_speaker(text: str, path: str, path_to_all_docs: str, path_to_files1: str, find_in_previous_doc=False):
    """
    ��������� ��� � ������� '������� �. �.' �� ������.
    :param path_to_files1: - ������ � ���� � ������.
    :param text - ������ � ������� ���������;
    :param path - ������ ���� � ���������;
    :param path_to_all_docs - ���� � ���������� � �����������-�������� (����� �����������);
    :param find_in_previous_doc - �������� ��� ��������� ������ ������� � ����������-�������;
    :return ������ � ��� �������
    """
    # ��������� ����:
    # - ����� � ��������� ����� (�������) # - ������ # - �. �. (�������� � �������)
    # ����� ��� �����
    if '��������������� �����������' in path:
        pass
    # ����� ��� ����������
    if find_in_previous_doc:  # ����� ��� ������ ������� � ���������� ���������
        speaker_is_founded = False
        prom_path = path  # ���� ��� �������� ���������� ���������� � �����
        while not speaker_is_founded:
            previous_doc = get_previous_part(prom_path, path_to_files1)
            if previous_doc is None:
                return ''
            else:
                # print(previous_doc)
                old_files = files_in_directory(path_to_all_docs, previous_doc[1:])
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
