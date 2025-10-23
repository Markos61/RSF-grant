# -*- coding: cp1251 -*-
##
import re
import sys
sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from find_files import *


def get_previous_part(filename: str, path_to_files1: str) -> str:
    """
    Возвращает имя файла с предыдущей частью.
    Если номер части = 1, то возвращает None.
    :param path_to_files1: Путь к файлам.
    :param filename: Путь к обрабатываемому файлу;
    :return Строка, содержащая путь к файлу-предку
    """
    filename = filename.replace(path_to_files1, '')
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


def extract_speaker(text: str, path: str, path_to_all_docs: str, path_to_files1: str, find_in_previous_doc=False):
    """
    Извлекает ФИО в формате 'Фамилия И. О.' из текста.
    :param path_to_files1: - строка с путём к файлам.
    :param text - строка с текстом документа;
    :param path - полный путь к документу;
    :param path_to_all_docs - путь к директории с документами-предками (всеми документами);
    :param find_in_previous_doc - параметр для активации поиска спикера в документах-предках;
    :return Строка с ФИО спикера
    """
    # Регулярка ищет:
    # - слово с заглавной буквы (Фамилия) # - пробел # - И. О. (инициалы с точками)
    # Ветка для речей
    if 'Непроизнесенные выступления' in path:
        pass
    # Ветка для стенограмм
    if find_in_previous_doc:  # Ветка для поиска спикера в предыдущем документе
        speaker_is_founded = False
        prom_path = path  # Путь для хранения предыдущих документов в цикле
        while not speaker_is_founded:
            previous_doc = get_previous_part(prom_path, path_to_files1)
            if previous_doc is None:
                return ''
            else:
                # print(previous_doc)
                old_files = files_in_directory(path_to_all_docs, previous_doc[1:])
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
