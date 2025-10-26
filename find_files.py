# -*- coding: cp1251 -*-
##
import os
from tqdm import tqdm


def xlsx_in_directory(path1, select):
    """
    :param path1: Путь к директории
    :param select:  Подстрока для получения конкретных имён файлов;
    :return: Список абсолютных путей
    """
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        for file in files1:
            # Получение полного пути к файлу
            file_path1 = os.path.join(root, file)
            if select in file_path1:
                if '.xlsx' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


def files_in_directory(path1, select):
    """
    :param path1: Путь к директории
    :param select:  Подстрока для получения конкретных имён файлов;
    :return: Список абсолютных путей
    """
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        for file in files1:
            # Получение полного пути к файлу
            file_path1 = os.path.join(root, file)
            if select in file_path1:
                if '.txt' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


def download_data(x, verbose=False):
    """
    :param x: Список путей к файлам;
    :param verbose: Параметр отображения;
    :return: Возвращает список строк (тексты стенограмм);
    """
    data = []
    if verbose:
        for file in tqdm(x, desc="Загрузка данных"):
            with open(file, encoding='cp1251') as f:
                data.append(f.read())
    else:
        for file in x:
            with open(file, encoding='cp1251') as f:
                data.append(f.read())
    return data
