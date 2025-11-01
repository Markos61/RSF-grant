# -*- coding: cp1251 -*-
##
import pandas as pd


def writer(res: list, name: str):
    """
    Функция записывает результат формализации текста
    :param name: Имя файла для записи
    :param res: результат работы функции formalize_text
    :return: создаёт файл Excel
    """
    # Запись в файл
    names = ['speakers', 'actors', 'actions', 'objects', 'modality', 'tonality',
             'sentence', 'path', 'date', 'connected sentences']
    all_params = [[] for _ in range(len(names))]
    for triad in res:
        for num in range(len(triad['actors'])):
            for idx, _ in enumerate(names):
                try:
                    if _ == 'sentence' or _ == 'path' or _ == 'connected sentences':
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
        'sentence': all_params[6],
        'path': all_params[7],
        'date': all_params[8],
        'connected sentences': all_params[9]

    })

    dataframe.to_excel(fr'{name}', index=False)
