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
    speakers_ind, actors_ind, actions_ind, objects_ind, action_descs, mode, ton, PER_ind, LOC_ind, ORG_ind, sent_ind = [], [], [], [], [], [], [], [], [], [], []
    all_params = [speakers_ind, actors_ind, actions_ind, objects_ind, mode, ton, PER_ind, LOC_ind, ORG_ind,
                  sent_ind]  # action_descs
    names = ['speakers', 'actors', 'actions', 'objects', 'modality', 'tonality',
             'PER', "LOC", "ORG", 'sentence']  # 'action descriptions'

    for triad in res:
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
        'modality': all_params[4],
        'tonality': all_params[5],
        'PER': all_params[6],
        "LOC": all_params[7],
        "ORG": all_params[8],
        'sentence': all_params[9]
    })

    dataframe.to_excel(fr'{name}', index=False)
    print(name)
