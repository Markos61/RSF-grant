# -*- coding: cp1251 -*-
##
import pandas as pd
from tqdm import tqdm
import sys

sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from find_files import xlsx_in_directory

# Пути и настройки
input_dir = r'E:\Грант\Формализация и схожесть\Данные'
threshold = 0.8  # 0.9

# sims = ['sim_1', 'sim_2', 'sim_3', 'sim_4', 'sim_5', 'sim_6', 'sim_7', 'sim_8',
#        'sim_9', 'sim_10', 'sim_11', 'sim_12', 'sim_13', 'sim_14', 'sim_15', 'sim_16',
#        'sim_17', 'sim_18', 'sim_19', 'sim_20', 'sim_21', 'sim_22', 'sim_23', 'sim_24',
#        'sim_25', 'sim_26', 'sim_27', 'sim_28', 'sim_29', 'sim_30',
#        'sim_31', 'sim_32', 'sim_33', 'sim_34', 'sim_35', 'sim_36', 'sim_37', 'sim_38', 'sim_39']
sims = ['sim_1', 'sim_2', 'sim_3', 'sim_4', 'sim_5', 'sim_6', 'sim_7', 'sim_8',
        'sim_9', 'sim_10', 'sim_11', 'sim_12']


# Находим файлы
files = xlsx_in_directory(input_dir, '')

filtered_dfs = []

# Перебираем все файлы
for file in tqdm(files, desc='Поиск наиболее похожих'):
    df = pd.read_excel(file, engine='openpyxl')

    # Находим столбцы sim_1 - sim_39, которые есть в текущем файле
    sim_columns = [col for col in df.columns if col.startswith('sim_') and col[4:].isdigit()]

    # Фильтруем строки, где хотя бы в одном столбце sim значение > threshold
    df_filtered = df[df[sim_columns].gt(threshold).any(axis=1)]

    filtered_dfs.append(df_filtered)

##
for sim_column in sims:
    # Объединяем все отфильтрованные строки в один DataFrame
    filtered_df = pd.concat(filtered_dfs, ignore_index=True).sort_values(
        by=sim_column, ascending=False  # можно сортировать по первому sim, если нужно
    )
    filtered_df = filtered_df[filtered_df[sim_column] > threshold]

    new_order = ["date", "speakers", "sentence", "connected sentences",
                 "path", f"{sim_column}", "actors", "actions", "objects"]
    filtered_df = filtered_df[new_order]

    # Сохраняем результат
    filtered_df.to_excel(fr'E:\Грант\Формализация и схожесть\Результаты\Сходство_{sim_column}.xlsx', index=False)
