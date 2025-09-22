##
import os
from tqdm import tqdm
import pandas as pd


def files_in_directory(path1, select):
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        if select in os.path.basename(root):
            for file in files1:
                # Получение полного пути к файлу
                file_path1 = os.path.join(root, file)
                if '.csv' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


target_classes = ['Неолиберализм', 'Социализм', 'Дирижизм',
                  'Особый путь', 'Экологизм', 'Неопределено']
slovars = [{}, {}, {}, {}, {}, {}]
years = list(range(1990, 2026))
# years.append('Без года')
for idx, ideology_name in enumerate(target_classes):
    files = files_in_directory(fr'E:\Грант\Распределение стенограмм (новая модель)\{ideology_name}', '')
    for i in years:
        slovars[idx][str(i)] = 0
    for file in tqdm(files):
        for i in years:
            # a = f"\{i}\\"
            if str(i) in file:
                slovars[idx][str(i)] += 1
                break

##
df = pd.DataFrame(slovars)

df1 = df.transpose()

print(df1.sum(0))
a = 0
for i in df1.sum(0):
    a += i
print(a)

print(slovars[0])
##
df1.to_excel(fr'E:\Грант\Распределение стенограмм_новая модель.xlsx')

##
import os
from tqdm import tqdm
import pandas as pd


def files_in_directory(path1, select):
    all_txt = []
    for root, dirs, files1 in os.walk(path1):
        if select in os.path.basename(root):
            for file in files1:
                # Получение полного пути к файлу
                file_path1 = os.path.join(root, file)
                if '.txt' in file_path1:
                    all_txt.append(file_path1)
    return all_txt


tems = [5, 6, 7, 9, 10, 11, 12, 13, 14, 17, 20, 27, 30, 31, 33, 34, 36, 39, 41, 43, 45, 47]
slovar = {}
for tema in tems:
    files = files_in_directory(fr'E:\Грант\Распределение стенограмм\Неопределено\{str(tema)}', '')
    slovar[tema] = len(files)

df = pd.DataFrame([slovar])
df1 = df.transpose()

df1.to_excel(fr'E:\Грант\Распределение_по темам.xlsx')
