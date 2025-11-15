# -*- coding: cp1251 -*-
##
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from similarity_funcs import make_example_tensor, similarity_economic_meaning, get_embedding, get_example_narratives
from find_files import xlsx_in_directory


examples = None  # глобальная переменная для процессов


def init_worker(example_tensors):
    global examples
    examples = example_tensors


def process_file(args):
    file = args
    try:
        df = pd.read_excel(file, engine='openpyxl')
        nar_list = []

        def clean(x):
            x = x.replace('[', '').replace(']', '').replace("'", '').replace(",", '').strip()
            return "" if x == "-" else x

        for _, row in df[['actors', 'actions', 'objects']].fillna('-').replace('[]', '-').iterrows():
            # nar_list.extend([row['actors'].replace('[', '').replace(']', '').replace("'", '').replace(",", ''),
            #                 row['actions'].replace('[', '').replace(']', '').replace("'", '').replace(",", ''),
            #                 row['objects'].replace('[', '').replace(']', '').replace("'", '').replace(",", '')])
            # actor = row['actors'].replace('[', '').replace(']', '').replace("'", '').replace(",", '')
            # action = row['actions'].replace('[', '').replace(']', '').replace("'", '').replace(",", '')
            # obj = row['objects'].replace('[', '').replace(']', '').replace("'", '').replace(",", '')
            actor, action, obj = clean(row['actors']), clean(row['actions']), clean(row['objects'])
            narrative = " ".join(p for p in [actor, action, obj] if p)
            nar_list.append(narrative)

        embs = get_embedding(nar_list, verbose=False)
        res = similarity_economic_meaning(embs, examples)
        sim_df = pd.DataFrame(res, columns=[f'sim_{i+1}' for i in range(len(res[0]))])
        df_new = pd.concat([df.reset_index(drop=True), sim_df.reset_index(drop=True)], axis=1)
        path = file.replace(r'E:\Грант\Формализация', r'E:\Грант\Формализация и схожесть\Данные')
        df_new.to_excel(path, index=False)
        return file, 'OK'
    except Exception as e:
        return file, f"Error: {e}"


if __name__ == '__main__':
    files = xlsx_in_directory(r'E:\Грант\Формализация', '1997')
    # Нарративы-примеры
    narratives = get_example_narratives()
    # Тензоры нарративов-примеров
    examples = make_example_tensor(narratives)

    tasks = [file for file in files]

    with Pool(processes=8, initializer=init_worker, initargs=(examples,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="Подсчёт схожести"))
        # Печать статуса файлов
        # for file_path, status in results:
        #    print(file_path, status)
