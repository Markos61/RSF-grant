# -*- coding: cp1251 -*-
##
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import sys
sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from similarity_funcs import make_example_tensor, similarity_economic_meaning, get_embedding, get_example_narratives
from find_files import xlsx_in_directory


examples = None  # ���������� ���������� ��� ���������


def init_worker(example_tensors):
    global examples
    examples = example_tensors


def process_file(args):
    file = args
    try:
        df = pd.read_excel(file, engine='openpyxl')
        nar_list = []
        for _, row in df[['actors', 'actions', 'objects']].fillna('-').replace('[]', '-').iterrows():
            nar_list.extend([row['actors'].replace('[', '').replace(']', '').replace("'", '').replace(",", ''),
                             row['actions'].replace('[', '').replace(']', '').replace("'", '').replace(",", ''),
                             row['objects'].replace('[', '').replace(']', '').replace("'", '').replace(",", '')])

        embs = get_embedding(nar_list, verbose=False)
        res = similarity_economic_meaning(embs, examples)
        sim_df = pd.DataFrame(res, columns=[f'sim_{i+1}' for i in range(len(res[0]))])
        df_new = pd.concat([df.reset_index(drop=True), sim_df.reset_index(drop=True)], axis=1)
        path = file.replace(r'E:\�����\������������', r'E:\�����\������������ � ��������\������')
        df_new.to_excel(path, index=False)
        return file, 'OK'
    except Exception as e:
        return file, f"Error: {e}"


if __name__ == '__main__':
    files = xlsx_in_directory(r'E:\�����\������������', '')
    # ���������-�������
    narratives = get_example_narratives()
    # ������� ����������-��������
    examples = make_example_tensor(narratives)

    tasks = [file for file in files]

    with Pool(processes=6, initializer=init_worker, initargs=(examples,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="������� ��������"))
        # ������ ������� ������
        for file_path, status in results:
            print(file_path, status)
