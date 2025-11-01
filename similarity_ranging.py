# -*- coding: cp1251 -*-
##
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(r"E:\PycharmProjects\PARsing\grant")
from find_files import xlsx_in_directory


files = xlsx_in_directory(r'E:\�����\������������ � ��������\������', '')
dfs = []
filtered_dfs = []  # ��������� ������ ��� �������� ��������������� ������
narrative = 'sim_2'
for file in tqdm(files, desc='����� �������� �������'):
    df = pd.read_excel(file, engine='openpyxl')
    dfs.append(df)
    df_filtered = df[df[narrative] > 0.75]
    filtered_dfs.append(df_filtered)

all_dfs = pd.concat(dfs, ignore_index=True)

# ���������� �� � ���� ���������
filtered_df = pd.concat(filtered_dfs, ignore_index=True).sort_values(
    by=narrative, ascending=False
)
filtered_df.to_excel(fr'E:\�����\������������ � ��������\����������\��������_{narrative}.xlsx', index=False)

##
df_rynok = all_dfs[all_dfs['sentence'].astype(str).str.contains('����������', case=False, na=False)]

