# -*- coding: cp1251 -*-
##
from torch.utils.data import Dataset


class NarrativeDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        with open(self.paths[item], encoding='cp1251') as f:
            text = f.read()
        return text, self.paths[item]
