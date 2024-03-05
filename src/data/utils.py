import numpy as np
from typing import Dict
import torch
import gdown
import os
import zipfile


def load_audio(path) -> Dict:
    y = torch.from_numpy(np.load(path)[:, :512])
    return {'audio': y}


def collate_list_of_dicts(input_set) -> Dict:
    dictionary = {}
    class_list_set = set()

    for item in input_set:
        # class_list_set.update(item['label'])  # in case of list in 'label
        class_list_set.add(item['label'])

    dictionary['classlist'] = list(class_list_set)
    dictionary['audio'] = torch.stack([item['audio'] for item in input_set])
    dictionary['target'] = torch.stack([item['target'] for item in input_set])
    return dictionary


def export_zip_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("../../data/raw")


def download_dataset(url, dataset_name, file_name, export):
    full_path = '../../data/external/' + dataset_name
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    gdown.download(url, full_path + "/" + file_name)
    if export:
        export_zip_file(full_path + "/" + file_name)


def open_csv(file_name):
    with open(file_name) as file:
        content = file.read().splitlines()
        for row in content:
            print(row)
            break


if __name__ == '__main__':
    data = load_audio('D:/magisterka-dane/' + '00/7400.npy')
    print(data['audio'].shape)
