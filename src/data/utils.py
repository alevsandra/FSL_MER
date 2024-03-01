import numpy as np
from typing import Dict
import torch


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


if __name__ == '__main__':
    data = load_audio('D:/magisterka-dane/' + '00/7400.npy')
    print(data['audio'].shape)
