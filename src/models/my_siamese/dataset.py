import random
from typing import List

import pandas as pd
from torch.utils.data import Dataset

from src.data.utils import *

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]
pme_mo_data_url = 'https://drive.google.com/uc?id=1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ'
pme_mo_readme_url = 'https://drive.google.com/uc?id=1KQ0zjRiBQynnHyVPU7DGpUWvtPmCBOcq'

TROMPA_spectrograms = 'https://drive.google.com/uc?id=1Vy2QroaGrkjH2ZjovHxmPFAcfUmxoqZL'
TROMPA_annotations = 'https://raw.githubusercontent.com/juansgomez87/vis-mtg-mer/main/data/summary.csv'

DEAM_audio = 'http://cvml.unige.ch/databases/DEAM/DEAM_audio.zip'
DEAM_annotations = 'http://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip'


class PMEmo(Dataset):
    def __init__(self, download, classes, padding=False, augmentation=False):
        if download:
            download_dataset(pme_mo_readme_url, "PMEmo", "README.txt", False, True)
            download_dataset(pme_mo_data_url, "PMEmo", "PMEmo2019.zip", True, True)
        self.classes = classes
        self.padding = padding
        self.augmentation = augmentation
        self.annotations_csv = os.path.join(ROOT_DIR, 'data/raw/PMEmo2019/annotations/', 'static_annotations.csv')
        self.static_annotations = pd.read_csv(self.annotations_csv)
        for index, record in self.static_annotations.iterrows():
            self.static_annotations.at[index, 'label'] = assign_label(record['Arousal(mean)'], record['Valence(mean)'],
                                                                      True)

    def __len__(self):
        filtered_annotations = self.static_annotations[self.static_annotations['label'].isin(self.class_list)]
        return filtered_annotations.shape[0]

    def __getitem__(self, index):
        def get_sample(class_id):
            sample_id = random.choice(self.class_to_indices[class_id])
            audio_path = os.path.join(ROOT_DIR, 'data/raw/PMEmo2019/chorus/', f"{sample_id}.mp3")
            return make_melspectrogram(audio_path, self.padding, self.augmentation)

        if index % 2 == 0:
            label = 1
            idx = random.sample(self.classes, 1)[0]
            sample_1 = get_sample(idx)
            sample_2 = get_sample(idx)
        else:
            label = 0
            idx1, idx2 = random.sample(self.classes, 2)
            sample_1 = get_sample(idx1)
            sample_2 = get_sample(idx2)
        return sample_1['audio'], sample_2['audio'], torch.from_numpy(np.array([label], dtype=np.float32))

    @property
    def class_list(self) -> List[str]:
        return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        class_indices = {}
        for label in self.class_list:
            items = self.static_annotations[self.static_annotations['label'] == label]
            class_indices[label] = items['musicId'].to_list()
        return class_indices


def main_pme():
    p = PMEmo(False, ['power', 'surprise', 'tension', 'sadness', 'tenderness', 'transcendence'])
    out = p[1]
    print(out)
    print(len(p))
    for key, item in p.class_to_indices.items():
        print(key, len(item))

if __name__ == '__main__':
    main_pme()