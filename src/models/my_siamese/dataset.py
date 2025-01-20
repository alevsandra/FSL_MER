import random
from typing import List, Dict, Any, Tuple

import pandas as pd
from torch.utils.data import Dataset

from src.data.utils import *
from src.data.download_mtg import main_download
from src.data.mtg_jamendo_dataset.scripts import commons

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]
pme_mo_data_url = 'https://drive.google.com/uc?id=1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ'
pme_mo_readme_url = 'https://drive.google.com/uc?id=1KQ0zjRiBQynnHyVPU7DGpUWvtPmCBOcq'

TROMPA_spectrograms = 'https://drive.google.com/uc?id=1Vy2QroaGrkjH2ZjovHxmPFAcfUmxoqZL'
TROMPA_annotations = 'https://raw.githubusercontent.com/juansgomez87/vis-mtg-mer/main/data/summary.csv'

DEAM_audio = 'http://cvml.unige.ch/databases/DEAM/DEAM_audio.zip'
DEAM_annotations = 'http://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip'


class SiameseNetworkDataset(Dataset):
    def __init__(self, classes):
        self.classes = classes

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        This function selects two random items from the dataset based on the index. If the index is even,
        the two items are chosen from the same class. If the index is odd, the two items are chosen from different
        classes.
        """
        if index % 2 == 0:
            label = 1
            idx = random.sample(self.classes, 1)[0]
            sample_1 = self.get_sample(idx)
            sample_2 = self.get_sample(idx)
        else:
            label = 0
            idx1, idx2 = random.sample(self.classes, 2)
            sample_1 = self.get_sample(idx1)
            sample_2 = self.get_sample(idx2)
        return sample_1, sample_2, torch.from_numpy(np.array([label], dtype=np.float32))

    def get_sample(self, idx):
        """
        Extract specific dataset sample based on index.
        """
        raise NotImplementedError

    @property
    def class_list(self) -> List[str]:
        """
        The class_list property returns a list of class labels available in the dataset.
        This property enables users of the dataset to easily access a list of all the classes in the dataset.

        Returns:
            List[str]: A list of class labels available in the dataset.
        """
        raise NotImplementedError

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        """
        Returns a dictionary where the keys are class labels and the values are
        lists of indices in the dataset that belong to that class.
        This property enables users of the dataset to easily access
        examples that belong to specific classes.

        Returns:
            Dict[str, List[int]]: A dictionary mapping class labels to lists of dataset indices.
        """
        raise NotImplementedError


class MTGJamendo(SiameseNetworkDataset):
    def __init__(self, download, output_dir, input_file, class_file, classes):
        super().__init__(classes)
        if download:
            output = os.path.join(ROOT_DIR, output_dir)
            main_download(output)
        self.tracks, self.tags, self.extra = commons.read_file(input_file)
        self.class_file = class_file
        self.output_dir = output_dir

    def __len__(self):
        length = 0
        for k, v in self.tracks.items():
            for label in v['tags']:
                if label[13:] in self.classes:
                    length += 1
                    break
        return length

    def get_sample(self, class_id):
        sample_id = random.choice([*self.class_to_indices[class_id]])
        item = self.tracks[sample_id]
        path = os.path.join(ROOT_DIR, self.output_dir, item['path'].replace(".mp3", ".npy"))
        return load_mtg_melspectrogram(path)['audio']

    @property
    def class_list(self) -> List[str]:
        if self.classes is None:
            with open(self.class_file) as f:
                lines = f.read().splitlines()
            return lines
        else:
            return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        return self.tags['mood/theme']


class PMEmo(SiameseNetworkDataset):
    def __init__(self, download, classes, padding=False, augmentation=False):
        super().__init__(classes)
        if download:
            download_dataset(pme_mo_readme_url, "PMEmo", "README.txt", False, True)
            download_dataset(pme_mo_data_url, "PMEmo", "PMEmo2019.zip", True, True)
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

    def get_sample(self, class_id):
        sample_id = random.choice(self.class_to_indices[class_id])
        audio_path = os.path.join(ROOT_DIR, 'data/raw/PMEmo2019/chorus/', f"{sample_id}.mp3")
        return make_melspectrogram(audio_path, self.padding, self.augmentation)['audio']

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


class TrompaMer(SiameseNetworkDataset):
    def __init__(self, download, classes, padding=False, augmentation=False):
        super().__init__(classes)
        if download:
            download_dataset(TROMPA_annotations, "TROMPA_MER", "summary.csv", False, False)
            download_dataset(TROMPA_spectrograms, "TROMPA_MER", "spectrograms.zip", True, True)
        self.classes = classes
        self.padding = padding
        self.augmentation = augmentation
        self.annotations_csv = os.path.join(ROOT_DIR, 'data/external/TROMPA_MER/summary.csv')
        self.annotations = pd.read_csv(self.annotations_csv, index_col=0, sep='\t')
        for index, record in self.annotations.iterrows():
            self.annotations.at[index, 'label'] = assign_label(record['norm_energy'], record['norm_valence'], False)

    def __len__(self):
        filtered_annotations = self.annotations[self.annotations['label'].isin(self.class_list)]
        return filtered_annotations.shape[0]

    def get_sample(self, class_id):
        sample_id = random.choice(self.class_to_indices[class_id])
        annotations = self.annotations.loc[[sample_id]]
        track_name = annotations['cdr_track_num'].values[0]
        return load_melspectrogram('data/raw/spectrograms/' + str(track_name) + '-sample.npy', self.padding,
                                   self.augmentation)['audio']

    @property
    def class_list(self) -> List[str]:
        return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        class_indices = {}
        for label in self.class_list:
            items = self.annotations[self.annotations['label'] == label]
            class_indices[label] = items.index.values.tolist()
        return class_indices


class DEAM(SiameseNetworkDataset):
    def __init__(self, download, classes, padding=False, augmentation=False):
        super().__init__(classes)
        if download:
            download_dataset(DEAM_annotations, "DEAM", "DEAM_Annotations.zip", True, False)
            download_dataset(DEAM_audio, "DEAM", "DEAM_audio.zip", True, False)
        self.classes = classes
        self.padding = padding
        self.augmentation = augmentation

        arousal_annotations_path = ('data/raw/annotations/annotations averaged per song/dynamic (per second '
                                    'annotations)/arousal.csv')
        valence_annotations_path = ('data/raw/annotations/annotations averaged per song/dynamic (per second '
                                    'annotations)/valence.csv')
        self.arousal_annotations_csv = os.path.join(ROOT_DIR, arousal_annotations_path)
        self.valence_annotations_csv = os.path.join(ROOT_DIR, valence_annotations_path)
        self.annotations = pd.read_csv(self.arousal_annotations_csv, index_col=0)
        self.valence_annotations = pd.read_csv(self.valence_annotations_csv, index_col=0)
        self.annotations['arousal(mean)'] = self.annotations.mean(axis=1)
        self.valence_annotations['valence(mean)'] = self.valence_annotations.mean(axis=1)
        self.annotations['valence(mean)'] = self.valence_annotations['valence(mean)']
        for index, record in self.annotations.iterrows():
            self.annotations.at[index, 'label'] = assign_label(record['arousal(mean)'], record['valence(mean)'], False)

    def __len__(self):
        filtered_annotations = self.annotations[self.annotations['label'].isin(self.class_list)]
        return filtered_annotations.shape[0]

    def get_sample(self, class_id):
        sample_id = random.choice(self.class_to_indices[class_id])
        annotations = self.annotations.loc[[sample_id]]
        audio_path = os.path.join(ROOT_DIR, 'data/raw/MEMD_audio/DEAM_spectrograms/' + str(
            annotations.index.values[0]) + '.npy')
        return load_prepared_melspectrogram(audio_path, self.padding)['audio']

    @property
    def class_list(self) -> List[str]:
        return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        class_indices = {}
        for label in self.class_list:
            items = self.annotations[self.annotations['label'] == label]
            class_indices[label] = items.index.values.tolist()
        return class_indices


class JointDataset(SiameseNetworkDataset):
    def __init__(self, download, classes, padding=False):
        super().__init__(classes)
        self.pme_mo = PMEmo(download, classes, padding)
        self.trompa = TrompaMer(download, classes, padding)
        self.deam = DEAM(download, classes, padding)
        pme_mo_dict = pd.DataFrame(
            {'id': self.pme_mo.static_annotations['musicId'] + 100000, 'label': self.pme_mo.static_annotations['label'],
             'dataset': 'PMEmo'})
        trompa_dict = pd.DataFrame(
            {'id': self.trompa.annotations.index + 10000, 'label': self.trompa.annotations['label'],
             'dataset': 'TROMPA_MER'})
        deam_dict = pd.DataFrame(
            {'id': self.deam.annotations.index, 'label': self.deam.annotations['label'], 'dataset': 'DEAM'})
        self.annotations = pd.concat([pme_mo_dict, trompa_dict, deam_dict])
        self.classes = classes

    def __len__(self):
        annotations_filtered = self.annotations[self.annotations['label'].isin(self.class_list)]
        return annotations_filtered.shape[0]

    def __getitem__(self, index):
        annotations = self.annotations[self.annotations['id'] == index]
        match annotations['dataset'].values[0]:
            case 'PMEmo':
                return self.pme_mo[index - 100000]
            case 'TROMPA_MER':
                return self.trompa[index - 10000]
            case 'DEAM':
                return self.deam[index]

    @property
    def class_list(self) -> List[str]:
        return self.classes

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        class_indices = {}
        for label in self.class_list:
            items = self.annotations[self.annotations['label'] == label]
            class_indices[label] = items['id'].to_list()
        return class_indices


def main_pme():
    p = PMEmo(False, ['power', 'surprise', 'tension', 'sadness', 'tenderness', 'transcendence'])
    out = p[1]
    print(out)
    print(len(p))
    for key, item in p.class_to_indices.items():
        print(key, len(item))


def main_mtg():
    output_dir = 'D:/magisterka-dane'
    input_file = '../../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv'
    class_file_path = '../../mtg_jamendo_dataset/data/tags/moodtheme.txt'
    TRAIN_CLASSES = ['ambiental', 'background', 'ballad', 'calm', 'cool', 'dark', 'deep', 'dramatic', 'dream',
                     'emotional', 'energetic', 'epic', 'fast', 'fun', 'funny', 'groovy', 'happy', 'heavy', 'hopeful',
                     'horror', 'inspiring', 'love', 'meditative', 'melancholic', 'mellow', 'melodic', 'motivational',
                     'nature', 'party', 'positive', 'powerful', 'relaxing', 'retro', 'romantic', 'sad']
    dataset = MTGJamendo(False, output_dir, input_file, class_file_path, TRAIN_CLASSES)
    out = dataset[5]
    print(out)


def main_trompa():
    trompa = TrompaMer(False,
                       ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
                        'tenderness', 'transcendence'])
    print(len(trompa))
    out = trompa[1]
    print(out)
    for key, item in trompa.class_to_indices.items():
        print(key, len(item))


def main_deam():
    deam = DEAM(False,
                ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
                 'tenderness', 'transcendence'])
    out = deam[1]
    print(out)


def main_joint():
    joint = JointDataset(False,
                         ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
                          'tenderness', 'transcendence'])
    out = joint[100001]
    print(out)


if __name__ == '__main__':
    main_deam()
