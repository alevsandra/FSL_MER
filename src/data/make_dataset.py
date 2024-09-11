import click
import random
from torch.utils.data import Dataset
from src.data.download_mtg import main_download
from src.data.mtg_jamendo_dataset.scripts import commons
from typing import List, Dict, Any, Tuple
from src.data.utils import *
import pandas as pd

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]
pme_mo_data_url = 'https://drive.google.com/uc?id=1UzC3NCDj30j9Ba7i5lkMzWO5gFqSr0OJ'
pme_mo_readme_url = 'https://drive.google.com/uc?id=1KQ0zjRiBQynnHyVPU7DGpUWvtPmCBOcq'

TROMPA_spectrograms = 'https://drive.google.com/uc?id=1Vy2QroaGrkjH2ZjovHxmPFAcfUmxoqZL'
TROMPA_annotations = 'https://raw.githubusercontent.com/juansgomez87/vis-mtg-mer/main/data/summary.csv'

DEAM_audio = 'http://cvml.unige.ch/databases/DEAM/DEAM_audio.zip'
DEAM_annotations = 'http://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip'


class ClassConditionalDataset(Dataset):

    def __getitem__(self, index: int) -> Dict[Any, Any]:
        """
        Grab an item from the dataset. The item returned must be a dictionary.
        """
        raise NotImplementedError

    @property
    def class_list(self) -> List[str]:
        """
        The classlist property returns a list of class labels available in the dataset.
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

        Implement me!

        Returns:
            Dict[str, List[int]]: A dictionary mapping class labels to lists of dataset indices.
        """
        raise NotImplementedError


class EpisodeDataset(Dataset):
    """
        A dataset for sampling few-shot learning tasks from a class-conditional dataset.

    Args:
        dataset (ClassConditionalDataset): The dataset to sample episodes from.
        n_way (int): The number of classes to sample per episode.
            Default: 5.
        n_support (int): The number of samples per class to use as support.
            Default: 5.
        n_query (int): The number of samples per class to use as query.
            Default: 20.
        n_episodes (int): The number of episodes to generate.
            Default: 100.
    """

    def __init__(self,
                 dataset: ClassConditionalDataset,
                 n_way: int = 5,
                 n_support: int = 5,
                 n_query: int = 20,
                 n_episodes: int = 100):
        self.dataset = dataset
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

    def __getitem__(self, index: int) -> Tuple[Dict, Dict]:
        """Sample an episode from the class-conditional dataset.

        Each episode is a tuple of two dictionaries: a support set and a query set.
        The support set contains a set of samples from each of the classes in the
        episode, and the query set contains another set of samples from each of the
        classes. The class labels are added to each item in the support and query
        sets, and the list of classes is also included in each dictionary.

        Yields:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the support
            set and the query set for an episode.
        """
        # seed the random number generator so we can reproduce this episode
        rng = random.Random(index)

        # sample the list of classes for this episode
        episode_class_list = rng.sample(self.dataset.class_list, self.n_way)

        # sample the support and query sets for this episode
        support, query = [], []
        for c in episode_class_list:
            # grab the dataset indices for this class
            all_indices = self.dataset.class_to_indices[c]

            if not len(all_indices) > 0:
                continue

            # sample the support and query sets for this class
            indices = random.choices([*all_indices], k=self.n_support + self.n_query)
            items = [self.dataset[i] for i in indices]

            # add the class label to each item
            for item in items:
                item["target"] = torch.tensor(episode_class_list.index(c))
                item["label"] = c  # MTGJamendo items are multiclass, hence restriction to the class of interest

            # split the support and query sets
            support.extend(items[:self.n_support])
            query.extend(items[self.n_support:])

        # collate the support and query sets
        support = collate_list_of_dicts(support)
        query = collate_list_of_dicts(query)

        support["class_list"] = episode_class_list
        query["class_list"] = episode_class_list

        return support, query

    def __len__(self):
        return self.n_episodes

    def print_episode(self, support, query):
        """Print a summary of the support and query sets for an episode.

        Args:
            support (Dict[str, Any]): The support set for an episode.
            query (Dict[str, Any]): The query set for an episode.
        """
        print("Support Set:")
        print(f"  Class list: {support['class_list']}")
        print(f"  Audio Shape: {support['audio'].shape}")
        print(f"  Target Shape: {support['target'].shape}")
        print()
        print("Query Set:")
        print(f"  Class list: {query['class_list']}")
        print(f"  Audio Shape: {query['audio'].shape}")
        print(f"  Target Shape: {query['target'].shape}")


class MTGJamendo(ClassConditionalDataset):
    def __init__(self, download, outputdir, input_file, class_file, classes):
        if download:
            output = os.path.join(ROOT_DIR, outputdir)
            main_download(output)
        self.tracks, self.tags, self.extra = commons.read_file(input_file)
        self.class_file = class_file
        self.output_dir = outputdir
        self.classes = classes

    def __len__(self):
        length = 0
        for k, v in self.tracks.items():
            for label in v['tags']:
                if label[13:] in self.classes:
                    length += 1
                    break
        return length

    def __getitem__(self, index):
        item = self.tracks[index]
        path = os.path.join(ROOT_DIR, self.output_dir, item['path'].replace(".mp3", ".npy"))
        x = load_mtg_melspectrogram(path)
        x["label"] = item['tags']
        return x

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


class PMEmo(ClassConditionalDataset):
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
        annotations = self.static_annotations[self.static_annotations['musicId'] == index]
        audio_path = os.path.join(ROOT_DIR, 'data/raw/PMEmo2019/chorus/', str(index) + '.mp3')
        item = make_melspectrogram(audio_path, self.padding, self.augmentation)
        item['label'] = annotations['label'].values[0]
        return item

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


class TrompaMer(ClassConditionalDataset):
    def __init__(self, download, classes, padding=False, augmentation=False):
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

    def __getitem__(self, index):
        annotations = self.annotations.loc[[index]]
        track_name = annotations['cdr_track_num'].values[0]
        item = load_melspectrogram('data/raw/spectrograms/' + str(track_name) + '-sample.npy', self.padding,
                                   self.augmentation)
        item['label'] = annotations['label'].values[0]
        return item

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


class DEAM(ClassConditionalDataset):
    def __init__(self, download, classes, padding=False, augmentation=False):
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

    def __getitem__(self, index):
        annotations = self.annotations.loc[[index]]
        if self.augmentation:
            audio_path = os.path.join(ROOT_DIR, 'data/external/DEAM/DEAM_audio/MEMD_audio/' + str(annotations.index.values[0]) + '.mp3')
            item = make_melspectrogram(audio_path, self.padding, self.augmentation)
        else:
            audio_path = os.path.join(ROOT_DIR,
                                      'data/raw/MEMD_audio/DEAM_spectrograms/' + str(annotations.index.values[0]) + '.npy')
            item = load_prepared_melspectrogram(audio_path, self.padding)
        item['label'] = annotations['label'].values[0]
        return item

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


class AugmentedData(ClassConditionalDataset):
    def __init__(self, classes, padding=False, augmentation=False):
        self.classes = classes
        # FIXME: in case padding=True augmentation script needs to be modified not to cut spectrograms
        self.padding = padding
        self.augmentation = augmentation
        self.annotations_csv = os.path.join(ROOT_DIR, 'data/processed/augmentation_annotations.csv')
        self.annotations = pd.read_csv(self.annotations_csv, index_col=0)

    def __len__(self):
        filtered_annotations = self.annotations[self.annotations['label'].isin(self.class_list)]
        return filtered_annotations.shape[0]

    def __getitem__(self, index):
        annotations = self.annotations.loc[[index]]
        item = load_melspectrogram('data/processed/augmentation/' + str(index) + '.npy', self.padding,
                                   self.augmentation)
        item['label'] = annotations['label'].values[0]
        return item

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


class JointDataset(ClassConditionalDataset):
    def __init__(self, download, classes, padding=False, augmentation=False):
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
        if augmentation:
            self.augmented = AugmentedData(classes, padding)
            augmented_dict = pd.DataFrame(
                {'id': self.augmented.annotations.index, 'label': self.augmented.annotations['label'],
                 'dataset': 'Augmented'})
            self.annotations = pd.concat([pme_mo_dict, trompa_dict, deam_dict, augmented_dict])
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
            case _:
                return self.augmented[index]

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


@click.command()
@click.option('--download', default=False)
@click.option('--outputdir', default='D:/magisterka-dane', help='argument for download')
@click.argument('input_file', default='mtg_jamendo_dataset/data/autotagging_moodtheme.tsv')
@click.argument('class_file_path', default='mtg_jamendo_dataset/data/tags/moodtheme.txt')
def main_mtg(download, outputdir, input_file, class_file_path):
    TRAIN_CLASSES = ['ambiental', 'background', 'ballad', 'calm', 'cool', 'dark', 'deep', 'dramatic', 'dream',
                     'emotional', 'energetic', 'epic', 'fast', 'fun', 'funny', 'groovy', 'happy', 'heavy', 'hopeful',
                     'horror', 'inspiring', 'love', 'meditative', 'melancholic', 'mellow', 'melodic', 'motivational',
                     'nature', 'party', 'positive', 'powerful', 'relaxing', 'retro', 'romantic', 'sad']
    dataset = MTGJamendo(download, outputdir, input_file, class_file_path, TRAIN_CLASSES)
    print(len(dataset))
    print(dataset.class_list)
    # print(dataset[5])
    print(dataset.class_to_indices['romantic'])
    episodes = EpisodeDataset(
        dataset,
        n_way=5,
        n_support=5,
        n_query=20,
        n_episodes=100,
    )

    support, query = episodes[0]
    episodes.print_episode(support, query)


def main_pme():
    p = PMEmo(False, ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
                      'tenderness', 'transcendence'])
    print(p[1])
    print(len(p))
    for key, item in p.class_to_indices.items():
        print(key, len(item))


def main_trompa():
    trompa = TrompaMer(False,
                       ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
                        'tenderness', 'transcendence'])
    print(len(trompa))
    print(trompa[1])
    for key, item in trompa.class_to_indices.items():
        print(key, len(item))


if __name__ == '__main__':
    # main_pme()
    deam = JointDataset(False,
                        ['joy', 'power', 'surprise', 'anger', 'tension', 'fear', 'sadness', 'bitterness', 'peace',
                         'tenderness', 'transcendence'], False, False)
    for key, item in deam.class_to_indices.items():
        print(key, len(item))
