import click
import random
import torch
from torch.utils.data import Dataset
from src.data.MTG_Jamendo.download import main_download
from src.data.mtg_jamendo_dataset.scripts import commons
from typing import List, Dict, Any, Tuple
from src.data.utils import load_audio, collate_list_of_dicts


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

            # sample the support and query sets for this class
            indices = rng.sample(all_indices, self.n_support + self.n_query)
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
    def __init__(self, download,
                 dataset, type, download_from, outputdir, unpack, remove,
                 input_file, class_file, instruments):
        if download:
            main_download(dataset, type, download_from, outputdir, unpack, remove)
        self.tracks, self.tags, self.extra = commons.read_file(input_file)
        self.class_file = class_file
        self.output_dir = outputdir
        self.instruments = instruments

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        item = self.tracks[index]
        data = load_audio(self.output_dir + "/" + item['path'].replace(".mp3", ".npy"))
        data["label"] = item['tags']
        return data

    @property
    def class_list(self) -> List[str]:
        if self.instruments is None:
            with open(self.class_file) as f:
                lines = f.read().splitlines()
            return lines
        else:
            return self.instruments

    @property
    def class_to_indices(self) -> Dict[str, List[int]]:
        class_indices = {}
        indices_set = set()
        for label in self.class_list:
            indices_set.clear()
            for key, value in self.tracks.items():
                if label in value['tags']:
                    indices_set.add(key)
            class_indices[label] = list(indices_set)
        return class_indices


@click.command()
@click.option('--download', default=False)
@click.option('--dataset', default='autotagging_moodtheme', help='argument for download')
@click.option('--type', default='melspecs', help='argument for download')
@click.option('--download_from', default='mtg-fast', help='argument for download')
@click.option('--outputdir', default='D:/magisterka-dane', help='argument for download')
@click.option('--unpack', default=True, help='argument for download')
@click.option('--remove', default=True, help='argument for download')
@click.argument('input_file', default='mtg_jamendo_dataset/data/autotagging_moodtheme.tsv')
@click.argument('class_file_path', default='mtg_jamendo_dataset/data/tags/moodtheme.txt')
def main(download, dataset, type, download_from, outputdir, unpack, remove, input_file, class_file_path):
    dataset = MTGJamendo(download, dataset, type, download_from, outputdir, unpack, remove, input_file, class_file_path,
                         None)
    print(len(dataset))
    print(dataset.class_list)
    # print(dataset[5])
    print(dataset.class_to_indices['mood/theme---uplifting'])
    episodes = EpisodeDataset(
        dataset,
        n_way=5,
        n_support=5,
        n_query=20,
        n_episodes=100,
    )

    support, query = episodes[0]
    episodes.print_episode(support, query)


if __name__ == '__main__':
    main()
