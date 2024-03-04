from src.data.make_dataset import MTGJamendo, EpisodeDataset
from mtg_architecture import Backbone, PrototypicalNet

if __name__ == '__main__':
    sample_rate = 16000

    dataset = MTGJamendo(False,
                         'autotagging_moodtheme',
                         'melspecs',
                         'mtg-fast',
                         'D:/magisterka-dane',
                         True,
                         True,
                         '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                         '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                         None)

    episodes = EpisodeDataset(
        dataset,
        n_way=5,
        n_support=5,
        n_query=20,
        n_episodes=100,
    )

    support, query = episodes[0]
    episodes.print_episode(support, query)

    backbone = Backbone()
    protonet = PrototypicalNet(backbone)

    logits = protonet(support, query)
    print(f"got logits with shape {logits.shape}")
