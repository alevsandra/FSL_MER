import pytorch_lightning as pl
from torch.utils.data import DataLoader

from mtg_architecture import Backbone, PrototypicalNet, FewShotLearner
from src.data.make_dataset import MTGJamendo, EpisodeDataset


if __name__ == '__main__':
    sample_rate = 16000  # sample rate of the audio
    n_way = 5  # number of classes per episode
    n_support = 5  # number of support examples per class
    n_query = 20  # number of samples per class to use as query
    n_train_episodes = int(50000)  # number of episodes to generate for training
    n_val_episodes = 100  # number of episodes to generate for validation
    num_workers = 10  # number of workers to use for data loading

    TRAIN_INSTRUMENTS = [
        'mood/theme---ambiental',
        'mood/theme---background',
        'mood/theme---ballad',
        'mood/theme---calm',
        'mood/theme---cool',
        'mood/theme---dark',
        'mood/theme---deep',
        'mood/theme---dramatic',
        'mood/theme---dream',
        'mood/theme---emotional',
        'mood/theme---energetic',
        'mood/theme---epic',
        'mood/theme---fast',
        'mood/theme---fun',
        'mood/theme---funny',
        'mood/theme---groovy',
        'mood/theme---happy',
        'mood/theme---heavy',
        'mood/theme---hopeful',
        'mood/theme---horror',
        'mood/theme---inspiring',
        'mood/theme---love',
        'mood/theme---meditative',
        'mood/theme---melancholic',
        'mood/theme---mellow',
        'mood/theme---melodic',
        'mood/theme---motivational',
        'mood/theme---nature',
        'mood/theme---party',
        'mood/theme---positive',
        'mood/theme---powerful',
        'mood/theme---relaxing',
        'mood/theme---retro',
        'mood/theme---romantic',
        'mood/theme---sad',
    ]

    TEST_INSTRUMENTS = [
        'mood/theme---slow',
        'mood/theme---soft',
        'mood/theme---soundscape',
        'mood/theme---upbeat',
        'mood/theme---uplifting'
    ]

    train_data = MTGJamendo(False,
                            'autotagging_moodtheme',
                            'melspecs',
                            'mtg-fast',
                            'D:/magisterka-dane',
                            True,
                            True,
                            '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                            '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                            TRAIN_INSTRUMENTS)

    val_data = MTGJamendo(False,
                          'autotagging_moodtheme',
                          'melspecs',
                          'mtg-fast',
                          'D:/magisterka-dane',
                          True,
                          True,
                          '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                          '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                          TEST_INSTRUMENTS)

    train_episodes = EpisodeDataset(
        dataset=train_data,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_train_episodes
    )

    val_episodes = EpisodeDataset(
        dataset=val_data,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_val_episodes
    )

    train_loader = DataLoader(train_episodes, batch_size=None, num_workers=num_workers)
    val_loader = DataLoader(val_episodes, batch_size=None, num_workers=num_workers, persistent_workers=True)

    backbone = Backbone()
    protonet = PrototypicalNet(backbone)

    learner = FewShotLearner(protonet, num_classes=len(TRAIN_INSTRUMENTS))

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=1,
        log_every_n_steps=1,
        val_check_interval=50
    )

    trainer.fit(learner, train_loader, val_dataloaders=val_loader)
