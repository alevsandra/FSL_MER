import pytorch_lightning as pl
from torch.utils.data import DataLoader

from common_architecture import PrototypicalNet, FewShotLearner
from src.data.make_dataset import MTGJamendo, EpisodeDataset, PMEmo
from mtg_architecture import Backbone


def train_mtg():
    sample_rate = 16000  # sample rate of the audio
    n_way = 5  # number of classes per episode
    n_support = 5  # number of support examples per class
    n_query = 20  # number of samples per class to use as query
    n_train_episodes = int(50000)  # number of episodes to generate for training
    n_val_episodes = 100  # number of episodes to generate for validation
    num_workers = 10  # number of workers to use for data loading

    TRAIN_INSTRUMENTS = [
        'ambiental',
        'background',
        'ballad',
        'calm',
        'cool',
        'dark',
        'deep',
        'dramatic',
        'dream',
        'emotional',
        'energetic',
        'epic',
        'fast',
        'fun',
        'funny',
        'groovy',
        'happy',
        'heavy',
        'hopeful',
        'horror',
        'inspiring',
        'love',
        'meditative',
        'melancholic',
        'mellow',
        'melodic',
        'motivational',
        'nature',
        'party',
        'positive',
        'powerful',
        'relaxing',
        'retro',
        'romantic',
        'sad',
    ]

    TEST_INSTRUMENTS = [
        'slow',
        'soft',
        'soundscape',
        'upbeat',
        'uplifting'
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
