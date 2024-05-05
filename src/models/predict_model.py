import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from common_architecture import PrototypicalNet, FewShotLearner
from src.data.make_dataset import MTGJamendo, EpisodeDataset
from mtg_architecture import Backbone

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]

if __name__ == '__main__':
    checkpoint_path = os.path.join(ROOT_DIR, 'models/classic-firefly-33.ckpt')
    n_way = 5  # number of classes per episode
    n_support = 5  # number of support examples per class
    n_query = 20  # number of samples per class to use as query
    n_train_episodes = int(50000)  # number of episodes to generate for training
    n_val_episodes = 100  # number of episodes to generate for validation
    num_workers = 10  # number of workers to use for data loading

    TRAIN_CLASSES = [
        'happy',
        'film',
        'energetic',
        'relaxing',
        'emotional',
        'melodic',
        'dark',
        'epic',
        'dream',
        'love',
        'inspiring',
        'sad',
        'meditative',
        'advertising',
        'motivational',
        'deep',
        'romantic',
        'christmas',
        'documentary',
        'corporate',
        'positive',
        'summer',
        'space',
        'background',
        'fun',
        'melancholic',
        'commercial',
        'drama',
        'movie',
        'action',
        'ballad',
        'dramatic',
        'sport',
        'trailer',
        'party',
        'game',
        'nature',
        'cool',
        'powerful',
        'hopeful',
        'retro',
        'funny',
        'groovy',
        'holiday',
        'travel',
        'horror',
        'sexy',
        'fast',
        'slow',
        'upbeat',
        'heavy',
        'mellow',
        'uplifting',
        'adventure'
    ]

    TEST_CLASSES = [
        'soundscape',
        'soft',
        'ambiental',
        'calm',
        'children'
    ]

    val_data = MTGJamendo(False,
                          'autotagging_moodtheme',
                          'melspecs',
                          'mtg-fast',
                          'D:/magisterka-dane',
                          True,
                          True,
                          '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                          '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                          TEST_CLASSES)

    val_episodes = EpisodeDataset(
        dataset=val_data,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_val_episodes
    )

    val_loader = DataLoader(val_episodes, batch_size=None, num_workers=num_workers, persistent_workers=True)

    backbone = Backbone()
    protonet = PrototypicalNet(backbone)

    learner = FewShotLearner.load_from_checkpoint(checkpoint_path=checkpoint_path, protonet=protonet)
    learner.eval()

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=1,
        log_every_n_steps=1,
        val_check_interval=50
    )

    trainer.validate(model=learner, dataloaders=val_loader)
