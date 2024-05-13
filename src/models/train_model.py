import pytorch_lightning as pl
import torch.cuda
import wandb
from torch.utils.data import DataLoader

from common_architecture import PrototypicalNet, FewShotLearner
from src.data.make_dataset import *
from backbone_model import Backbone, BackboneMTG
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

TRAIN_CLASSES_MTG = [
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
    'ambiental',
    'calm',
    'children',
    'adventure',
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
    'heavy',
    'mellow',
    'sexy',
    'fast'
]

TEST_CLASSES_MTG = [
    'slow',
    'soft',
    'soundscape',
    'upbeat',
    'uplifting'
]

TRAIN_CLASSES = ['joy', 'power', 'surprise', 'sadness', 'bitterness', 'tenderness', 'transcendence']

TEST_CLASSES = ['fear', 'peace', 'tenderness', 'anger', 'tension']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(download, n_way, n_support, n_query, n_train_episodes, n_val_episodes, dataset, lr_values, wandb_project,
          ckpt_filename):
    num_workers = 2  # number of workers to use for data loading
    sr = 44100  # sample rate of the audio

    if dataset == "MTG":
        train_data = MTGJamendo(download,
                                'D:/magisterka-dane',
                                '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                                '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                                TRAIN_CLASSES_MTG)
        val_data = MTGJamendo(False,
                              'D:/magisterka-dane',
                              '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                              '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                              TEST_CLASSES_MTG)
    elif dataset == "Joint":
        train_data = JointDataset(download, TRAIN_CLASSES)
        val_data = JointDataset(False, TEST_CLASSES)
    elif dataset == "PMEmo":
        train_data = PMEmo(download, TRAIN_CLASSES)
        val_data = PMEmo(False, TEST_CLASSES)

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

    for lr in lr_values:
        backbone = Backbone()
        if dataset == "MTG":
            backbone = BackboneMTG()
        protonet = PrototypicalNet(backbone).to(DEVICE)

        learner = FewShotLearner(protonet, num_classes=n_way, learning_rate=lr).to(DEVICE)

        wandb_logger = WandbLogger(project=wandb_project, job_type='train', log_model=True)
        checkpoint_callback = ModelCheckpoint(dirpath='models/',
                                              monitor="loss/val",
                                              mode="min",
                                              filename=ckpt_filename + '-val_loss-{learning_rate}-{step}')
        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=1,
            log_every_n_steps=1,
            val_check_interval=50,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor]
        )

        trainer.fit(learner, train_loader, val_dataloaders=val_loader)
        torch.cuda.empty_cache()
        wandb.finish()


if __name__ == '__main__':
    way = 5  # number of classes per episode
    support = 5  # number of support examples per class
    query = 20  # number of samples per class to use as query
    no_train_episodes = int(5000)  # number of episodes to generate for training
    no_val_episodes = 100  # number of episodes to generate for validation

    torch.set_float32_matmul_precision('medium')

    # mtg training
    # train(False, way, support, query, int(50000), val_episodes, "MTG", [3e-4, 2e-4],
    #       'FSL_MTG_Jamendo', 'mtg-jamendo')

    # pmemo training
    # train(False, way, support, query, train_episodes, val_episodes, "PMEmo", [1e-4, 1e-3],
    #       'FSL_PMEmo', 'pmemo')

    # joint train
    train(False, way, support, query, no_train_episodes, no_val_episodes, "Joint", [1e-5],
          'FSL_JointDataset', 'joint-dataset')
