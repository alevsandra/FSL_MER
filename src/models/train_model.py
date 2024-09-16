import pytorch_lightning as pl
import torch.cuda
import wandb
from torch.utils.data import DataLoader
import warnings

from common_architecture import PrototypicalNet, FewShotLearner, FewShotNegativeLearner
from src.data.make_dataset import *
from backbone_model import Backbone, BackboneMTG
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from constants import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]


def train(download, n_way, n_support, n_query, n_train_episodes, n_val_episodes, dataset, lr_values, wandb_project,
          ckpt_filename, padding, augmentation=False, MUSIC=False):
    num_workers = 10  # number of workers to use for data loading

    match dataset:
        case "MTG":
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
        case "Joint":
            if augmentation:
                train_data = JointDataset(download, TRAIN_CLASSES_AUGMENTED, padding, augmentation)
                val_data = JointDataset(False, TEST_CLASSES_AUGMENTED, padding, augmentation)
            else:
                train_data = JointDataset(download, TRAIN_CLASSES, padding, augmentation)
                val_data = JointDataset(False, TEST_CLASSES, padding, augmentation)
        case "PMEmo":
            train_data = PMEmo(download, TRAIN_CLASSES_PMEMO, padding)
            val_data = PMEmo(False, TEST_CLASSES_PMEMO, padding)
        case "TROMPA":
            if n_way == 3:
                train_data = TrompaMer(download, TRAIN_CLASSES_TROMPA_3WAY, padding)
                val_data = TrompaMer(False, TRAIN_CLASSES_TROMPA_3WAY, padding)
            else:
                train_data = TrompaMer(download, TRAIN_CLASSES_TROMPA, padding)
                val_data = TrompaMer(False, TRAIN_CLASSES_TROMPA, padding)
        case "DEAM":
            if n_way == 3:
                train_data = DEAM(download, TRAIN_CLASSES_DEAM_3WAY, padding)
                val_data = DEAM(False, TEST_CLASSES_DEAM_3WAY, padding)
            else:
                train_data = DEAM(download, TRAIN_CLASSES, padding)
                val_data = DEAM(False, TEST_CLASSES, padding)
        case _:
            raise Exception("Wrong dataset name")

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

    train_loader = DataLoader(train_episodes, batch_size=None, num_workers=num_workers, persistent_workers=True)
    val_loader = DataLoader(val_episodes, batch_size=None, num_workers=num_workers, persistent_workers=True)

    for lr in lr_values:
        backbone = Backbone()
        if dataset == "MTG":
            backbone = BackboneMTG()
        protonet = PrototypicalNet(backbone).to(DEVICE)

        if MUSIC:
            path = os.path.join(ROOT_DIR, "models/model.ckpt")
            checkpoint = torch.load(path)
            protonet.load_state_dict(checkpoint["state_dict"], strict=False)
            learner = FewShotNegativeLearner(protonet, num_classes=n_way, learning_rate=lr, threshold=0.01).to(
                DEVICE)
        else:
            learner = FewShotLearner(protonet, num_classes=n_way, learning_rate=lr).to(DEVICE)

        wandb_logger = WandbLogger(project=wandb_project, job_type='train', log_model=True)
        wandb_logger.experiment.config.update({"k_way": n_way, "n_support": n_support, "n_query": n_query,
                                               "TRAIN_CLASSES": train_data.classes, "TEST_CLASSES": val_data.classes,
                                               "padding": padding, "augmentation": augmentation, "MUSIC": MUSIC})

        checkpoint_callback = ModelCheckpoint(dirpath='../../models/',
                                              monitor="loss/val",
                                              mode="min",
                                              filename=ckpt_filename + '-val_loss-{lr-Adam:.0E}-{step}')
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
    no_train_episodes = int(10000)  # number of episodes to generate for training
    no_val_episodes = 100  # number of episodes to generate for validation

    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings("ignore", category=UserWarning)

    # # mtg training
    # train(False, way, support, query, int(50000), no_val_episodes, "MTG", [3e-4, 2e-4],
    #       'FSL_MTG_Jamendo', 'mtg-jamendo', False, False)

    # pmemo training
    # train(False, 2, support, query, 1000, no_val_episodes, "PMEmo", [1e-5],
    #       'FSL_PMEmo', 'pmemo-padding', True, False)

    # TROMPA-MER training
    # train(False, way, support, query, 2000, no_val_episodes, "TROMPA", [1e-5, 1e-4, 1e-3],
    #       'FSL_TROMPA-MER', 'trompa-mer-padding', True, False)

    # DEAM training
    # train(False, way, support, query, 3000, no_val_episodes, "DEAM", [1e-5, 1e-4, 1e-3],
    #       'FSL_DEAM', 'deam-padding', True, False)

    # # joint train
    train(False, way, support, query, no_train_episodes, no_val_episodes, "Joint", [1e-5],
          'FSL_JointDataset', 'joint-dataset', False, True)
