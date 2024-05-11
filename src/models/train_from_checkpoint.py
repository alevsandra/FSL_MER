from torch.utils.data import DataLoader

from src.models.common_architecture import *
from src.data.make_dataset import JointDataset, EpisodeDataset
from backbone_model import Backbone
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

TRAIN_CLASSES = ['joy', 'power', 'surprise', 'sadness', 'bitterness', 'tenderness', 'transcendence']

TEST_CLASSES = ['fear', 'peace', 'tenderness', 'anger', 'tension']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_from_checkpoint(checkpoint_path, wandb_id):
    n_way = 5  # number of classes per episode
    n_support = 5  # number of support examples per class
    n_query = 20  # number of samples per class to use as query
    n_train_episodes = int(2000)  # number of episodes to generate for training
    n_val_episodes = 100  # number of episodes to generate for validation
    num_workers = 2  # number of workers to use for data loading

    train_data = JointDataset(False, TRAIN_CLASSES)

    val_data = JointDataset(False, TEST_CLASSES)

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

    train_loader = DataLoader(train_episodes, batch_size=None, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_episodes, batch_size=None, shuffle=False, num_workers=num_workers,
                            persistent_workers=True)

    checkpoint = torch.load(checkpoint_path)
    for key in checkpoint:
        print(key)

    backbone = Backbone()
    protonet = PrototypicalNet(backbone).to(DEVICE)
    protonet.load_state_dict(checkpoint["model_state_dict"])

    learner = FewShotLearner.load_from_checkpoint(checkpoint_path=checkpoint_path, protonet=protonet).to(DEVICE)
    learner.optimizer.load_state_dict(checkpoint["optimizer_states"][0])

    wandb_logger = WandbLogger(project='FSL_JointDataset',
                               id=wandb_id,
                               job_type='train',
                               log_model=True,
                               resume='must')
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/',
                                          monitor="step",
                                          mode="max",
                                          filename='joint-dataset-latest-{lr}-{step}',
                                          every_n_train_steps=1000)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=1,
        log_every_n_steps=1,
        val_check_interval=50,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        resume_from_checkpoint=checkpoint_path
    )

    trainer.state.global_step = checkpoint["global_step"]
    trainer.fit(learner, train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    chkpt_path = 'checkpoints/joint-dataset-latest-lr=0-step=2000-v2.ckpt'
    run_id = '75cytqfa'
    train_from_checkpoint(chkpt_path, run_id)
