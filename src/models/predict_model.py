import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import tqdm

from common_architecture import PrototypicalNet, FewShotLearner
from src.data.make_dataset import *
from backbone_model import Backbone, BackboneMTG

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]

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

TEST_CLASSES_MTG = [
    'soundscape',
    'soft',
    'ambiental',
    'calm',
    'children'
]

TRAIN_CLASSES = ['joy', 'power', 'surprise', 'sadness', 'bitterness', 'tenderness', 'transcendence']

TEST_CLASSES = ['fear', 'peace', 'tenderness', 'anger', 'tension']

TRAIN_CLASSES_PMEMO = ['surprise',   'tension', 'sadness', 'transcendence']

TEST_CLASSES_PMEMO = ['power', 'tenderness']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model_from_ckpt(checkpoint_path, dataset):
    checkpoint = torch.load(checkpoint_path)
    backbone = Backbone()
    if dataset == "MTG":
        backbone = BackboneMTG()
    protonet = PrototypicalNet(backbone).to(DEVICE)
    protonet.load_state_dict(checkpoint["state_dict"], strict=False)

    learner = FewShotLearner.load_from_checkpoint(checkpoint_path=checkpoint_path, protonet=protonet).to(DEVICE)
    learner.optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    return learner


def predict(n_way, n_support, n_query, n_val_episodes, dataset, checkpoint_path):
    num_workers = 10  # number of workers to use for data loading

    match dataset:
        case "MTG":
            val_data = MTGJamendo(False,
                                  'D:/magisterka-dane',
                                  '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                                  '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                                  TEST_CLASSES_MTG)
        case "Joint":
            val_data = JointDataset(False, TEST_CLASSES)
        case "PMEmo":
            val_data = PMEmo(False, TEST_CLASSES_PMEMO)
        case "DEAM":
            val_data = DEAM(False, TEST_CLASSES)
        case "TROMPA":
            val_data = TrompaMer(False, TEST_CLASSES)
        case _:
            raise Exception("Wrong dataset name")

    val_episodes = EpisodeDataset(
        dataset=val_data,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
        n_episodes=n_val_episodes
    )

    val_loader = DataLoader(val_episodes, batch_size=None, num_workers=num_workers, persistent_workers=True)

    learner = get_model_from_ckpt(checkpoint_path, dataset)
    learner.eval()

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=1,
        log_every_n_steps=1,
        val_check_interval=50
    )

    trainer.validate(model=learner, dataloaders=val_loader)


def test(n_way, n_support, n_query, n_episodes, dataset, checkpoint_path, output):
    match dataset:
        case "MTG":
            test_episodes = EpisodeDataset(
                dataset=MTGJamendo(False,
                                   'D:/magisterka-dane',
                                   '../data/mtg_jamendo_dataset/data/autotagging_moodtheme.tsv',
                                   '../data/mtg_jamendo_dataset/data/tags/moodtheme.txt',
                                   TEST_CLASSES_MTG),
                n_way=n_way,
                n_support=n_support,
                n_query=n_query,
                n_episodes=n_episodes
            )
        case "Joint":
            test_episodes = EpisodeDataset(
                dataset=JointDataset(False, TEST_CLASSES),
                n_way=n_way,
                n_support=n_support,
                n_query=n_query,
                n_episodes=n_episodes
            )
        case "PMEmo":
            test_episodes = EpisodeDataset(
                dataset=PMEmo(False, TEST_CLASSES_PMEMO),
                n_way=n_way,
                n_support=n_support,
                n_query=n_query,
                n_episodes=n_episodes
            )
        case "DEAM":
            test_episodes = EpisodeDataset(
                dataset=DEAM(False, TEST_CLASSES),
                n_way=n_way,
                n_support=n_support,
                n_query=n_query,
                n_episodes=n_episodes
            )
        case "TROMPA":
            test_episodes = EpisodeDataset(
                dataset=TrompaMer(False, TEST_CLASSES),
                n_way=n_way,
                n_support=n_support,
                n_query=n_query,
                n_episodes=n_episodes
            )
        case _:
            raise Exception("Wrong dataset name")

    learner = get_model_from_ckpt(checkpoint_path, dataset)

    metric = Accuracy(num_classes=n_way, task="multiclass").to(DEVICE)

    # collect all the embeddings in the test set
    # so we can plot them later
    embedding_table = []
    pbar = tqdm.tqdm(range(len(test_episodes)))
    for episode_idx in pbar:
        support, query = test_episodes[episode_idx]

        # move all tensors to cuda if necessary
        batch_device(support, DEVICE)
        batch_device(query, DEVICE)

        # get the embeddings
        logits = learner.protonet(support, query)

        # compute the accuracy
        acc = metric(logits, query["target"])
        pbar.set_description(f"Episode {episode_idx} // Accuracy: {acc.item():.2f}")

        # add all the support and query embeddings to our records
        for subset_idx, subset in enumerate((support, query)):
            for emb, label in zip(subset["embeddings"], subset["target"]):
                embedding_table.append({
                    "embedding": emb.detach().cpu().numpy(),
                    "label": support["classlist"][label],
                    "marker": ("support", "query")[subset_idx],
                    "episode_idx": episode_idx
                })

        # also add the prototype embeddings to our records
        for class_idx, emb in enumerate(support["prototypes"]):
            embedding_table.append({
                "embedding": emb.detach().cpu().numpy(),
                "label": support["classlist"][class_idx],
                "marker": "prototype",
                "episode_idx": episode_idx
            })
    total_acc = metric.compute()
    print(f"Total accuracy, averaged across all episodes: {total_acc:.2f}")
    f = open(output, "a")
    f.write(checkpoint_path + "\n")
    f.write(f"Total accuracy, averaged across all episodes: {total_acc:.2f}\n")
    f.close()


@click.command()
@click.option('--ckpt_files', default=["models/joint-dataset-lr=1e-5-step=7850.ckpt"],
              help='list of paths to checkpoint', multiple=True)
@click.option('--output', default='results.txt', help='path to output file')
def main(ckpt_files, output):
    torch.set_float32_matmul_precision('medium')
    for ckpt_file in list(ckpt_files):
        ckpt_path = os.path.join(ROOT_DIR, ckpt_file)
        n_way = 5  # number of classes per episode
        n_support = 5  # number of support examples per class
        n_query = 20  # number of samples per class to use as query
        n_val_episodes = 100  # number of episodes to generate for validation

        dataset = "Joint"
        if "pmemo" in ckpt_path.casefold():
            dataset = "PMEmo"
            n_way = 2
        elif "mtg" in ckpt_path.casefold():
            dataset = "MTG"
        elif "deam" in ckpt_path.casefold():
            dataset = "DEAM"
        elif "trompa" in ckpt_path.casefold():
            dataset = "TROMPA"

        predict(n_way, n_support, n_query, n_val_episodes, dataset, ckpt_path)

        test(n_way, n_support, 15, 50, dataset, ckpt_path, output)


if __name__ == '__main__':
    main()
