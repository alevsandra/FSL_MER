from typing import Union
import umap

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import tqdm
import warnings
import plotly.express as px
from constants import *

from common_architecture import PrototypicalNet, FewShotLearner
from src.data.make_dataset import *
from backbone_model import Backbone, BackboneMTG

ROOT_DIR = os.path.split(os.environ['VIRTUAL_ENV'])[0]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def dim_reduce(
        embeddings: np.ndarray,
        n_components: int = 3,
        method: str = 'umap',
):
    """
    This function performs dimensionality reduction on a given set of embeddings.
    It can use either UMAP, t-SNE, or PCA for this purpose. The number of components
    to reduce the data to and the method used for reduction can be specified as arguments.
    It returns the projected embeddings as a NumPy array.

    Args:
        embeddings (np.ndarray): An array of embeddings, with shape (n_samples, n_features)
        n_components (int): The number of dimensions to reduce the embeddings to. Default: 3
        method (str): The method of dimensionality reduction to use.
                        One of 'umap', 'tsne', or 'pca'. Default: 'umap'

    Returns:
        proj (np.ndarray): The dimensionality-reduced embeddings, with shape (n_samples, n_components)
    """

    if method == 'umap':
        reducer = umap.UMAP(
            n_neighbors=5,
            n_components=n_components,
            metric='euclidean'
        )
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            init='pca',
            learning_rate='auto'
        )

    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f'dunno how to do {method}')

    proj = reducer.fit_transform(embeddings)

    return proj


def embedding_plot(
        proj: np.ndarray,
        color_labels: List[Union[int, str]],
        marker_labels: List[int] = None,
        title: str = ''
):
    """
    Plot a set of embeddings that have been reduced using dim_reduce.

    Args:
        proj: a numpy array of shape (n_samples, n_components)
        color_labels: a list of labels to color the points by
        marker_labels: a list of labels to use as markers
        title: the title of the plot

    Returns:
        a plotly figure object
    """

    n_components = proj.shape[-1]
    if n_components == 2:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            label=color_labels
        ))
        fig = px.scatter(
            df, x='x', y='y',
            color='label',
            title=title,
            symbol=marker_labels
        )
    elif n_components == 3:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            label=color_labels
        ))
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='label',
            symbol=marker_labels,
            title=title
        )
    else:
        raise ValueError(f"can only plot 2 or 3 components but got {n_components}")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return fig


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
            val_data = TrompaMer(False, TEST_CLASSES_TROMPA)
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

    metrics = {
        "accuracy": Accuracy(num_classes=n_way, task="multiclass").to(DEVICE)
    }

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

        # compute the metrics
        results = {}
        for name, metric in metrics.items():
            results[name] = metric(logits, query["target"])

        pbar.set_description(f"Episode {episode_idx} // Accuracy: {results['accuracy'].item():.3f}")

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
    final_results = {}
    for name, metric in metrics.items():
        final_results[name] = metric.compute()
    print(f"Total accuracy, averaged across all episodes: {final_results['accuracy']:.3f}")
    f = open(output, "a")
    f.write(checkpoint_path + "\n")
    for name, metric in final_results.items():
        f.write(f"Total {name}, averaged across all episodes: {metric:.3f}\n")
    f.close()

    # perform a TSNE over all embeddings in the test dataset
    embeddings = dim_reduce(
        embeddings=np.stack([d["embedding"] for d in embedding_table]),
        method="tsne",
        n_components=2,
    )

    # replace the original 512-dim embeddings with the 2-dim tsne embeddings
    # in our embedding table
    for entry, dim_reduced_embedding in zip(embedding_table, embeddings):
        entry["embedding"] = dim_reduced_embedding

    fig = embedding_plot(
        proj=np.stack([d["embedding"] for d in embedding_table]),
        color_labels=[d["label"] for d in embedding_table],
        marker_labels=[d["marker"] for d in embedding_table],
        title="Przestrzeń osadzenia zbioru " + dataset,
    )

    fig.show()


@click.command()
@click.option('--ckpt_files', default=["models/joint-dataset-lr=1e-5-step=7850.ckpt"],
              help='list of paths to checkpoint', multiple=True)
@click.option('--output', default='results.txt', help='path to output file')
def main(ckpt_files, output):
    torch.set_float32_matmul_precision('medium')
    warnings.filterwarnings("ignore", category=UserWarning)

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

        test(n_way, n_support, n_query, n_val_episodes, dataset, ckpt_path, output)


if __name__ == '__main__':
    main()
